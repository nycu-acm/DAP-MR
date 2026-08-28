import os
import gc
import random
import lpips
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path

import diffusers
from diffusers.optimization import get_scheduler

from pisasr import CSDLoss, PiSASR, active_inactive_ranking_loss
from src.my_utils.training_utils import parse_args
from src.datasets.dataset import PairedSROnlineTxtDataset
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def run_finetune_stage(args):
    """v2-Mix fine-tune (2026-07-16 follow-up to the mixed-degradation + oracle-router
    audit, memory pisasr-dapmr-phase0-audit): a dedicated 2-stage post-hoc fine-tune,
    resumed from an already-trained checkpoint (--resume_ckpt), replacing the normal
    from-scratch Phase 1 (router pretrain) / Phase 2 (pix-then-sem) loop entirely.

    Stage 1 "controller": freeze router + LoRA, train ONLY the degradation-prompt
    directions/scale (text_prototypes/restoration_tokens/restoration_token_scale) via
    mixed-degradation reconstruction with the prompt fed DIRECTLY from ground truth
    (beta=1.0, router never called) — isolates and fixes token/controller interaction
    without router error confounding the signal.

    Stage 2 "router": freeze LoRA + degradation-prompt, train ONLY the router via a
    dose-regression + active/inactive-ranking loss, PLUS a reconstruction loss whose
    conditioning is annealed from ground truth to the router's own (now-gradient-attached)
    prediction over --teacher_forcing_anneal_frac of training — trains the router to be
    calibrated on mixed-degradation inputs specifically, using the same synthetic curriculum
    Stage 1 used.

    Deliberately NOT unfreezing both stages' parameter groups at once (matches the user's
    explicit ask): running them as two separate invocations lets an audit
    (scripts/audit/phase1_mixed_oracle_audit.py) attribute improvement to the right stage.
    """
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    assert args.resume_ckpt is not None, (
        "--finetune_stage requires --resume_ckpt pointing at the base checkpoint "
        "(e.g. dapmr_specificity_fix_v2_esrgan_physic's model_8500.pkl).")
    stage = args.finetune_stage
    assert stage in ("controller", "router")

    net_pisasr = PiSASR(args)
    if args.enable_xformers_memory_efficient_attention:
        net_pisasr.unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    _pix_sem = ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                'default_encoder_sem', 'default_decoder_sem', 'default_others_sem']
    net_pisasr.unet.set_adapter(_pix_sem)

    unwrapped = accelerator.unwrap_model(net_pisasr)
    lora_params = [p for n, p in net_pisasr.unet.named_parameters() if "lora" in n]
    router_params = unwrapped.router_parameters()
    degrad_prompt_params = [p for p in
                             (unwrapped.text_prototypes, unwrapped.restoration_tokens,
                              unwrapped.restoration_token_scale) if p is not None]

    for p in lora_params:
        p.requires_grad = False
    if stage == "controller":
        for p in router_params:
            p.requires_grad = False
        for p in degrad_prompt_params:
            p.requires_grad = True
        trainable_params = degrad_prompt_params
        unwrapped.set_router_eval()
    else:  # router
        for p in degrad_prompt_params:
            p.requires_grad = False
        for p in router_params:
            p.requires_grad = True
        trainable_params = router_params
        unwrapped.set_router_train()
    net_pisasr.unet.train()

    opt = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_train.total_steps = args.max_train_steps
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size,
                                            shuffle=True, num_workers=args.dataloader_num_workers)

    net_pisasr, opt, dl_train, lr_scheduler = accelerator.prepare(net_pisasr, opt, dl_train, lr_scheduler)
    net_lpips = accelerator.prepare(net_lpips)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(args.max_train_steps), desc=f"Finetune[{stage}]",
                         disable=not accelerator.is_local_main_process)
    global_step = 0
    dl_iter = iter(dl_train)
    while global_step < args.max_train_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl_train)
            batch = next(dl_iter)
        x_tgt = batch["output_pixel_values"]

        with accelerator.accumulate(net_pisasr):
            args._current_step = global_step
            um = accelerator.unwrap_model(net_pisasr)
            logs = {}

            if stage == "controller":
                pred, _, _ = um.compute_mixed_curriculum_sr_loss(x_tgt, args, beta=1.0)
                loss_sr = (F.mse_loss(pred.float(), x_tgt.float()) * args.lambda_l2
                           + net_lpips(pred.float(), x_tgt.float()).mean() * args.lambda_lpips)
                total_loss = loss_sr
                logs["loss_sr"] = loss_sr.item()
            else:  # router
                beta = um._teacher_forcing_alpha(args)
                pred, labels_syn, router_pred = um.compute_mixed_curriculum_sr_loss(
                    x_tgt, args, beta=beta, detach_router=False)
                loss_sr = (F.mse_loss(pred.float(), x_tgt.float()) * args.lambda_l2
                           + net_lpips(pred.float(), x_tgt.float()).mean() * args.lambda_lpips)
                loss_dose = F.smooth_l1_loss(router_pred, labels_syn) * args.lambda_dose
                loss_rank = (active_inactive_ranking_loss(router_pred, labels_syn, margin=args.rank_margin)
                             * args.lambda_rank if args.lambda_rank > 0
                             else torch.zeros((), device=x_tgt.device))
                total_loss = loss_sr + loss_dose + loss_rank
                logs.update({"loss_sr": loss_sr.item(), "loss_dose": loss_dose.item(),
                             "loss_rank": loss_rank.item() if torch.is_tensor(loss_rank) else loss_rank,
                             "beta": beta})

            _lambda_spec = getattr(args, "lambda_specificity", 0.0)
            if _lambda_spec > 0:
                spec_pred, spec_tgt = um.compute_specificity_batch_loss(x_tgt, args)
                loss_spec = F.l1_loss(spec_pred.float(), spec_tgt.float())
                total_loss = total_loss + loss_spec * _lambda_spec
                logs["loss_spec"] = loss_spec.item()

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                opt.step(); lr_scheduler.step(); opt.zero_grad(set_to_none=True)
                progress_bar.update(1)
                global_step += 1
                logs["loss"] = total_loss.detach().item()
                if accelerator.is_main_process:
                    progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                    outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                    accelerator.unwrap_model(net_pisasr).save_model(outf)

    print(f"Finetune stage {stage!r} completed.")


def main(args):
    if getattr(args, "finetune_stage", "none") != "none":
        run_finetune_stage(args)
        return

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pisasr = PiSASR(args)
    if args.enable_xformers_memory_efficient_attention:
        net_pisasr.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    _pix_sem = ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                'default_encoder_sem', 'default_decoder_sem', 'default_others_sem']
    net_pisasr.unet.set_adapter(_pix_sem)
    net_pisasr.set_train_pix()

    # Collect parameter groups
    unwrapped = accelerator.unwrap_model(net_pisasr)

    lora_params = [p for n, p in net_pisasr.unet.named_parameters() if "lora" in n]

    router_params = unwrapped.router_parameters()

    # Degradation prompt (text_prototypes / restoration_tokens / restoration_token_scale):
    # frozen by default (fixed at text-encoder-derived init for the whole run). When
    # --degrad_prompt_learnable, trained jointly with the LoRA weights throughout Phase 2 via
    # its own always-on optimizer group (same treatment as domain_prompt_params below) — NOT
    # phase-gated with router_params. Note: text_prototypes itself never receives gradient
    # either way (it's a dead parameter after __init__, only used to seed restoration_tokens'
    # initial value via .clone() once) — kept in the group for symmetry/checkpoint clarity,
    # harmless either way (optimizer skips params with grad=None). text_prototypes is None
    # under --lgdp_variable_tokens (no single pooled token to keep) — filtered out here so
    # AdamW never sees a None entry.
    degrad_prompt_learnable = getattr(args, "degrad_prompt_learnable", False)
    degrad_prompt_params = [p for p in
                             (unwrapped.text_prototypes, unwrapped.restoration_tokens,
                              unwrapped.restoration_token_scale) if p is not None]
    if not degrad_prompt_learnable:
        for p in degrad_prompt_params:
            p.requires_grad_(False)

    # Domain prompt (--domain_prompt) IS trained — unlike the degradation prompt above, it's
    # not phase-gated with the router either: it gets gradient from the same forward pass as
    # lora_params (it's part of dynamic_prompt -> UNet -> total_loss) every Phase-2 step, so it
    # gets its own always-on optimizer group, separate from router_params' pretrain/freeze
    # schedule. A no-op when --domain_prompt is unset ("" -> 0-length content, no forward
    # contribution, so grad stays None and the has_grad_domain check below skips the step).
    domain_prompt_params = [unwrapped.domain_prompt_content, unwrapped.domain_prompt_eos]

    # Khởi tạo Optimizer
    opt_unet = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    opt_router = torch.optim.AdamW(router_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    opt_domain_prompt = torch.optim.AdamW(domain_prompt_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    opt_degrad_prompt = torch.optim.AdamW(degrad_prompt_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay)

    # Đưa vào prepare
    lr_scheduler_unet = get_scheduler(
        args.lr_scheduler, optimizer=opt_unet,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power
    )
    lr_scheduler_router = get_scheduler(
        args.lr_scheduler, optimizer=opt_router,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power
    )
    lr_scheduler_domain_prompt = get_scheduler(
        args.lr_scheduler, optimizer=opt_domain_prompt,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power
    )
    lr_scheduler_degrad_prompt = get_scheduler(
        args.lr_scheduler, optimizer=opt_degrad_prompt,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power
    )

    def set_requires_grad(params_list, requires_grad):
        for p in params_list:
            p.requires_grad = requires_grad

    # initialize the dataset
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dataset_train.total_steps = args.max_train_steps  # drives domain_rand_anneal_frac schedule
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    (net_pisasr, opt_unet, opt_router, opt_domain_prompt, opt_degrad_prompt, dl_train,
     lr_scheduler_unet, lr_scheduler_router, lr_scheduler_domain_prompt, lr_scheduler_degrad_prompt) = accelerator.prepare(
        net_pisasr, opt_unet, opt_router, opt_domain_prompt, opt_degrad_prompt, dl_train,
        lr_scheduler_unet, lr_scheduler_router, lr_scheduler_domain_prompt, lr_scheduler_degrad_prompt
    )
    net_lpips = accelerator.prepare(net_lpips)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    if not hasattr(args, 'default_null_text_ratio'):
        args.default_null_text_ratio = getattr(args, 'null_text_ratio', 0.5)

    # Spec C.13 (null_text_ratio conflict): the controllable-SR loss trains the model so
    # that an ALL-ZERO prompt means "user requested nothing removed -> output stays
    # degraded (== input)". The main forward's CFG-style dropout (null_text_ratio) ALSO
    # feeds an all-zero prompt, but there it means "unconditional -> output should still be
    # fully clean (== x_tgt)". Training both meanings onto the same shared weights is a
    # direct contradiction. Auto-disable the CFG dropout whenever controllable-SR training
    # is active rather than requiring every launch script to remember --null_text_ratio 0.
    if getattr(args, "lambda_specificity", 0.0) > 0 and args.default_null_text_ratio > 0:
        accelerator.print(
            f"[null_text_ratio] forcing 0.0 (was {args.default_null_text_ratio}) — "
            f"lambda_specificity={args.lambda_specificity} > 0 means an all-zero prompt "
            f"already has a meaning (\"remove nothing\", e.g. the matched case with "
            f"strength=0 -> target=degraded input unchanged); CFG-style unconditional "
            f"dropout would teach the same all-zero prompt the opposite meaning "
            f"(\"output fully clean\") on the same weights. See spec C.13 / memory "
            f"pisasr-dapmr-phase0-audit.")
        args.default_null_text_ratio = 0.0

    # =========================================================================
    # PHASE 1: PRE-TRAIN ROUTER VỚI PSEUDO-LABELS (SELF-SUPERVISED)
    # =========================================================================
    force_router_intensity = getattr(args, "force_router_intensity", None)
    if force_router_intensity is not None:
        # "No router" ablation: forward() bypasses the router entirely and uses a fixed
        # constant intensity for every image — nothing to pretrain, router_params stay
        # frozen and unused for the whole run.
        accelerator.print(f"⏭️  Bỏ qua Phase 1 (router pretrain): --force_router_intensity="
                           f"{force_router_intensity} — router bị bypass hoàn toàn trong forward().")
        set_requires_grad(router_params, False)
    elif getattr(args, 'use_degradation', True):
        accelerator.print("🚀 BẮT ĐẦU PHASE 1: PRE-TRAIN ROUTER...")
        
        num_pretrain_steps = getattr(args, 'router_pretrain_steps', 1000)

        set_requires_grad(lora_params, False)
        set_requires_grad(router_params, True)

        unwrapped = accelerator.unwrap_model(net_pisasr)
        unwrapped.unet.eval()
        unwrapped.set_router_train()

        pretrain_pbar = tqdm(range(num_pretrain_steps), desc="Phase 1: Router Pre-train", disable=not accelerator.is_local_main_process)
        step_router = 0
        dl_train_iter = iter(dl_train)

        while step_router < num_pretrain_steps:
            try:
                batch = next(dl_train_iter)
            except StopIteration:
                dl_train_iter = iter(dl_train)
                batch = next(dl_train_iter)

            with accelerator.accumulate(net_pisasr):
                loss_router = unwrapped.compute_router_loss(
                    batch["output_pixel_values"], c_src=batch["conditioning_pixel_values"]
                )

                # Counterfactual router loss (opt-in): same HR content, two DIFFERENT physics
                # degradations synthesized on the fly -> router must reproduce the LABEL
                # DIFFERENCE between the two views. Content cancels out (it's identical in
                # both views), so only degradation evidence can explain the difference —
                # pushes the router toward degradation-evidence features instead of
                # content-correlated shortcuts, which fits the router's motivation better
                # than the absolute-label regression above on its own.
                lambda_cf = getattr(args, "lambda_cf_router", 0.0)
                loss_cf_val = None
                if lambda_cf > 0:
                    loss_cf = unwrapped.compute_counterfactual_router_loss(batch["output_pixel_values"])
                    loss_router = loss_router + lambda_cf * loss_cf
                    loss_cf_val = loss_cf.item()

                accelerator.backward(loss_router)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(router_params, args.max_grad_norm)
                    opt_router.step()
                    lr_scheduler_router.step()
                    opt_router.zero_grad()
                    pretrain_pbar.update(1)
                    step_router += 1
                    if step_router % 100 == 0:
                        _cf_msg = f" cf_loss={loss_cf_val:.4f}" if loss_cf_val is not None else ""
                        accelerator.print(f"  [Router pretrain {step_router}/{num_pretrain_steps}] loss={loss_router.item():.4f}{_cf_msg}")

        accelerator.print("✅ HOÀN THÀNH PHASE 1: MẠNG ROUTER ĐÃ HỘI TỤ!")
        # Router is pretrained ONLY here — freeze it (and set eval() so its dropout doesn't
        # add noise) for the rest of training. Phase 2 never touches router_params again.
        set_requires_grad(router_params, False)
        unwrapped.set_router_eval()
        unwrapped.unet.train()
        # Phase 1 froze ALL lora_params (pix+sem) via set_requires_grad(lora_params, False)
        # above; restore PiSA's pix-first schedule for the start of Phase 2 (was otherwise
        # never re-enabled, silently skipping the pix phase entirely).
        unwrapped.set_train_pix()
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: START MAIN TRAINING LOOP (PiSASR)
    # =========================================================================
    global_step = 0
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0
    lambda_csd = 0
    router_finetune_active = False
    router_finetune_steps = getattr(args, "router_finetune_steps", 0)
    if args.resume_ckpt is not None:
        args.pix_steps = 1
    for _ in range(0, args.num_training_epochs):
        for batch in dl_train:
            with accelerator.accumulate(net_pisasr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                
                if global_step == args.pix_steps:
                    _sem_adapters = [
                        'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                        'default_encoder_sem', 'default_decoder_sem', 'default_others_sem']
                    _unwrapped_pix = accelerator.unwrap_model(net_pisasr)
                    _unwrapped_pix.unet.set_adapter(_sem_adapters)
                    _unwrapped_pix.set_train_sem()
                    lambda_l2    = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd   = args.lambda_csd

                # requires_grad for pix vs sem params is already set by set_train_pix()/
                # set_train_sem() above; do NOT force-enable all lora_params here, or it
                # silently undoes PiSA's two-phase pix-then-sem discipline (sem would start
                # training early, and pix would never actually freeze after the switch).
                # (domain_prompt_params, if --domain_prompt is set, train continuously here
                # via their own always-on optimizer group — see domain_prompt_params above.)

                # Router fine-tune window: the router was pretrained ONLY on synthetic
                # degradations (Phase 1) then frozen — it may be miscalibrated on real LR
                # statistics it never saw. For the last `router_finetune_steps` steps, unfreeze
                # it so gradients from the real-data SR reconstruction loss below (which
                # already flow through intensities_pred -> dynamic_prompt -> model_pred -> loss
                # regardless of requires_grad) actually update it — letting it adapt to the real
                # distribution via the true downstream task instead of only the synthetic proxy.
                if (not router_finetune_active and router_finetune_steps > 0
                        and global_step >= args.max_train_steps - router_finetune_steps):
                    router_finetune_active = True
                    _um_rf = accelerator.unwrap_model(net_pisasr)
                    set_requires_grad(router_params, True)
                    _um_rf.set_router_train()
                    accelerator.print(f"[router_finetune] unfrozen at step {global_step} "
                                       f"(last {router_finetune_steps} steps, jointly with LoRA)")

                # Set CFG Ratio
                args.null_text_ratio = args.default_null_text_ratio
                # Drives PiSASR._teacher_forcing_alpha's anneal schedule (spec C.9).
                args._current_step = global_step

                # ====================================================
                # SYNTHETIC DEGRADATION-MAGNITUDE-SUPERVISED SR LOSS (backward separately)
                # ====================================================
                # Synthesizes a degraded LR view of x_tgt with a KNOWN 6-dim label and runs
                # it through the SAME LoRA UNet, weighting the degradation prompt by that
                # ground-truth label (not the router's own guess). This gives the LoRA a
                # direct, correctly-labeled magnitude -> output gradient that real-data
                # training alone (prompt weight always = router's own prediction) can't
                # provide. Shares lora_params with the main DDP forward below, so — like the
                # old router aux loss — it must be backwarded SEPARATELY through the
                # unwrapped model first, or DDP's "mark variable ready twice" guard trips.
                _lambda_synth_sr = getattr(args, "lambda_synth_sr", 0.0)
                if (_lambda_synth_sr > 0
                        and getattr(args, "use_degradation", True)
                        and random.random() < getattr(args, "synth_sr_ratio", 0.5)):
                    _um_synth = accelerator.unwrap_model(net_pisasr)
                    synth_pred = _um_synth.compute_synthetic_sr_loss(x_tgt, args)
                    loss_synth_sr = F.mse_loss(synth_pred.float(), x_tgt.float())
                    accelerator.backward(loss_synth_sr * _lambda_synth_sr)
                    _synth_sr_loss_val = loss_synth_sr.item()
                else:
                    _synth_sr_loss_val = None

                # ====================================================
                # SPECIFICITY BATCH LOSS (backward separately, same DDP reason)
                # ====================================================
                # Cheap single-forward matched/mismatched specificity + continuous partial-
                # restoration loss (spec C.8/C.10) — replaces the old separate controllable-SR
                # + match/mismatch branches (up to 4 UNet forward/backward passes per step)
                # with ONE mixed batch (each sample independently matched or mismatched), so
                # this costs exactly one extra forward pass. Directly targets the Phase 0
                # audit's root-cause finding: one restoration token (attenuation) fires at a
                # near-constant level regardless of which degradation is actually present,
                # drowning out other well-calibrated slots.
                # --specificity_window_start/_end (2026-07-25, user request): forward_curriculum
                # ALREADY trains "matched, full-severity" behavior every step via its own
                # ground-truth label (beta=1 until the teacher-forcing anneal near the end) --
                # so firing specificity's matched/mismatch probe on EVERY step for the whole
                # run duplicates that signal at the cost of a full extra forward/backward each
                # time. When the window is set, specificity fires ONLY for global_step in
                # [start, end) and REPLACES the main forward_curriculum step entirely that
                # iteration (see _skip_main_forward below) -- outside the window it's off.
                # Unset (default -1) falls back to the legacy always-on-every-step behavior
                # (gated only by --specificity_batch_ratio), running ALONGSIDE the main loss.
                _spec_win_start = getattr(args, "specificity_window_start", -1)
                _spec_win_end   = getattr(args, "specificity_window_end", -1)
                _spec_windowed  = _spec_win_start >= 0 and _spec_win_end >= 0
                if _spec_windowed:
                    _spec_fire = _spec_win_start <= global_step < _spec_win_end
                else:
                    _spec_fire = random.random() < getattr(args, "specificity_batch_ratio", 1.0)

                _lambda_spec = getattr(args, "lambda_specificity", 0.0)
                if (_lambda_spec > 0
                        and getattr(args, "use_degradation", True)
                        and _spec_fire):
                    _um_spec = accelerator.unwrap_model(net_pisasr)
                    spec_pred, spec_tgt = _um_spec.compute_specificity_batch_loss(x_tgt, args)
                    loss_specificity = F.l1_loss(spec_pred.float(), spec_tgt.float())
                    _lambda_lpips_spec = getattr(args, "lambda_lpips_specificity", 0.0)
                    if _lambda_lpips_spec > 0:
                        loss_specificity = loss_specificity + net_lpips(spec_pred.float(), spec_tgt.float()).mean() * _lambda_lpips_spec
                    accelerator.backward(loss_specificity * _lambda_spec)
                    _spec_loss_val = loss_specificity.item()
                else:
                    _spec_loss_val = None

                # When windowed and inside the window, OR when --specificity_replace_main is
                # set and specificity fired (2026-07-26, user request: spread the "replace,
                # don't add" behavior across the whole run via --specificity_batch_ratio
                # instead of confining it to one late window), specificity's backward() above
                # is the ONLY backward this step -- the main forward_curriculum/loss block
                # below is skipped entirely (one forward this step, not two).
                _spec_replace_main = getattr(args, "specificity_replace_main", False)
                _skip_main_forward = (_spec_windowed or _spec_replace_main) and _spec_fire

                # ====================================================
                # DELTA-RESPONSE LOSS (backward separately, same DDP reason)
                # ====================================================
                # [F(x,c)-F(x,0)] must match [y_c^GT - x] — stops the prompt from acting as a
                # global "volume knob" that improves any input regardless of relevance.
                _lambda_delta = getattr(args, "lambda_delta_response", 0.0)
                if (_lambda_delta > 0
                        and getattr(args, "use_degradation", True)
                        and random.random() < getattr(args, "delta_response_ratio", 0.5)):
                    _um_delta = accelerator.unwrap_model(net_pisasr)
                    loss_delta = _um_delta.compute_delta_response_loss(x_tgt, args)
                    accelerator.backward(loss_delta * _lambda_delta)
                    _delta_loss_val = loss_delta.item()
                else:
                    _delta_loss_val = None

                # ====================================================
                # DOSE-RESPONSE LOSS (backward separately, same DDP reason)
                # ====================================================
                # Output difference between two requested-removal fractions u_a<u_b must
                # track the GT target difference — makes low-strength requests (e.g. u=0.25)
                # produce a response clearly above the noise floor.
                _lambda_dose = getattr(args, "lambda_dose_response", 0.0)
                if (_lambda_dose > 0
                        and getattr(args, "use_degradation", True)
                        and random.random() < getattr(args, "dose_response_ratio", 0.5)):
                    _um_dose = accelerator.unwrap_model(net_pisasr)
                    loss_dose = _um_dose.compute_dose_response_loss(x_tgt, args)
                    accelerator.backward(loss_dose * _lambda_dose)
                    _dose_loss_val = loss_dose.item()
                else:
                    _dose_loss_val = None

                # ====================================================
                # FORWARD VÀ TÍNH LOSS
                # ====================================================
                # --use_mixed_curriculum (2026-07-16 v2-Mix-from-scratch recipe): replace the
                # real-paired-data forward (x_src = real LR, unknown/unlabeled degradation
                # composition) with forward_curriculum (x_tgt only, LR SYNTHESIZED with a
                # known multi-degradation label every step) — see PiSASR.forward_curriculum's
                # docstring for why (freezing LoRA and only retuning the degradation-prompt
                # tokens on this same curriculum, in an earlier "Stage 1" experiment, failed
                # to fix the mixed-degradation interaction problem the Phase 0 audit found;
                # the correction-performing LoRA weights themselves need this signal, so it
                # now drives the MAIN loss instead of a side branch).
                _loss_router_dose_val = None
                if _skip_main_forward:
                    # Specificity's own backward() already ran above -- do NOT call
                    # accelerator.backward() again this step. total_loss/intensities_pred
                    # below are placeholders for the logging code further down only.
                    total_loss = loss_specificity.detach() * _lambda_spec
                    intensities_pred = torch.zeros(
                        x_tgt.shape[0], accelerator.unwrap_model(net_pisasr).num_deg,
                        device=x_tgt.device)
                else:
                    _curriculum_dose_random = getattr(args, "curriculum_dose_random", False)
                    if _curriculum_dose_random:
                        # 2026-07-26 (user request): merged curriculum+specificity forward --
                        # target_tgt is per-sample (full/partial/mismatch), NOT always x_tgt;
                        # is_full_dose gates CSD (see forward_curriculum_dosed's docstring for
                        # why CSD only fires on full-dose samples).
                        x_tgt_pred, target_tgt, latents_pred, prompt_embeds, neg_prompt_embeds, \
                            intensities_pred, is_full_dose = \
                            accelerator.unwrap_model(net_pisasr).forward_curriculum_dosed(x_tgt, args)
                    elif getattr(args, "use_mixed_curriculum", False):
                        x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, intensities_pred = \
                            accelerator.unwrap_model(net_pisasr).forward_curriculum(x_tgt, args)
                        target_tgt = x_tgt
                        is_full_dose = None
                    else:
                        x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, intensities_pred = \
                            net_pisasr(x_src, x_tgt, batch=batch, args=args)
                        target_tgt = x_tgt
                        is_full_dose = None

                    # 1. L2 Loss
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), target_tgt.float(), reduction="mean") * lambda_l2

                    _csd_fire = args.no_csd is False and (is_full_dose is None or bool(is_full_dose.all()))
                    loss_csd = (net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args) * lambda_csd
                               if _csd_fire else torch.tensor(0.0, device=x_src.device))
                    total_loss = (loss_l2 + loss_csd
                                  + net_lpips(x_tgt_pred.float(), target_tgt.float()).mean() * lambda_lpips)

                    # Explicit router dose loss (2026-07-16 v2-Mix recipe, --lambda_router_dose_online):
                    # free (no extra UNet forward) — reuses forward_curriculum's own router_pred/
                    # labels_syn, already in the SAME graph as x_tgt_pred, so it's just added into
                    # total_loss before the one backward() call below, not backward'd separately.
                    _lambda_router_dose = getattr(args, "lambda_router_dose_online", 0.0)
                    if _lambda_router_dose > 0:
                        _um_dose = accelerator.unwrap_model(net_pisasr)
                        _router_pred = getattr(_um_dose, "last_router_pred", None)
                        _labels_syn = getattr(_um_dose, "last_labels_syn", None)
                        if _router_pred is not None and _labels_syn is not None:
                            loss_router_dose = F.smooth_l1_loss(_router_pred, _labels_syn) * _lambda_router_dose
                            total_loss = total_loss + loss_router_dose
                            _loss_router_dose_val = loss_router_dose.item()

                    accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    has_grad = any(p.grad is not None for p in lora_params)
                    if has_grad:
                        accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                        opt_unet.step()
                        lr_scheduler_unet.step()
                        opt_unet.zero_grad(set_to_none=True)

                    # Only populated once router_finetune_active unfreezes router_params
                    # above; a no-op (has_grad False) for the rest of training.
                    has_grad_router = any(p.grad is not None for p in router_params)
                    if has_grad_router:
                        accelerator.clip_grad_norm_(router_params, args.max_grad_norm)
                        opt_router.step()
                        lr_scheduler_router.step()
                        opt_router.zero_grad(set_to_none=True)

                    # Only populated when --domain_prompt is set (0-length content otherwise
                    # -> no forward contribution -> grad stays None); no-op if unset.
                    has_grad_domain = any(p.grad is not None for p in domain_prompt_params)
                    if has_grad_domain:
                        accelerator.clip_grad_norm_(domain_prompt_params, args.max_grad_norm)
                        opt_domain_prompt.step()
                        lr_scheduler_domain_prompt.step()
                        opt_domain_prompt.zero_grad(set_to_none=True)

                    # Only populated when --degrad_prompt_learnable is set (else requires_grad
                    # was left False above -> grad stays None); no-op otherwise. text_prototypes
                    # never gets grad regardless (dead parameter after __init__ — see comment
                    # above degrad_prompt_params) but is harmless to include here.
                    has_grad_degrad = any(p.grad is not None for p in degrad_prompt_params)
                    if has_grad_degrad:
                        accelerator.clip_grad_norm_(degrad_prompt_params, args.max_grad_norm)
                        opt_degrad_prompt.step()
                        lr_scheduler_degrad_prompt.step()
                        opt_degrad_prompt.zero_grad(set_to_none=True)

                # --- Logging ---
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    dataset_train.current_step = global_step  # drives domain_rand_anneal_frac schedule

                    logs = {"loss": total_loss.detach().item()}
                    if _synth_sr_loss_val is not None:
                        logs["synth_sr/loss"] = _synth_sr_loss_val
                    if _spec_loss_val is not None:
                        logs["specificity/loss"] = _spec_loss_val
                    if _delta_loss_val is not None:
                        logs["delta_response/loss"] = _delta_loss_val
                    if _dose_loss_val is not None:
                        logs["dose_response/loss"] = _dose_loss_val
                    if _loss_router_dose_val is not None:
                        logs["router_dose_online/loss"] = _loss_router_dose_val
                    if getattr(args, "teacher_forcing_anneal_frac", 0.0) > 0:
                        logs["teacher_forcing/alpha"] = accelerator.unwrap_model(net_pisasr)._teacher_forcing_alpha(args)
                    _um = accelerator.unwrap_model(net_pisasr)
                    for _i, _dn in enumerate(_um.deg_names):
                        logs[f"inten/{_dn}"] = intensities_pred[:, _i].mean().detach().item()
                    # LGDP-specific logs
                    if hasattr(_um, "last_prompt_weights"):
                        _deg_names = _um.deg_names
                        for _i, _dn in enumerate(_deg_names):
                            logs[f"lgdp_weight/{_dn}"] = _um.last_prompt_weights[_i].item()
                        _rts = _um.restoration_token_scale.detach().cpu()
                        for _i, _dn in enumerate(_deg_names):
                            logs[f"restoration_token_scale/{_dn}"] = _rts[_i].item()
                        # Print top-2 prompt weights for qualitative monitoring
                        if accelerator.is_main_process and global_step % 100 == 0:
                            _pw = _um.last_prompt_weights.tolist()
                            _sorted = sorted(zip(_deg_names, _pw), key=lambda x: -x[1])
                            accelerator.print("top prompts: " + ", ".join(
                                f"{n}={v:.3f}" for n, v in _sorted[:2]))
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == args.checkpointing_steps - 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step+1}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(outf)

                    if global_step % args.eval_freq == args.eval_freq - 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(net_pisasr)
                        unwrapped_model.eval()
                        for batch_val in dl_val:
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            assert x_src.shape[0] == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred, _, _, _, _ = accelerator.unwrap_model(net_pisasr)(x_src, x_tgt, batch=batch_val, args=args)
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                output_pil.save(os.path.join(args.output_dir, "eval", f"fid_{global_step}", x_basename))
                        gc.collect()
                        torch.cuda.empty_cache()
                        unwrapped_model.train()
                        # .train() recurses into all submodules — re-freeze the router's eval
                        # mode (dropout) since it's normally fixed after Phase 1 and must stay
                        # deterministic. Skip this once router_finetune_active: it's genuinely
                        # training again during the fine-tune window and should keep dropout on.
                        if not router_finetune_active:
                            unwrapped_model.set_router_eval()

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                print("Training completed.")
                outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                accelerator.unwrap_model(net_pisasr).save_model(outf)
                return

if __name__ == "__main__":
    args = parse_args()
    main(args)
