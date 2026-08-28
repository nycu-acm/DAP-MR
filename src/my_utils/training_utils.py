import os
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--is_module", default=False)
    parser.add_argument("--tracker_project_name", type=str, default="pisasr2025")

    # args for the loss function
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_csd", default=1.0, type=float)

    # args for the csd training
    parser.add_argument("--pretrained_model_path_csd", default='preset/models/stable-diffusion-2-1-base', type=str)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--neg_prompt_csd", default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--pos_prompt_csd", default="", type=str)
    parser.add_argument("--cfg_csd", default=1.0, type=float)

    # args for the `t` test
    parser.add_argument("--timesteps1", default=1, type=float)
    # details about the model architecture
    parser.add_argument("--pretrained_model_path", default='preset/models/stable-diffusion-2-1-base')
    # # unet lora setting
    parser.add_argument("--lora_rank_unet_pix", default=4, type=int)
    parser.add_argument("--lora_rank_unet_sem", default=4, type=int)

    # dataset options
    parser.add_argument("--dataset_txt_paths", default='/gt_path.txt', type=str)
    parser.add_argument("--highquality_dataset_txt_paths", default=None, type=str)
    parser.add_argument("--dataset_test_folder",
                        default="/testfolder")
    parser.add_argument("--null_text_ratio", default=0.1, type=float)
    parser.add_argument("--prob", default=0.5, type=float)
    parser.add_argument("--resolution_ori", type=int, default=512,)
    parser.add_argument("--resolution_tgt", type=int, default=512,)

    # resume
    parser.add_argument("--resume_ckpt", default=None, type=str)

    # LGDP prompt bank and routing args
    parser.add_argument("--prompt_bank_mode", type=str, default="target_state",
                        choices=["target_state", "instruction", "degradation_noun", "short_noun",
                                 "correct", "pos", "neg", "random", "nonsense"],
                        help="Which text prompt bank to use for LGDP prototype initialization.")
    parser.add_argument("--lgdp_injection_mode", type=str, default="direct_feed",
                        choices=["direct_feed", "text_ordinal", "text_numeric", "ram_tag"],
                        help="direct_feed (default) = _compose_lgdp_prompt's learned-vector "
                             "blend (BOS + 6 restoration tokens weighted by router intensity), "
                             "unchanged original scheme. text_ordinal/text_numeric = build a "
                             "literal sentence from the router's [B,6] intensities each forward "
                             "pass (e.g. \"restore strong speckle noise\" / \"speckle noise "
                             "level 0.25\") and run it through the frozen CLIP text encoder as "
                             "the entire UNet prompt -- see _build_lgdp_text_prompt/"
                             "_compose_lgdp_prompt_text in pisasr.py. Non-differentiable "
                             "intensity->text step, so this cuts the SR loss's gradient to the "
                             "router (router still trains via Phase 1 pretrain/CF/dose-loss, "
                             "none of which call this path). ram_tag = run the frozen RAM "
                             "(Recognize Anything Model) tagger on the LR image itself each "
                             "forward pass, and run the resulting tag string through the frozen "
                             "CLIP text encoder as the entire UNet prompt -- see "
                             "load_ram_tagger/build_ram_tag_prompt/_compose_lgdp_prompt_ram in "
                             "pisasr.py. Real per-image content replaces the router-driven "
                             "prompt entirely; the router is not needed in this mode -- pass "
                             "--router_pretrain_steps 0 --lambda_router_dose_online 0.0 "
                             "--teacher_forcing_anneal_frac 0.0 so it is never called.")
    parser.add_argument("--ram_pretrained_path", type=str,
                        default="/mnt/HDD1/tuong/SR/PiSA-SR_med/src/ram_pretrain_model/ram_swin_large_14m.pth",
                        help="Checkpoint for the RAM tagger, only loaded when "
                             "--lgdp_injection_mode ram_tag.")
    parser.add_argument("--lgdp_text_phrase_style", type=str, default="degradation_noun",
                        choices=["degradation_noun", "restoration_verb"],
                        help="Only used by --lgdp_injection_mode text_ordinal/text_numeric. "
                             "degradation_noun (default) names each slot by the problem "
                             "(NOUN_PHRASES_FULL6/MERGED3, e.g. 'depth blur'). restoration_verb "
                             "names it by the fix action instead (RESTORATION_VERB_PHRASES_"
                             "FULL6/MERGED3, e.g. 'deblur') -- see pisasr.py.")
    parser.add_argument("--lgdp_token_scheme", type=str, default="full6",
                        choices=["full6", "merged3"],
                        help="full6 (default) = one restoration token/router-output per "
                             "physical degradation type (psf_blur, haze, attenuation, "
                             "speckle, additive_noise, compression), 6 total -- unchanged "
                             "original LGDP scheme. merged3 = groups those same 6 physical "
                             "degradation generators into 3 restoration tokens by shared "
                             "frequency content: blur_detail (psf_blur+compression), "
                             "photometric (haze+attenuation), texture_noise "
                             "(speckle+additive_noise) -- see DEG_TYPE_TO_GROUP_MERGED3 in "
                             "pisasr.py. The 6 physics degradation functions themselves are "
                             "unchanged; only the per-type severity label is max-pooled into "
                             "its group's slot before it reaches the router loss/prompt "
                             "injection. Ablation to test whether 3 coarser tokens generalize "
                             "better/worse than 6 fine-grained ones. Must match between "
                             "training and --resume_ckpt/eval (asserted).")
    parser.add_argument("--restoration_token_init_scale", type=float, default=1.0,
                        help="Initial value for restoration_token_scale parameters.")
    parser.add_argument("--lgdp_prompt_strength", type=float, default=1.0,
                        help="Global scale applied to the 6 weighted restoration tokens that, "
                             "together with BOS, form the entire UNet prompt (0=degenerate to BOS-only).")
    parser.add_argument("--lgdp_prompt_length", type=int, default=7, choices=[7, 77],
                        help="Sequence length of the UNet's text condition. 7 (default) = "
                             "direct-feed [BOS + 6 restoration tokens] only. 77 = pad out to "
                             "CLIP's native context length using the null (\"\") prompt's own "
                             "padding/EOS tail, so cross-attention runs over the sequence "
                             "length its positional embeddings/attention statistics were "
                             "actually calibrated on during SD pretraining, instead of an "
                             "out-of-distribution length-7 sequence.")
    parser.add_argument("--lgdp_variable_tokens", type=lambda x: x.lower() != "false",
                        default=False,
                        help="If true, each degradation's prompt text is kept as its FULL "
                             "content-token block (variable length, longer description = "
                             "more tokens) instead of EOS-pooled to exactly 1 token, and the "
                             "6 blocks are concatenated: [BOS] + [deg1 tokens] + ... + "
                             "[deg6 tokens]. Each degradation's router intensity/scale is "
                             "broadcast across its own block. Mutually exclusive in practice "
                             "with --domain_prompt (target_state bank alone already uses ~75 "
                             "of the 76 available content-token slots before CLIP's native "
                             "77-length context, safety-truncated if a bank's total exceeds "
                             "that). Default false = original 1-token-per-degradation scheme, "
                             "zero effect on existing runs/checkpoints.")
    parser.add_argument("--domain_prompt", type=str, default="",
                        help="Optional static domain-description text (e.g. \"ultrasound "
                             "image\"), spliced into the LGDP sequence as "
                             "[BOS] + [6 degrad tokens] + [domain prompt content] + [EOS]. "
                             "Default \"\" disables it entirely (zero effect on existing "
                             "runs/checkpoints). LEARNABLE: initialized from this text via "
                             "the frozen text encoder, then trained jointly with the LoRA "
                             "weights (its own optimizer group, always-on through Phase 2) "
                             "and saved into the checkpoint — test_pisasr.py does not need "
                             "to repeat this exact string, it loads the trained value.")
    parser.add_argument("--degrad_prompt_learnable", type=lambda x: x.lower() != "false",
                        default=False,
                        help="If true, the degradation prompt (text_prototypes/"
                             "restoration_tokens/restoration_token_scale) is trained "
                             "jointly with the LoRA weights throughout Phase 2 (own "
                             "always-on optimizer group, same treatment as "
                             "--domain_prompt), instead of the default: fixed at its "
                             "text-encoder-derived initialization for the whole run "
                             "(text_prototypes itself never receives gradient either way — "
                             "it's a dead parameter after __init__, only used to seed "
                             "restoration_tokens' initial value once).")
    parser.add_argument("--router_arch", choices=["base", "conv"], default="base",
                        help="base (default) = original vis_projector/vis_norm/spatial_pool/"
                             "DegradationRouter (patch-unfold -> 6-global-statistic pool -> MLP). "
                             "conv = ConvRouterHead: small CNN run directly over the VAE latent, "
                             "pooling (mean+std) only at the very end, so local spatial "
                             "degradation evidence (speckle/psf texture) survives much longer "
                             "than the 6-global-statistic pool allows. Standalone diagnostic "
                             "(router_indomain_outdomain.py) found conv beats base on BOTH "
                             "in-domain accuracy and out-of-domain generalization to unseen "
                             "anatomies, trained on Cardiac only in both cases, with ~1/3 the "
                             "params — see memory pisasr-degradation-router.")
    parser.add_argument("--force_router_intensity", type=float, default=None,
                        help="If set, bypass the router ENTIRELY (vis_projector/vis_norm/"
                             "spatial_pool/router never called in forward()) and use this "
                             "constant value for every degradation intensity on every "
                             "image. Also skips Phase 1 router pretraining entirely (there "
                             "is nothing to pretrain — router_params stay frozen and unused "
                             "for the whole run). Default None = router behaves normally.")
    # Synthetic degradation-magnitude-supervised SR loss: teaches the LoRA to respond to the
    # degradation prompt's magnitude using a KNOWN-correct label (not the router's own guess).
    parser.add_argument("--lambda_synth_sr", type=float, default=0.0,
                        help="Weight for the synthetic degradation-magnitude-supervised SR loss "
                             "(0=disabled, default off — zero effect on existing runs).")
    parser.add_argument("--synth_sr_ratio", type=float, default=0.5,
                        help="Probability per step of running the synthetic branch when "
                             "lambda_synth_sr > 0 (controls extra compute / mix with real data).")
    parser.add_argument("--single_degradation_prob", type=float, default=0.5,
                        help="Probability that a synthesized sample (router loss AND synthetic "
                             "SR loss) uses exactly ONE isolated degradation type instead of the "
                             "usual random combination + forced compression. Lets each "
                             "restoration token get an individually-calibrated magnitude signal.")
    parser.add_argument("--controllable_max_degradations", type=int, default=4,
                        help="Max number of degradations composited into a controllable "
                             "training input (used by the delta/dose-response losses' "
                             "multi-degradation triplet builder — see "
                             "PiSASR._synthesize_controllable_triplet).")
    parser.add_argument("--controllable_min_severity", type=float, default=0.2,
                        help="Lower bound for a degradation's full-strength severity "
                             "(sampled U(this, 1.0)) wherever a synthetic training sample "
                             "needs one: the specificity batch loss below and the "
                             "delta/dose-response losses. Kept above 0 so there is always "
                             "something meaningful to remove.")
    # Cheap single-forward matched/mismatched specificity + continuous partial-restoration
    # loss (spec C.8/C.10). Supersedes an earlier multi-forward design (separate
    # controllable-SR + match/mismatch branches, up to 4 UNet forward/backward passes per
    # step) with one that costs exactly ONE forward per step: each sample in the batch is
    # independently matched (correct slot requested, target = partial restoration) or
    # mismatched (wrong slot requested, target = the degraded input unchanged) — see
    # PiSASR.compute_specificity_batch_loss / _synthesize_specificity_batch. Directly
    # targets the Phase 0 audit's root-cause finding (memory pisasr-dapmr-phase0-audit):
    # one restoration token (attenuation) acts as a near-constant, always-on correction
    # regardless of which degradation is actually present, drowning out other slots.
    parser.add_argument("--lambda_specificity", type=float, default=0.0,
                        help="Weight for the specificity batch loss (0=disabled, default "
                             "off — zero effect on existing runs).")
    parser.add_argument("--specificity_batch_ratio", type=float, default=1.0,
                        help="Probability per step of running the specificity batch branch "
                             "when lambda_specificity > 0. 1.0=every step (it costs only ONE "
                             "extra forward pass, unlike the old multi-branch design).")
    parser.add_argument("--specificity_matched_prob", type=float, default=0.7,
                        help="Per-SAMPLE probability (not per-step) of the matched case "
                             "(correct slot requested) vs mismatched (wrong slot requested) "
                             "within the specificity batch.")
    parser.add_argument("--lambda_lpips_specificity", type=float, default=0.0,
                        help="Weight of the LPIPS term within the specificity batch loss "
                             "(L = L1 + lambda_lpips_specificity * LPIPS). 0=L1-only "
                             "(default).")
    parser.add_argument("--specificity_window_start", type=int, default=-1,
                        help="2026-07-25 efficiency restructuring (user request): if >=0 "
                             "(together with --specificity_window_end), specificity loss "
                             "ONLY fires for global_step in [start, end) and REPLACES the "
                             "main forward_curriculum/forward step entirely that iteration "
                             "(exactly one UNet forward that step, not two). Outside the "
                             "window, specificity is off. Rationale: forward_curriculum "
                             "already trains matched/full-severity behavior every step via "
                             "its own ground-truth label (beta=1 until the teacher-forcing "
                             "anneal near the end), so running specificity's matched/"
                             "mismatch probe on every step for the whole run duplicates "
                             "that signal at the cost of a full extra forward/backward each "
                             "time. Default -1 disables windowing (falls back to the legacy "
                             "always-on-every-step behavior gated only by "
                             "--specificity_batch_ratio, running ALONGSIDE the main loss as "
                             "an extra forward) — zero effect on existing runs/checkpoints.")
    parser.add_argument("--specificity_window_end", type=int, default=-1,
                        help="See --specificity_window_start.")
    parser.add_argument("--specificity_replace_main", type=lambda x: x.lower() != "false",
                        default=False,
                        help="2026-07-26 (user request): extends the windowed 'replace, "
                             "don't add' behavior to the NON-windowed, "
                             "--specificity_batch_ratio-driven firing case. When true, "
                             "any step where specificity fires (per --specificity_batch_ratio, "
                             "e.g. 0.2) REPLACES the main forward_curriculum step entirely "
                             "(one UNet forward that step, not two) exactly like the windowed "
                             "case, but spread probabilistically across the WHOLE run instead "
                             "of confined to one late block -- total compute stays flat versus "
                             "the plain no-specificity baseline (always exactly one forward "
                             "per step), while specificity exposure is continuous throughout "
                             "training instead of a single late burst. No-op when "
                             "--specificity_window_start/_end are set (windowing already wins "
                             "there). Default false preserves exact legacy behavior "
                             "(additive every firing step) for existing runs/checkpoints.")
    parser.add_argument("--curriculum_dose_random", type=lambda x: x.lower() != "false",
                        default=False,
                        help="2026-07-26 (user request): merges specificity-style partial-"
                             "dose/mismatch supervision DIRECTLY into forward_curriculum's "
                             "own esrgan_physic synthesis (via generate_labeled_lr_dosed), "
                             "instead of a separate compute_specificity_batch_loss forward "
                             "pass. When true, forward_curriculum_dosed replaces "
                             "forward_curriculum: each sample is randomly full-dose "
                             "(unchanged forward_curriculum behavior), partial-dose (request "
                             "a random fraction of the true severity, target = the SAME "
                             "esrgan_physic realization re-rendered at the remaining "
                             "severity), or mismatch (request removal of an absent "
                             "degradation, target = input unchanged) -- see "
                             "--curriculum_dose_partial_prob/_mismatch_prob/_min_u/_max_u. "
                             "ONE UNet forward trains both objectives every step at the SAME "
                             "cost as plain forward_curriculum -- leave --lambda_specificity "
                             "at its default 0 when this is set (the two mechanisms overlap). "
                             "Requires --online_degrade_source esrgan_physic. Default false "
                             "preserves forward_curriculum's exact existing behavior — zero "
                             "effect on existing runs/checkpoints.")
    parser.add_argument("--curriculum_dose_partial_prob", type=float, default=0.3,
                        help="Only used when --curriculum_dose_random. Per-sample "
                             "probability of the partial-dose case (of the samples that "
                             "aren't --online_clean_prob-bypassed).")
    parser.add_argument("--curriculum_dose_mismatch_prob", type=float, default=0.1,
                        help="Only used when --curriculum_dose_random. Per-sample "
                             "probability of the mismatch case, checked BEFORE "
                             "--curriculum_dose_partial_prob (so the two are mutually "
                             "exclusive; remaining probability mass is the full-dose case).")
    parser.add_argument("--curriculum_dose_isolate_prob", type=float, default=0.0,
                        help="2026-08-12 (user request), only used when --curriculum_dose_random. "
                             "Per-sample probability of the ISOLATE case, checked BEFORE mismatch/"
                             "partial/full: renders the input with only ONE randomly-chosen "
                             "physical degradation active (dose=onehot(k)), unlike every other "
                             "case here which always renders at dose=[1]*6 (all 6 simultaneously "
                             "present). Addresses a train/eval distribution gap found by the "
                             "direct-control audit (figure_direct_eval_*.py): genuinely isolated "
                             "single-degradation inputs, which that eval probes directly, were "
                             "never seen in training at any --curriculum_dose_random probability "
                             "mix before this flag existed. u sampled the same way as the partial "
                             "case (_sample_user_multiplier). Default 0.0 preserves exact existing "
                             "behavior for runs that don't set it.")
    parser.add_argument("--curriculum_dose_min_u", type=float, default=0.1,
                        help="Only used when --curriculum_dose_random. Lower bound of the "
                             "per-sample uniform range for the partial-dose removal "
                             "fraction u (control = u * true_severity, target rendered at "
                             "remaining severity (1-u)).")
    parser.add_argument("--curriculum_dose_max_u", type=float, default=0.9,
                        help="Only used when --curriculum_dose_random. Upper bound of the "
                             "per-sample uniform range for u -- kept below 1.0 so the "
                             "partial-dose case stays distinguishable from the full-dose "
                             "case (which already covers u=1 with target=clean c_tgt).")
    # Delta-response loss (spec C.11): the CHANGE in output from a control vector must match
    # the actual degradation removed, not act as a global "volume knob".
    parser.add_argument("--lambda_delta_response", type=float, default=0.0,
                        help="Weight for the delta-response loss (0=disabled, default off).")
    parser.add_argument("--delta_response_ratio", type=float, default=0.5,
                        help="Probability per step of running the delta-response branch when "
                             "lambda_delta_response > 0.")
    # Dose-response loss (spec C.12): output difference between two requested-removal
    # fractions u_a < u_b on the same input must track the ground-truth target difference.
    parser.add_argument("--lambda_dose_response", type=float, default=0.0,
                        help="Weight for the dose-response loss (0=disabled, default off).")
    parser.add_argument("--dose_response_ratio", type=float, default=0.5,
                        help="Probability per step of running the dose-response branch when "
                             "lambda_dose_response > 0.")
    # Teacher forcing before router (spec C.9): early in training, the synthetic SR loss's
    # prompt uses the ground-truth degradation label exclusively; annealed toward the
    # router's OWN (detached) prediction over the last teacher_forcing_anneal_frac of
    # training, so prompt-specificity error and router-prediction error aren't confounded
    # from the start. See PiSASR._teacher_forcing_alpha.
    parser.add_argument("--teacher_forcing_anneal_frac", type=float, default=0.0,
                        help="Fraction of TOTAL training steps (from the END) over which the "
                             "synthetic SR loss's prompt is linearly annealed from the "
                             "ground-truth label (alpha=1) to the router's own prediction "
                             "(alpha=0). 0.0=disabled (default, alpha stays 1.0 for the whole "
                             "run — zero effect on existing runs). Also drives Stage 2 "
                             "(--finetune_stage router)'s beta anneal.")
    parser.add_argument("--teacher_forcing_alpha_override", type=float, default=None,
                        help="2026-07-27 (user request, label-only vs router-only ablation): "
                             "if set, PiSASR._teacher_forcing_alpha returns this FIXED value "
                             "every step instead of computing the --teacher_forcing_anneal_frac "
                             "schedule -- e.g. 0.0 makes the router's own prediction condition "
                             "the prompt from step 0 of Phase 2 (no anneal window), while "
                             "--router_finetune_steps still independently controls when the "
                             "router itself unfreezes for joint gradient. Default None preserves "
                             "the normal anneal-frac-driven schedule — zero effect on existing "
                             "runs/checkpoints.")

    # v2-Mix fine-tune curriculum (2026-07-16 follow-up to the mixed-degradation + oracle-
    # router audit): --finetune_stage bypasses the normal from-scratch Phase 1 (router
    # pretrain) / Phase 2 (pix-then-sem) training loop entirely and runs a dedicated
    # 2-stage post-hoc fine-tune, resumed from an already-trained checkpoint via
    # --resume_ckpt (required when this is set). See PiSASR.compute_mixed_curriculum_sr_loss
    # / run_finetune_stage in train_pisasr.py.
    parser.add_argument("--finetune_stage", type=str, default="none",
                        choices=["none", "controller", "router"],
                        help="none (default) = normal training, zero effect on existing "
                             "runs. 'controller' (Stage 1) = freeze router+LoRA, train ONLY "
                             "the degradation-prompt directions/scale on mixed-degradation "
                             "reconstruction with the prompt fed directly from ground truth "
                             "(fixes token/controller interaction). 'router' (Stage 2) = "
                             "freeze LoRA+degradation-prompt, train ONLY the router via a "
                             "dose+ranking loss plus a teacher-forcing-annealed "
                             "reconstruction loss (fixes router calibration on mixed inputs).")
    parser.add_argument("--online_degrade_source", type=str, default="curriculum",
                        choices=["curriculum", "esrgan_physic"],
                        help="Sampler forward_curriculum uses. 'curriculum' (default) = "
                             "_synthesize_mixed_curriculum_batch, the independently "
                             "re-implemented port. 'esrgan_physic' = "
                             "_synthesize_online_esrgan_batch, an exact online port of "
                             "generate_lr_esrgan_physic.py (the ACTUAL generator that "
                             "produced the static training/eval LR files) — matches the "
                             "eval-time degradation distribution exactly, unlike "
                             "'curriculum' (which caused a real Cardiac/BUSI regression "
                             "in a from-scratch run, likely from a train/eval degradation-"
                             "distribution mismatch). See "
                             "src/datasets/esrgan_physic_online.py.")
    parser.add_argument("--lambda_router_dose_online", type=float, default=0.0,
                        help="Weight for an EXPLICIT router dose-regression loss "
                             "(SmoothL1(router_pred, ground_truth_label)) computed for "
                             "free from forward_curriculum's own forward pass (no extra "
                             "UNet call) — self.last_router_pred / self.last_labels_syn. "
                             "0=disabled (default). Intended for the --router_finetune_steps "
                             "window: forces the router's own prediction to explicitly "
                             "match the online-generated label, on top of the indirect "
                             "gradient it already gets via the teacher-forcing-blended "
                             "reconstruction loss.")
    parser.add_argument("--use_mixed_curriculum", type=lambda x: x.lower() != "false",
                        default=False,
                        help="v2-Mix-from-scratch recipe (2026-07-16): replace the main "
                             "Phase 2 training forward entirely — instead of a real paired "
                             "LR/HR image (unknown/unlabeled degradation composition), "
                             "SYNTHESIZE the LR from the HR via the mixed-degradation "
                             "curriculum (PiSASR.forward_curriculum), so every step has an "
                             "exact known multi-degradation label. Prompt conditioning is "
                             "teacher-forcing-annealed (--teacher_forcing_anneal_frac) from "
                             "the ground-truth label (beta=1, early) to the router's own "
                             "prediction (beta=0, late). Also swaps Phase 1's router-"
                             "pretrain sampler to the same curriculum. Default False = "
                             "zero effect on existing runs (real paired data, as before).")
    parser.add_argument("--online_clean_prob", type=float, default=0.1,
                        help="Probability per sample (in _synthesize_online_esrgan_batch, "
                             "--online_degrade_source esrgan_physic only) of bypassing the "
                             "degrade generator entirely and using the clean HR as-is with "
                             "label=[0]*6 — lets the router calibrate what genuinely low "
                             "severity looks like (the real generator never produces a "
                             "near-clean sample on its own). 0.0=disabled (default behavior "
                             "before this flag existed).")
    parser.add_argument("--random_degrad_order", type=lambda x: x.lower() != "false",
                        default=False,
                        help="2026-08-09 (user request): only used when --online_degrade_source "
                             "esrgan_physic. When true, _synthesize_online_esrgan_batch calls "
                             "generate_labeled_lr_random_order instead of generate_labeled_lr — "
                             "each sample independently randomizes HOW MANY (1-4) of the 4 "
                             "US-physics effects (haze, attenuation, speckle, additive_noise) "
                             "fire and in WHAT ORDER they're composed, on top of the severity "
                             "randomization both generators already do. Applies identically to "
                             "Phase 1 (router pretrain, via compute_router_loss) and Phase 2 "
                             "(forward_curriculum) since both dispatch through the same "
                             "_synthesize_online_esrgan_batch. The ESRGAN-backbone stage "
                             "(psf_blur+compression) keeps its fixed internal step order — see "
                             "generate_labeled_lr_random_order's docstring for why. Default "
                             "false preserves generate_labeled_lr's fixed atten->speckle->"
                             "noise->haze order — zero effect on existing runs/checkpoints.")
    parser.add_argument("--speckle_oversample_prob", type=float, default=0.45,
                        help="Within the mixed-degradation curriculum's two/three-degradation "
                             "samples, probability of drawing one of the speckle-containing "
                             "combos instead of a uniform random combo — speckle was the "
                             "single most consistent failure across every audit so far "
                             "(matched-slot rank 5-6/6, worst interaction of any slot).")
    parser.add_argument("--lambda_dose", type=float, default=1.0,
                        help="Weight for Stage 2's dose-regression term (SmoothL1(router_"
                             "pred, ground_truth_severity)).")
    parser.add_argument("--lambda_rank", type=float, default=0.0,
                        help="Weight for Stage 2's active/inactive ranking loss (0=disabled, "
                             "default off — zero effect elsewhere). See "
                             "active_inactive_ranking_loss in pisasr.py.")
    parser.add_argument("--rank_margin", type=float, default=0.1,
                        help="Margin for the active/inactive ranking loss.")
    # PiSA pixel/semantic branch scales.
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="Base pixel-branch scale for the LoRA gate.")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="Base semantic-branch scale for the LoRA gate.")
    parser.add_argument("--debug_router", action="store_true",
                        help="Print router intensities per image.")
    # Domain randomization on the MAIN LR input (non-label appearance aug for cross-domain).
    parser.add_argument("--domain_randomization", type=lambda x: x.lower() != "false",
                        default=False,
                        help="Apply photometric domain randomization (gain/contrast/gamma/brightness) "
                             "to the LR-HR training pair to widen appearance distribution and improve "
                             "zero-shot transfer to unseen domains (Cardiac/HC18). Not a degradation label.")
    parser.add_argument("--domain_rand_strength", type=float, default=0.2,
                        help="Strength of domain randomization (0=off). Gain/contrast ~U(1-s,1+s), "
                             "gamma ~logU(-s,s), brightness shift ~U(-s/2,s/2).")
    parser.add_argument("--domain_rand_prob", type=float, default=1.0,
                        help="Probability of applying domain randomization per sample (RandAugment-style "
                             "gate). 1.0=every sample (old behavior). Lower (e.g. 0.3-0.5) keeps most steps "
                             "on clean in-domain statistics while still exposing the model periodically "
                             "to randomized appearance for OOD robustness.")
    parser.add_argument("--domain_rand_anneal_frac", type=float, default=0.0,
                        help="Fraction of TOTAL training steps (from the END) over which domain_rand_strength "
                             "is linearly annealed to 0. 0.0=disabled (constant strength). E.g. 0.2 anneals "
                             "strength->0 over the last 20%% of steps, so training finishes fine-tuning on "
                             "clean in-domain appearance after learning robust features early on.")
    
    parser.add_argument("--no_csd", action="store_true", help="Nếu gọi cờ này, CSD Loss sẽ bị tắt.")
    parser.add_argument(
        "--router_pretrain_steps",
        type=int,
        default=1000,
        help="Số step Phase 1 pre-train router bằng synthetic pseudo-labels. Router bị đóng băng "
             "(fixed) sau Phase 1 cho tới router_finetune_steps cuối (nếu > 0)."
    )
    parser.add_argument(
        "--router_finetune_steps",
        type=int,
        default=0,
        help="Số step CUỐI của Phase 2 mở lại router (vis_projector/vis_norm/router) để train "
             "chung với LoRA qua chính real-data SR reconstruction loss — router chỉ pretrain "
             "trên synthetic degradation (Phase 1) nên có thể lệch phân bố so với real LR; "
             "cửa sổ fine-tune này cho nó thích nghi bằng tín hiệu downstream thật. 0=tắt "
             "(router giữ nguyên đông lạnh suốt Phase 2, hành vi mặc định cũ)."
    )
    parser.add_argument(
        "--lambda_cf_router", type=float, default=0.0,
        help="Weight for the counterfactual router loss (compute_counterfactual_router_loss) "
             "added into Phase 1 router pretraining: same HR content, two DIFFERENT physics "
             "degradations -> router must reproduce the label DIFFERENCE between the two "
             "views. Since content is identical, only degradation evidence can explain the "
             "difference, forcing the router toward degradation-evidence features instead of "
             "content-correlated shortcuts. 0=disabled (default, zero effect on existing runs)."
    )
    parser.add_argument(
        "--lambda_cf_contrast", type=float, default=0.5,
        help="Weight for the counterfactual loss's internal CONTRAST term (label-difference "
             "regression), relative to its absolute-label term. Only has effect when "
             "lambda_cf_router > 0."
    )

    # training details
    parser.add_argument("--output_dir", default='experiments/oup')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--pix_steps", type=int, default=10,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--eval_freq", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--deg_file_path", default="params.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
