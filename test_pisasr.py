import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob


from ram.models.ram_lora import ram
from ram import inference_ram as inference
# ram_transforms = transforms.Compose([
#             transforms.Resize((384, 384)),
#             transforms.ToTensor(), # Add this to convert PIL to Tensor
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
# RAM = ram(pretrained='/mnt/HDD1/tuong/SR/PiSA-SR_med/src/ram_pretrain_model/ram_swin_large_14m.pth',
#         pretrained_condition=None,
#         image_size=384,
#         vit='swin_l')
# RAM.eval()
# RAM.to("cuda", dtype=torch.float16)

def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    # model.apply_layerwise_wrappers()
    model.set_eval()
    gate_vec = getattr(args, "gate_router_intensity_vec", None)
    force_vec = getattr(args, "force_router_intensity_vec", None)
    force_val = getattr(args, "force_router_intensity", None)
    if gate_vec is not None:
        _gate = [float(v) for v in gate_vec.split(",")]
        assert len(_gate) == model.num_deg, (
            f"--gate_router_intensity_vec needs {model.num_deg} comma-separated values "
            f"(deg_names={model.deg_names}), got {len(_gate)}")
        _orig_router_forward = model.router.forward
        def _gated_router_forward(x, _g=_gate, _orig=_orig_router_forward):
            t = _orig(x)  # real per-image router prediction, magnitude untouched
            gate_t = torch.tensor(_g, device=t.device, dtype=t.dtype).unsqueeze(0)
            return t * gate_t  # zero out degradations not in the gate, keep real intensity for the rest
        model.router.forward = _gated_router_forward
    elif force_vec is not None:
        _vec = [float(v) for v in force_vec.split(",")]
        assert len(_vec) == model.num_deg, (
            f"--force_router_intensity_vec needs {model.num_deg} comma-separated values "
            f"(deg_names={model.deg_names}), got {len(_vec)}")
        def _fixed_router_forward(x, _v=_vec):
            b = x.shape[0]
            t = torch.tensor(_v, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1)
            return t
        model.router.forward = _fixed_router_forward
    elif force_val is not None:
        def _fixed_router_forward(x, _v=force_val, _n=model.num_deg):
            b = x.shape[0]
            return torch.full((b, _n), _v, device=x.device, dtype=x.dtype)
        model.router.forward = _fixed_router_forward
    # check_prompt_distance(model)
    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # x_src_ram = ram_transforms(input_image).unsqueeze(0).to("cuda", dtype=torch.float16)
        
        # Lấy kết quả từ RAM
        # caption = inference(x_src_ram, RAM)
        
        # # CHỈ LẤY CHUỖI TIẾNG ANH (phần tử số 0)
        # if isinstance(caption, tuple) or isinstance(caption, list):
        #     validation_prompt = caption[0]
        # else:
        #     validation_prompt = str(caption)
            
        # print(f"Generated Prompt từ RAM: {validation_prompt}")
        validation_prompt = "ultrasound image of fetal head"
        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")

        # ── Router / Gate debug ──────────────────────────────────────────────
        # gate_val = torch.sigmoid(model.gated_degrad_attn.gate).item()
        # deg_names = ["blur", "noise", "SR", "motionblur", "speckle", "contrast"]
        # inten = model.last_intensities.tolist()
        # print(f"  gate={gate_val:.4f} | " +
        #       " | ".join(f"{n}={v:.3f}" for n, v in zip(deg_names, inten)))
        # ────────────────────────────────────────────────────────────────────
        time_records.append(inference_time)

        if os.environ.get("DUMP_INTEN") and getattr(model, "last_intensities", None) is not None:
            with open(os.environ["DUMP_INTEN"], "a") as _f:
                _iv = model.last_intensities.tolist()
                _f.write(bname + "," + ",".join(f"{v:.4f}" for v in _iv) + "\n")

        if os.environ.get("DUMP_SPATIAL") and getattr(model, "last_spatial", None) is not None:
            import numpy as _np
            _sd = os.environ["DUMP_SPATIAL"]; os.makedirs(_sd, exist_ok=True)
            _np.save(os.path.join(_sd, os.path.splitext(bname)[0] + ".npy"), model.last_spatial.numpy())

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?")

    # LGDP prompt bank and routing args
    parser.add_argument("--prompt_bank_mode", type=str, default="target_state",
                        choices=["target_state", "instruction", "degradation_noun", "short_noun",
                                 "correct", "pos", "neg", "random", "nonsense"],
                        help="Which text prompt bank to use for LGDP prototype initialization.")
    parser.add_argument("--lgdp_injection_mode", type=str, default="direct_feed",
                        choices=["direct_feed", "text_ordinal", "text_numeric", "ram_tag"],
                        help="MUST match training. direct_feed (default) = learned-vector "
                             "blend. text_ordinal/text_numeric = literal sentence built from "
                             "router intensities, fed through the frozen CLIP text encoder — "
                             "see _compose_lgdp_prompt_text in pisasr.py. ram_tag = frozen RAM "
                             "tagger runs on the LR image, tag string fed through the frozen "
                             "CLIP text encoder — see _compose_lgdp_prompt_ram in pisasr.py.")
    parser.add_argument("--ram_pretrained_path", type=str,
                        default="/mnt/HDD1/tuong/SR/PiSA-SR_med/src/ram_pretrain_model/ram_swin_large_14m.pth",
                        help="Checkpoint for the RAM tagger, only loaded when "
                             "--lgdp_injection_mode ram_tag.")
    parser.add_argument("--lgdp_text_phrase_style", type=str, default="degradation_noun",
                        choices=["degradation_noun", "restoration_verb"],
                        help="MUST match training. Only used by text_ordinal/text_numeric. "
                             "degradation_noun (default) = 'depth blur' style; "
                             "restoration_verb = 'deblur' style — see pisasr.py.")
    parser.add_argument("--lgdp_token_scheme", type=str, default="full6",
                        choices=["full6", "merged3"],
                        help="MUST match training: full6 (default, 6 restoration tokens, one "
                             "per physical degradation type) or merged3 (3 tokens, grouped by "
                             "shared frequency content — see pisasr.py DEG_TYPE_TO_GROUP_MERGED3). "
                             "Asserted against the checkpoint's saved value.")
    parser.add_argument("--restoration_token_init_scale", type=float, default=1.0,
                        help="Initial value for restoration_token_scale (must match training).")
    parser.add_argument("--lgdp_prompt_strength", type=float, default=1.0,
                        help="Global scale applied to the 6 weighted restoration tokens that, "
                             "together with BOS, form the entire UNet prompt (0=degenerate to BOS-only).")
    parser.add_argument("--lgdp_prompt_length", type=int, default=7, choices=[7, 77],
                        help="Sequence length of the UNet's text condition — MUST match the "
                             "value the checkpoint was trained with (7=direct-feed [BOS+6 "
                             "tokens], 77=padded to CLIP's native context length).")
    parser.add_argument("--lgdp_variable_tokens", type=lambda x: x.lower() != "false",
                        default=False,
                        help="Must match training: whether restoration_tokens uses the "
                             "variable-length per-degradation content-token scheme instead "
                             "of 1-EOS-pooled-token-per-degradation. NOTE: the checkpoint's "
                             "actual restoration_tokens/token_segment_lengths are always "
                             "loaded regardless of this flag (auto-detected) — this only "
                             "matters if you're loading an OLD checkpoint that lacks the "
                             "token_segment_lengths key.")
    parser.add_argument("--domain_prompt", type=str, default="",
                        help="Optional static domain-description text spliced into the LGDP "
                             "sequence as [BOS] + [6 degrad tokens] + [domain prompt content] "
                             "+ [EOS]. LEARNABLE and saved in the checkpoint — if the "
                             "checkpoint has a trained domain prompt, its value is loaded "
                             "regardless of this flag; only used as a fallback placeholder "
                             "for older checkpoints without one (default \"\" = none).")
    parser.add_argument("--router_arch", choices=["base", "conv"], default="base",
                        help="Must match the checkpoint's training-time --router_arch "
                             "(PiSASR_eval asserts this) — see pisasr.py's ConvRouterHead.")
    parser.add_argument("--debug_router", action="store_true",
                        help="Print router intensities per image.")
    parser.add_argument("--router_gain", type=float, default=1.0,
                        help="Scalar multiplier applied to the router's OWN per-image "
                             "predicted intensities (preserves relative per-image/per-slot "
                             "variation, unlike --force_router_intensity which replaces the "
                             "prediction entirely). Default 1.0 = no change. See "
                             "PiSASR_eval.forward in pisasr.py.")
    parser.add_argument("--router_gain_slot", type=int, default=None,
                        help="Index (0-5) of a single degradation slot to apply "
                             "--router_gain_slot_value to, leaving the other 5 slots at their "
                             "raw/--router_gain-scaled prediction. Order: 0=psf_blur, 1=haze, "
                             "2=attenuation, 3=speckle, 4=additive_noise, 5=compression.")
    parser.add_argument("--router_gain_slot_value", type=float, default=1.0,
                        help="Gain multiplier applied only to --router_gain_slot's intensity.")
    parser.add_argument("--router_gain_vec", type=str, default=None,
                        help="Comma-separated 6-value per-slot gain multiplier, e.g. "
                             "'0.2,5,0.2,5,5,5' (order: psf_blur,haze,attenuation,speckle,"
                             "additive_noise,compression). Generalizes --router_gain_slot to "
                             "boost/suppress several slots at once.")
    parser.add_argument("--router_force_slot", type=int, default=None,
                        help="Index (0-5) of a single degradation slot to REPLACE (not scale) "
                             "with --router_force_slot_value for every image, leaving the "
                             "other 5 slots at their real per-image router prediction. Order: "
                             "0=psf_blur, 1=haze, 2=attenuation, 3=speckle, 4=additive_noise, "
                             "5=compression.")
    parser.add_argument("--router_force_slot_value", type=float, default=1.0,
                        help="Fixed value --router_force_slot is pinned to (0-1).")
    parser.add_argument("--force_router_intensity", type=float, default=None,
                        help="If set, bypass the router and force all 6 degradation "
                             "intensities to this constant value for every image (e.g. "
                             "1.0 = max intensity everywhere). Diagnostic only: tests "
                             "whether the router's per-image prediction matters at "
                             "inference vs. a fixed constant.")
    parser.add_argument("--force_router_intensity_vec", type=str, default=None,
                        help="If set, bypass the router and force the 6 degradation "
                             "intensities to this comma-separated 6-value vector for every "
                             "image, e.g. '1,0,0,0,0,0' for a psf_blur-only one-hot sweep "
                             "(order: psf_blur,haze,attenuation,speckle,additive_noise,"
                             "compression). Takes priority over --force_router_intensity.")
    parser.add_argument("--gate_router_intensity_vec", type=str, default=None,
                        help="If set, keep the router's real per-image predicted intensity "
                             "but zero out any degradation slot not in this comma-separated "
                             "6-value 0/1 gate mask, e.g. '1,1,0,0,0,0' to activate only "
                             "psf_blur+haze correction with the router's own magnitude for "
                             "each (order: psf_blur,haze,attenuation,speckle,additive_noise,"
                             "compression). Unlike --force_router_intensity_vec this does NOT "
                             "clamp magnitude to a constant. Takes priority over both other "
                             "force flags.")
    args = parser.parse_args()

    # Call the processing function
    pisa_sr(args)
