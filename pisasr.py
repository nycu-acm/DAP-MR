import os
import sys
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules
from peft.utils.other import transpose
import torchvision.transforms.functional as TF
import glob
sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.datasets.esrgan_physic_online import generate_labeled_lr, generate_labeled_lr_dosed, generate_labeled_lr_random_order
from src.my_utils.vaehook import VAEHook

DEGRADATION_NAMES = [
    "psf_blur",
    "haze",
    "attenuation",
    "speckle",
    "additive_noise",
    "compression",
]

TARGET_STATE_PROMPTS = [
    # 0 psf_blur
    "sharp focused ultrasound image with clear tissue boundaries and focused anatomical structures",
    # 1 haze (anechoic clutter)
    "ultrasound image with true anechoic black regions and minimal background haze or clutter",
    # 2 attenuation
    "ultrasound image with balanced depth brightness and consistent deep tissue visibility",
    # 3 speckle
    "clean ultrasound image with reduced speckle noise and natural granular tissue texture",
    # 4 additive_noise
    "clean ultrasound image with low electronic noise and smooth background",
    # 5 compression
    "high resolution ultrasound image with preserved fine details and minimal compression artifacts",
]

INSTRUCTION_PROMPTS = [
    "restore sharp focus and clear tissue boundary in ultrasound",
    "suppress anechoic haze and clutter, restore true black background",
    "compensate depth attenuation and restore deep tissue brightness",
    "suppress speckle noise while preserving anatomical texture",
    "remove electronic sensor noise and recover clean ultrasound signal",
    "recover high resolution detail and remove compression artifact",
]

DEGRADATION_NOUN_PROMPTS = [
    "depth defocus blur, soft edge, out of focus ultrasound",
    "anechoic haze, background clutter, washed out black region ultrasound",
    "deep tissue attenuation, dark distal region, signal loss",
    "speckle noise, granular echo texture, coherent ultrasound noise",
    "electronic sensor noise, receiver noise, random pixel noise ultrasound",
    "low resolution, compression artifact, pixelated ultrasound",
]

# Negative control: real CLIP text embeddings (not Gaussian noise) but semantically
# unrelated to ultrasound/degradation, one per slot in DEGRADATION_NAMES order.
# Unlike mode="random" (torch.randn, bypasses the text encoder entirely), this tests
# whether ANY text injection helps vs. whether ultrasound/degradation-relevant
# semantics specifically are what matters.
NONSENSE_PROMPTS = [
    "a photo of a dog running in the park",
    "a cat sitting on a couch",
    "a red sports car on the highway",
    "a bowl of fresh fruit on a table",
    "a mountain landscape with clouds",
    "a cup of coffee on a wooden desk",
]

# --- --lgdp_token_scheme merged3 -------------------------------------------------------
# Groups the same 6 physical degradation generators (_apply_single_degradation, deg_type
# 0-5) into 3 restoration tokens, by shared frequency content: high-frequency detail loss
# (blur_detail), low-frequency photometric/brightness distortion (photometric), and
# stochastic texture noise (texture_noise). The 6 physics functions and their deg_type
# indices are UNCHANGED — only the text-prototype/restoration-token/router-output count
# and the per-type label -> slot assignment change. See DEG_TYPE_TO_GROUP_MERGED3 for the
# deg_type -> group-slot mapping and pisasr-3token-scheme memory for the rationale.
DEGRADATION_NAMES_MERGED3 = ["blur_detail", "photometric", "texture_noise"]

# deg_type (index into DEGRADATION_NAMES / _apply_single_degradation) -> group slot index
# (index into DEGRADATION_NAMES_MERGED3). [psf_blur, haze, attenuation, speckle,
# additive_noise, compression] -> [blur_detail, photometric, photometric, texture_noise,
# texture_noise, blur_detail].
DEG_TYPE_TO_GROUP_MERGED3 = [0, 1, 1, 2, 2, 0]

TARGET_STATE_PROMPTS_MERGED3 = [
    # 0 blur_detail (psf_blur + compression)
    "sharp high resolution ultrasound image with clear focused tissue boundaries, "
    "preserved fine anatomical detail, and minimal compression artifacts",
    # 1 photometric (haze + attenuation)
    "ultrasound image with true anechoic black regions, minimal background haze, "
    "balanced depth brightness, and consistent deep tissue visibility",
    # 2 texture_noise (speckle + additive_noise)
    "clean ultrasound image with reduced speckle noise, low electronic noise, "
    "natural granular tissue texture, and a smooth background",
]

INSTRUCTION_PROMPTS_MERGED3 = [
    "restore sharp focus and high resolution detail, remove compression artifact in ultrasound",
    "suppress anechoic haze and clutter, compensate depth attenuation, restore true black "
    "background and deep tissue brightness",
    "suppress speckle noise and electronic sensor noise while preserving anatomical texture",
]

DEGRADATION_NOUN_PROMPTS_MERGED3 = [
    "depth defocus blur, soft edge, low resolution, compression artifact, pixelated ultrasound",
    "anechoic haze, background clutter, washed out black region, deep tissue attenuation, "
    "dark distal region, signal loss ultrasound",
    "speckle noise, granular echo texture, coherent ultrasound noise, electronic sensor "
    "noise, receiver noise, random pixel noise",
]

NONSENSE_PROMPTS_MERGED3 = [
    "a photo of a dog running in the park",
    "a red sports car on the highway",
    "a mountain landscape with clouds",
]

# --lgdp_injection_mode text_ordinal / text_numeric ----------------------------------
# Short noun phrases, one per degradation slot, reusing the exact vocabulary already
# used by this file's own CSD negative prompt (see the `neg_ids` string in forward()/
# forward_curriculum()) for consistency. Used to build a literal, human-readable
# sentence from the router's [B,K] intensities instead of feeding a learned-vector
# blend (_compose_lgdp_prompt) -- see _build_lgdp_text_prompt / _compose_lgdp_prompt_text.
NOUN_PHRASES_FULL6 = [
    "depth blur",          # 0 psf_blur
    "background haze",     # 1 haze
    "signal attenuation",  # 2 attenuation
    "speckle noise",       # 3 speckle
    "electronic noise",    # 4 additive_noise
    "compression artifact",  # 5 compression
]

NOUN_PHRASES_MERGED3 = [
    "depth blur and compression artifact",  # 0 blur_detail
    "background haze and signal attenuation",  # 1 photometric
    "speckle noise and electronic noise",  # 2 texture_noise
]

# --lgdp_text_phrase_style restoration_verb -------------------------------------------
# Alternative to NOUN_PHRASES_FULL6/MERGED3 for --lgdp_injection_mode text_ordinal/
# text_numeric (2026-07-25, user request): describes the ACTION to take (restoration
# verb) instead of the degradation itself (noun). Standard image-restoration-literature
# terms where one exists (deblur, dehaze, despeckle, denoise, deblock); "compensate
# attenuation" has no single-word equivalent so kept as a short verb phrase, matching
# INSTRUCTION_PROMPTS' wording for that slot.
RESTORATION_VERB_PHRASES_FULL6 = [
    "deblur",                  # 0 psf_blur
    "dehaze",                  # 1 haze
    "compensate attenuation",  # 2 attenuation
    "despeckle",                # 3 speckle
    "denoise",                  # 4 additive_noise
    "deblock",                  # 5 compression
]

RESTORATION_VERB_PHRASES_MERGED3 = [
    "deblur and deblock",                    # 0 blur_detail
    "dehaze and compensate attenuation",     # 1 photometric
    "despeckle and denoise",                 # 2 texture_noise
]


def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


def _structured_spatial_pool(spatial_pool_layer, tokens, H_tok=None, W_tok=None):
    """
    Structured spatial pooling for degradation detection.
    tokens: [B, L, D], where L patches are in raster order (row-major).
    Returns [B, D] feature vector via spatial_pool_layer(6*D → D).

    H_tok, W_tok: token grid dimensions from the unfold call that produced `tokens`.
    Pass explicitly for non-square latents (unfold with kernel=3, stride=2, pad=1:
    H_tok = (H_lat - 1)//2 + 1). Falls back to sqrt(L) for square inputs.

    6 components:
      global mean, global max, global std          — overall appearance (also picks up
                                                       haze/clutter's floor-raising in dark
                                                       regions and speckle/noise contrast)
      row_bot_mean - row_top_mean                  — attenuation gradient (bottom darker)
      col_min                                      — darkest column (general brightness-floor cue)
      col_std                                      — column brightness variation
    """
    B, L, D = tokens.shape
    if H_tok is None:
        H_tok = int(L ** 0.5)
    if W_tok is None:
        W_tok = L // H_tok
    t2    = tokens[:, :H_tok * W_tok, :].reshape(B, H_tok, W_tok, D)  # [B, H_tok, W_tok, D]
    row_m = t2.mean(dim=2)   # [B, H_tok, D] — mean across cols per row
    col_m = t2.mean(dim=1)   # [B, W_tok, D] — mean across rows per col
    return spatial_pool_layer(torch.cat([
        tokens.mean(1),                                                    # global mean
        tokens.max(1).values,                                              # global max
        tokens.std(1),                                                     # global std
        row_m[:, H_tok // 2:, :].mean(1) - row_m[:, :H_tok // 2, :].mean(1),  # attenuation signal
        col_m.min(1).values,                                               # darkest col
        col_m.std(1),                                                      # col variation
    ], dim=-1))


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def compute_degradation_text_prototypes(
    text_encoder,
    tokenizer,
    device,
    mode="target_state",
    return_null_centered=True,
    scheme="full6",
):
    """Compute [1, K, D] degradation text-prototype tensor using EOS pooling.

    Supported modes:
        "target_state"      -- proposed default: target-state visual descriptions.
        "instruction"       -- ablation: imperative instruction prompts.
        "degradation_noun"  -- ablation: degradation noun descriptions.
        "correct"           -- alias for "instruction" (legacy name).
        "pos"               -- legacy: clean-state descriptions. full6 scheme only.
        "neg"               -- legacy: degraded-state descriptions. full6 scheme only.
        "random"            -- ablation: random init, no CLIP text involved.
        "nonsense"          -- ablation: real CLIP text embeddings, semantically
                               unrelated to ultrasound (negative control with text
                               structure, unlike "random"'s raw Gaussian noise).
        "short_noun"        -- ablation: the same short 2-word noun phrases used by
                               --lgdp_injection_mode text_ordinal/text_numeric
                               (NOUN_PHRASES_FULL6/MERGED3), instead of the longer
                               multi-clause DEGRADATION_NOUN_PROMPTS.

    Uses EOS pooling (not mean pooling): argmax(input_ids) gives the CLIP EOS index,
    which is the sentence-level semantic summary CLIP uses internally.

    If return_null_centered=True, each prototype is centered by subtracting the null
    ("") EOS embedding, placing prototypes on the text-direction manifold.

    scheme: "full6" (default) -- K=6, one token per DEGRADATION_NAMES entry. "merged3"
    -- K=3, one token per DEGRADATION_NAMES_MERGED3 group (see DEG_TYPE_TO_GROUP_MERGED3).

    Returns [1, K, D], K depending on `scheme`.
    """
    ctx_dim = text_encoder.config.hidden_size
    K = 6 if scheme == "full6" else 3

    if mode == "random":
        return torch.randn(1, K, ctx_dim, device=device)

    if scheme == "full6":
        if mode == "target_state":
            texts = list(TARGET_STATE_PROMPTS)
        elif mode in ("instruction", "correct"):
            texts = list(INSTRUCTION_PROMPTS)
        elif mode == "degradation_noun":
            texts = list(DEGRADATION_NOUN_PROMPTS)
        elif mode == "short_noun":
            texts = list(NOUN_PHRASES_FULL6)
        elif mode == "nonsense":
            texts = list(NONSENSE_PROMPTS)
        elif mode == "pos":
            texts = [
                "sharp focus, clear edge, no depth blur ultrasound",
                "no background haze, true black anechoic regions, no clutter",
                "uniform brightness, no depth attenuation, bright deep tissue",
                "clean echo, no speckle noise, smooth tissue texture",
                "noise free, clean signal, no electronic noise ultrasound",
                "high resolution, sharp detail, no compression artifact",
            ]
        elif mode == "neg":
            texts = [
                "depth defocus blur, out of focus, soft edge ultrasound",
                "anechoic haze, background clutter, washed out black region ultrasound",
                "deep tissue attenuation, dark distal region, signal loss",
                "speckle noise, granular echo texture, coherent noise",
                "electronic noise, receiver noise, random pixel noise ultrasound",
                "low resolution, compression artifact, pixelated ultrasound",
            ]
        else:
            texts = None
    elif mode == "target_state":
        texts = list(TARGET_STATE_PROMPTS_MERGED3)
    elif mode in ("instruction", "correct"):
        texts = list(INSTRUCTION_PROMPTS_MERGED3)
    elif mode == "degradation_noun":
        texts = list(DEGRADATION_NOUN_PROMPTS_MERGED3)
    elif mode == "short_noun":
        texts = list(NOUN_PHRASES_MERGED3)
    elif mode == "nonsense":
        texts = list(NONSENSE_PROMPTS_MERGED3)
    elif mode in ("pos", "neg"):
        raise ValueError(
            f"mode={mode!r} has no merged3 variant (legacy, only defined for scheme="
            "'full6') -- pick target_state/instruction/degradation_noun/short_noun/nonsense/random.")
    else:
        texts = None

    if texts is None:
        raise ValueError(
            f"Unknown mode: {mode!r} "
            "(expected 'target_state', 'instruction', 'degradation_noun', 'short_noun', "
            "'correct', 'pos', 'neg', 'random', or 'nonsense')"
        )

    # EOS pooling: argmax(input_ids) is the canonical CLIP EOS index.
    # Mean pooling is dominated by padding tokens; EOS is the sentence-level summary.
    def eos_feat(text):
        ids = tokenizer(
            text, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        emb = text_encoder(ids)[0]          # [1, 77, D]
        eos = ids[0].argmax().item()        # CLIP EOS position
        return emb[:, eos:eos + 1, :]      # [1, 1, D]

    if return_null_centered:
        null_emb = eos_feat("")
        protos = [eos_feat(t) - null_emb for t in texts]   # each [1, 1, D]
    else:
        protos = [eos_feat(t) for t in texts]

    return torch.cat(protos, dim=1)  # [1, K, D]


def encode_domain_prompt(tokenizer, text_encoder, device, text):
    """Encode a static domain-prompt string (e.g. "ultrasound image") into the pieces
    needed to splice it into the LGDP sequence as [... degrad tokens] + [domain content]
    + [EOS]: content tokens (this text's own real words, excluding ITS OWN BOS/EOS) and
    its EOS embedding (sentence-level closing token, contextualized on the domain text).

    Empty string -> both pieces are zero-length/unused; callers should skip appending them
    (this is the default, so existing runs/checkpoints with no --domain_prompt are unaffected).

    Returns:
        content: [1, L, D] (L can be 0)
        eos:     [1, 1, D]
    """
    ctx_dim = text_encoder.config.hidden_size
    if not text or not text.strip():
        return (torch.zeros(1, 0, ctx_dim, device=device),
                torch.zeros(1, 1, ctx_dim, device=device))
    ids = tokenizer(
        text, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    embeds = text_encoder(ids)[0]              # [1, 77, D]
    eos_pos = ids[0].argmax().item()            # CLIP EOS position
    content = embeds[:, 1:eos_pos, :]            # [1, L, D], excludes this text's BOS/EOS
    eos = embeds[:, eos_pos:eos_pos + 1, :]      # [1, 1, D]
    return content, eos


def load_ram_tagger(pretrained_path, device):
    """Load the frozen RAM (Recognize Anything Model) tagger for --lgdp_injection_mode
    ram_tag (2026-08-15, user request: 4th variant of a token/prompt-scheme ablation
    against the router-driven direct_feed/text_ordinal/text_numeric schemes). Real
    per-image tags replace the router's degradation-intensity-driven prompt entirely —
    the router is not needed/trained at all in this mode (training scripts set
    --router_pretrain_steps 0 --lambda_router_dose_online 0.0 --teacher_forcing_anneal_frac
    0.0 so it's never even called). Mirrors TVT's online per-step RAM-captioning pattern
    (ram_lora.ram, swin_l backbone, fp16, frozen/eval).
    """
    from ram.models.ram_lora import ram as _ram_ctor
    model = _ram_ctor(pretrained=pretrained_path, pretrained_condition=None,
                       image_size=384, vit='swin_l')
    model.eval()
    model.requires_grad_(False)
    model.to(device, dtype=torch.float16)
    return model


def build_ram_tag_prompt(ram_model, x_img):
    """x_img: [B,C,H,W] pixel image in [-1,1] -> list of B tag-caption strings, one per
    sample. Uses RAM's OWN preprocessing (384x384 resize + ImageNet normalize), not the
    diffusion model's [-1,1] convention. Non-differentiable (discrete tag generation),
    same rationale as _build_lgdp_text_prompt's cut gradient for text_ordinal/text_numeric.
    """
    from ram import inference_ram as _inference_ram
    with torch.no_grad():
        ram_input = TF.resize((x_img.detach().float() * 0.5 + 0.5).clamp(0, 1), [384, 384],
                               antialias=True)
        ram_input = TF.normalize(ram_input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ram_device = next(ram_model.parameters()).device
        tags = _inference_ram(ram_input.to(ram_device, dtype=torch.float16), ram_model)
    return [f"{t}, ultrasound image" for t in tags]


def compute_degradation_content_tokens(text_encoder, tokenizer, device, mode="target_state",
                                        scheme="full6"):
    """Variable-length alternative to compute_degradation_text_prototypes: instead of
    EOS-pooling each degradation's text into a SINGLE token, keep ALL of that text's
    content tokens (its own real words, excluding its own BOS/EOS) and concatenate the K
    degradations' token blocks into one sequence. A degradation with a longer description
    gets more tokens in the prompt, not the same 1-token budget as a short one.

    scheme: "full6" (default) -- K=6 degradation blocks. "merged3" -- K=3 group blocks.

    Returns:
        concat_tokens: [1, L_total, D] (L_total = sum of each block's token count,
                        safety-truncated to at most 76 = 77 - 1(BOS), so [BOS]+concat_tokens
                        never exceeds CLIP's native 77-length context even without padding)
        segment_lengths: list of K ints, concat_tokens[:, sum(segment_lengths[:i]):
                          sum(segment_lengths[:i+1]), :] is block i's own block —
                          caller uses this to broadcast each degradation's router intensity
                          across its own block instead of one intensity per token.
    """
    if scheme == "full6":
        bank = {"target_state": TARGET_STATE_PROMPTS, "instruction": INSTRUCTION_PROMPTS,
                "correct": INSTRUCTION_PROMPTS, "degradation_noun": DEGRADATION_NOUN_PROMPTS,
                "short_noun": NOUN_PHRASES_FULL6,
                "nonsense": NONSENSE_PROMPTS}
    else:
        bank = {"target_state": TARGET_STATE_PROMPTS_MERGED3,
                "instruction": INSTRUCTION_PROMPTS_MERGED3,
                "correct": INSTRUCTION_PROMPTS_MERGED3,
                "degradation_noun": DEGRADATION_NOUN_PROMPTS_MERGED3,
                "short_noun": NOUN_PHRASES_MERGED3,
                "nonsense": NONSENSE_PROMPTS_MERGED3}
    if mode in bank:
        texts = list(bank[mode])
    else:
        raise ValueError(
            f"Unknown mode: {mode!r} for compute_degradation_content_tokens "
            "(expected 'target_state', 'instruction', 'degradation_noun', 'short_noun', 'correct', "
            "or 'nonsense' — 'random'/'pos'/'neg' have no natural per-word token content)"
        )

    blocks, segment_lengths = [], []
    for text in texts:
        content, _ = encode_domain_prompt(tokenizer, text_encoder, device, text)  # [1, L_i, D]
        blocks.append(content)
        segment_lengths.append(content.shape[1])

    concat_tokens = torch.cat(blocks, dim=1)  # [1, L_total, D]

    max_len = 76  # 77 - 1(BOS); leaves no room for a domain prompt in this mode
    if concat_tokens.shape[1] > max_len:
        overflow = concat_tokens.shape[1] - max_len
        print(f"[compute_degradation_content_tokens] mode={mode!r} total content "
              f"{concat_tokens.shape[1]} tokens > {max_len} budget — truncating "
              f"{overflow} trailing tokens off the last degradation's block.", flush=True)
        concat_tokens = concat_tokens[:, :max_len, :]
        segment_lengths[-1] = max(0, segment_lengths[-1] - overflow)

    return concat_tokens, segment_lengths


class DegradationRouter(nn.Module):
    """Single-head regression router.
    intensity_head → SmoothL1 against continuous labels in [0, 1]
    Inference intensity = sigmoid(intensity_head(x))
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_deg: int = 6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.intensity_head = nn.Linear(hidden_dim // 2, num_deg)

    def forward(self, x):
        return torch.sigmoid(self.intensity_head(self.backbone(x)))


class ConvRouterHead(nn.Module):
    """--router_arch conv: CNN run directly over the [B,4,H,W] VAE latent, instead of
    unfold-to-patches -> vis_projector -> 6-global-statistic pool -> DegradationRouter MLP
    (--router_arch base, the original). Global mean/max/std pooling over patch tokens throws
    away local spatial structure — degradations like speckle/psf blur ARE local texture
    statistics, so no amount of MLP depth downstream of that pool can recover what the pool
    already destroyed. This keeps spatial structure through several conv stages and only
    pools (mean AND std, so texture/noise magnitude survives) at the very end.

    Standalone diagnostic (router_indomain_outdomain.py, see memory
    pisasr-degradation-router) found this beats the base router on BOTH in-domain accuracy
    AND out-of-domain generalization to unseen anatomies, using ~1/3 the parameters —
    trained on Cardiac only in both cases, so this is not a data/domain-mixing effect.
    """

    def __init__(self, in_ch=4, width=128, num_deg=6):
        super().__init__()

        def blk(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.GroupNorm(8, cout),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            blk(in_ch, width),
            blk(width, width),
            blk(width, width * 2, 2),
            blk(width * 2, width * 2),
            blk(width * 2, width * 2, 2),
            blk(width * 2, width * 2),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 4, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, num_deg),
        )
        nn.init.constant_(self.head[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12, matches base router's init

    def forward(self, latent):
        f = self.net(latent.to(next(self.net.parameters()).dtype))
        gap = f.mean(dim=[2, 3])
        gsd = f.std(dim=[2, 3])
        return torch.sigmoid(self.head(torch.cat([gap, gsd], dim=1)))


def initialize_unet(rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break

    lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_pix)
    lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_modules_others_pix)
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)

    unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):

        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)


        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()

class PiSASR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        # --lgdp_injection_mode ram_tag: load the frozen RAM tagger once (see
        # load_ram_tagger/build_ram_tag_prompt above). Only loaded in this mode so the
        # other 3 injection modes pay zero extra VRAM/import cost.
        if getattr(args, "lgdp_injection_mode", "direct_feed") == "ram_tag":
            self.ram_model = load_ram_tagger(args.ram_pretrained_path, self.text_encoder.device)

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem, =\
                    initialize_unet(rank_pix=args.lora_rank_unet_pix, rank_sem=args.lora_rank_unet_sem, pretrained_model_path=args.pretrained_model_path, return_lora_module_names=True)

            self.lora_rank_unet_pix = args.lora_rank_unet_pix
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
                self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
                lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                    lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem
        else:
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_pix = stage1_args.lora_rank_unet_pix
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem

        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to('cuda')
        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

        # No domain/base prompt: the UNet's only text condition is [BOS] + the 6 degradation-
        # restoration tokens (see _compose_lgdp_prompt) — no null-prompt carrier, no additive
        # injection. BOS is taken from encoding "" once (frozen text encoder, no gradient) —
        # this is the contextualized attention-sink anchor token, cached as a buffer.
        with torch.no_grad():
            null_ids = self.tokenizer(
                "", max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            self.register_buffer("null_prompt_embeds", self.text_encoder(null_ids)[0])  # [1, 77, D]

        # Optional static domain prompt (e.g. "ultrasound image"), spliced in AFTER the
        # degradation tokens and BEFORE the closing EOS: [BOS] + [6 degrad tokens] +
        # [domain prompt content] + [EOS] + [PAD...]. LEARNABLE (nn.Parameter): initialized
        # from the frozen text encoder's encoding of --domain_prompt (like restoration_tokens
        # is initialized from text_prototypes), then trained jointly with the LoRA weights via
        # the main SR loss (see train_pisasr.py's domain_prompt_params optimizer group) — NOT
        # frozen like text_prototypes/restoration_tokens/restoration_token_scale. --domain_prompt
        # default "" disables it entirely (content has 0 length -> _compose_lgdp_prompt skips
        # it), zero effect on existing runs/checkpoints.
        with torch.no_grad():
            domain_content, domain_eos = encode_domain_prompt(
                self.tokenizer, self.text_encoder, self.text_encoder.device,
                getattr(args, "domain_prompt", ""),
            )
        self.domain_prompt_content = nn.Parameter(domain_content)  # [1, L, D]
        self.domain_prompt_eos = nn.Parameter(domain_eos)          # [1, 1, D]

        # --lgdp_token_scheme selects how many restoration tokens/router outputs exist:
        #  - "full6" (default): one token per DEGRADATION_NAMES entry (psf_blur, haze,
        #    attenuation, speckle, additive_noise, compression).
        #  - "merged3": one token per DEGRADATION_NAMES_MERGED3 group (blur_detail,
        #    photometric, texture_noise) — see DEG_TYPE_TO_GROUP_MERGED3. The 6 underlying
        #    physics functions (_apply_single_degradation) are unchanged; only the per-type
        #    severity label is max-pooled into its group's slot before it reaches the router
        #    loss / prompt injection (see _new_label_vec / _set_label).
        self.lgdp_token_scheme = getattr(args, "lgdp_token_scheme", "full6")
        # --lgdp_text_phrase_style: only consulted by _build_lgdp_text_prompt (text_ordinal/
        # text_numeric injection modes) — which vocabulary names each degradation slot with.
        self.lgdp_text_phrase_style = getattr(args, "lgdp_text_phrase_style", "degradation_noun")
        if self.lgdp_token_scheme not in ("full6", "merged3"):
            raise ValueError(f"Unknown --lgdp_token_scheme {self.lgdp_token_scheme!r} "
                              "(expected 'full6' or 'merged3')")
        if self.lgdp_token_scheme == "full6":
            self.num_deg = len(DEGRADATION_NAMES)  # 6
            self.deg_group_map = list(range(6))
            self.deg_names = list(DEGRADATION_NAMES)
        else:
            self.num_deg = len(DEGRADATION_NAMES_MERGED3)  # 3
            self.deg_group_map = list(DEG_TYPE_TO_GROUP_MERGED3)
            self.deg_names = list(DEGRADATION_NAMES_MERGED3)

        # LGDP restoration tokens — two mutually exclusive schemes:
        #  - default: EOS-pooled TEXT PROTOTYPES [1, num_deg, D], one token per degradation
        #    (NOT null-centered — these stand alone as the prompt now, not as offsets added
        #    onto an existing base, so they need their own full-magnitude semantic direction).
        #  - --lgdp_variable_tokens: each degradation keeps its FULL content-token block
        #    (variable length, longer description = more tokens) instead of 1 pooled token;
        #    see compute_degradation_content_tokens. text_prototypes has no role in this mode
        #    (there's no single pooled token to seed from).
        self.lgdp_variable_tokens = bool(getattr(args, "lgdp_variable_tokens", False))
        prompt_bank_mode = getattr(args, "prompt_bank_mode", "target_state")
        if self.lgdp_variable_tokens:
            self.text_prototypes = None
            with torch.no_grad():
                concat_tokens, self.token_segment_lengths = compute_degradation_content_tokens(
                    self.text_encoder, self.tokenizer, self.text_encoder.device,
                    mode=prompt_bank_mode, scheme=self.lgdp_token_scheme,
                )
            self.restoration_tokens = nn.Parameter(concat_tokens)  # [1, L_total, D]
        else:
            self.token_segment_lengths = None
            with torch.no_grad():
                self.text_prototypes = nn.Parameter(
                    compute_degradation_text_prototypes(
                        self.text_encoder, self.tokenizer, self.text_encoder.device,
                        mode=prompt_bank_mode, return_null_centered=False,
                        scheme=self.lgdp_token_scheme,
                    )
                )  # [1, num_deg, D]
            # LGDP: learnable restoration tokens initialized from text_prototypes; these ARE
            # the UNet prompt (weighted by router intensity), not an offset into a base prompt.
            self.restoration_tokens = nn.Parameter(self.text_prototypes.detach().clone())  # [1, K, D]

        # Store softplus_inv(init_scale) so that F.softplus(param) starts at init_scale.
        # This decouples magnitude (scale) from direction (restoration_tokens unit vector),
        # giving scale a clean gradient signal instead of being shadowed by restoration_tokens.
        # One scale per DEGRADATION (num_deg), broadcast across that degradation's own token
        # block in variable-tokens mode — not one scale per token.
        _rts_init = getattr(args, "restoration_token_init_scale", 1.0)
        _rts_raw  = torch.tensor(float(_rts_init)).expm1().clamp(min=1e-8).log().item()
        self.restoration_token_scale = nn.Parameter(
            torch.full((self.num_deg,), _rts_raw)
        )  # [K]; effective scale = softplus(this) ≥ 0

        self.ctx_dim = self.text_encoder.config.hidden_size

        # --router_arch selects the LR->intensity feature extractor:
        #  - "base" (default): 3x3-patch unfold -> vis_projector -> 6-global-statistic pool ->
        #    DegradationRouter MLP. Original PiSA-SR/LGDP router.
        #  - "conv": ConvRouterHead — CNN directly over the VAE latent (see class docstring).
        #    Beats "base" on in-domain AND out-of-domain accuracy with ~1/3 the params in the
        #    standalone router-only diagnostic (memory: pisasr-degradation-router).
        self.router_arch = getattr(args, "router_arch", "base")
        if self.router_arch == "conv":
            self.conv_router = ConvRouterHead(in_ch=4, num_deg=self.num_deg)
        else:
            # 4 channels × 3×3 patch = 36 features per token; lets router detect blur/noise spatially
            self.vis_projector = nn.Sequential(
                nn.Linear(36, 128),
                nn.GELU(),
                nn.Linear(128, self.ctx_dim),
            )
            self.vis_norm = nn.LayerNorm(self.ctx_dim)
            # Aggregates [mean, max, std] of patch tokens → ctx_dim.
            # Mean alone loses spatial structure (attenuation gradient, haze brightness floor);
            # max captures peak activations; std captures texture variance (speckle).
            self.spatial_pool = nn.Linear(self.ctx_dim * 6, self.ctx_dim)

            self.router = DegradationRouter(self.ctx_dim, num_deg=self.num_deg)
            nn.init.constant_(self.router.intensity_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12

        if args.resume_ckpt is not None:
            self.load_ckpt_from_state_dict(torch.load(args.resume_ckpt, map_location='cpu'))

    def router_parameters(self):
        """Trainable params of the router stack, whichever --router_arch is active — lets
        train_pisasr.py build its Phase-1 optimizer group without knowing the arch."""
        if self.router_arch == "conv":
            return list(self.conv_router.parameters())
        return (list(self.vis_projector.parameters()) + list(self.vis_norm.parameters())
                + list(self.spatial_pool.parameters()) + list(self.router.parameters()))

    def set_router_train(self):
        if self.router_arch == "conv":
            self.conv_router.train()
        else:
            self.vis_projector.train()
            self.vis_norm.train()
            self.router.train()

    def set_router_eval(self):
        if self.router_arch == "conv":
            self.conv_router.eval()
        else:
            self.vis_projector.eval()
            self.vis_norm.eval()
            self.router.eval()

    def _compute_intensities(self, latent_fp):
        """latent_fp: VAE latent [B,4,H,W] already cast to the router's param dtype.
        Dispatches on --router_arch (see __init__); every forward-pass call site (forward(),
        encode_degradation_prediction(), compute_router_loss()) goes through this so the two
        architectures can never silently drift apart."""
        if self.router_arch == "conv":
            return self.conv_router(latent_fp)
        _, _, H_lat, W_lat = latent_fp.shape
        H_tok = (H_lat - 1) // 2 + 1
        W_tok = (W_lat - 1) // 2 + 1
        patches = F.unfold(latent_fp, kernel_size=3, stride=2, padding=1).transpose(1, 2)
        tokens = self.vis_norm(self.vis_projector(patches))
        feat = _structured_spatial_pool(self.spatial_pool, tokens, H_tok, W_tok)
        return self.router(feat)

    def _router_param_dtype(self):
        if self.router_arch == "conv":
            return next(self.conv_router.parameters()).dtype
        return next(self.vis_projector.parameters()).dtype

    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n: _p.requires_grad = True
            if "sem" in n: _p.requires_grad = False

    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n: _p.requires_grad = True
            if "pix" in n: _p.requires_grad = False

    def _activate_pix_sem_adapters(self):
        """Activate all 6 pix/sem LoRA adapters (both halves must contribute to the forward
        computation in either phase — only their PARAMETER gradients should be gated by
        phase). PEFT's set_adapter() (called internally by set_weights_and_activate_adapters)
        has a documented side effect: it forces requires_grad_(True) on every adapter passed
        to it and requires_grad_(False) on the rest. Since this always lists BOTH pix and
        sem, calling it directly silently re-enables whichever half set_train_pix()/
        set_train_sem() just froze — on EVERY forward call, not just the moment right after
        the phase switch — completely defeating PiSA's two-phase pix-then-sem discipline.
        Snapshot + restore requires_grad around the call so it survives.
        """
        adapter_names = [
            "default_encoder_pix", "default_decoder_pix", "default_others_pix",
            "default_encoder_sem", "default_decoder_sem", "default_others_sem",
        ]
        _grad_state = {n: p.requires_grad for n, p in self.unet.named_parameters()
                       if "pix" in n or "sem" in n}
        set_weights_and_activate_adapters(self.unet, adapter_names, [1.0] * len(adapter_names))
        for n, p in self.unet.named_parameters():
            if n in _grad_state:
                p.requires_grad_(_grad_state[n])

    def load_ckpt_from_state_dict(self, sd):
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"], \
            sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        # Asserted BEFORE text_prototypes/router are touched below — see PiSASR_eval's
        # mirror of this for why (fail with a clear message, not a bare shape-mismatch
        # RuntimeError from whichever copy_/load_state_dict call happens to run first).
        ckpt_router_arch = sd.get("router_arch", "base")
        assert ckpt_router_arch == self.router_arch, (
            f"--resume_ckpt was trained with router_arch={ckpt_router_arch!r} but this run "
            f"was started with --router_arch {self.router_arch!r} — the two architectures "
            f"have incompatible weights, resuming would silently load nothing (or crash).")
        ckpt_token_scheme = sd.get("lgdp_token_scheme", "full6")
        assert ckpt_token_scheme == self.lgdp_token_scheme, (
            f"--resume_ckpt was trained with lgdp_token_scheme={ckpt_token_scheme!r} but "
            f"this run was started with --lgdp_token_scheme {self.lgdp_token_scheme!r} — "
            f"the router/restoration-token shapes are incompatible, resuming would crash "
            f"or silently corrupt weights.")

        # Router and degradation-prompt state. text_prototypes is None in
        # --lgdp_variable_tokens mode (no single pooled token to store) — skip it there.
        if self.text_prototypes is not None and sd.get("text_prototypes") is not None:
            self.text_prototypes.data.copy_(sd["text_prototypes"])
        if self.router_arch == "conv":
            self.conv_router.load_state_dict(sd["conv_router"])
        else:
            self.router.load_state_dict(sd["router"])
            self.vis_projector.load_state_dict(sd["vis_projector"])
            self.vis_norm.load_state_dict(sd["vis_norm"])
            self.spatial_pool.load_state_dict(sd["spatial_pool"])
        # restoration_tokens' shape depends on the token scheme (fixed [1,6,D] vs variable
        # [1,L_total,D]) — reassign outright instead of copy_ so a resume with a different
        # scheme/bank doesn't crash on a shape mismatch (mirrors domain_prompt's pattern).
        if sd["restoration_tokens"].shape == self.restoration_tokens.shape:
            self.restoration_tokens.data.copy_(sd["restoration_tokens"])
        else:
            self.restoration_tokens = nn.Parameter(sd["restoration_tokens"].to(self.restoration_tokens.device))
        self.restoration_token_scale.data.copy_(sd["restoration_token_scale"])
        if self.lgdp_variable_tokens and "token_segment_lengths" in sd:
            self.token_segment_lengths = sd["token_segment_lengths"]

        # Learnable domain prompt: restore the checkpoint's TRAINED values when resuming,
        # but only if its shape matches the current --domain_prompt's (re-derived above at
        # __init__ time) — a mismatch means this resume is using a different domain prompt
        # string/tokenization than the checkpoint was trained with, so keep the freshly
        # text-encoder-initialized value instead of crashing on a shape mismatch.
        if ("domain_prompt_content" in sd
                and sd["domain_prompt_content"].shape == self.domain_prompt_content.shape):
            self.domain_prompt_content.data.copy_(sd["domain_prompt_content"])
            self.domain_prompt_eos.data.copy_(sd["domain_prompt_eos"])

    def _compose_lgdp_prompt(self, intensities_pred, args):
        """Compose the UNet's entire text condition directly from the degradation prompt:
        [BOS] + 6 restoration tokens, each scaled by the router's predicted intensity for
        that degradation, optionally followed by a static domain prompt (--domain_prompt)
        + its own EOS: [BOS] + [degrad tokens] + [domain prompt content] + [EOS]. No
        null-prompt carrier, no additive injection — this short sequence IS the prompt,
        fed straight to cross-attention. BOS is kept as a fixed attention-sink anchor (its
        own contextualized "" encoding at position 0; diag_position_attention.py measured
        ~86% of UNet cross-attention mass lands on BOS) since dropping it entirely was
        assessed too risky without direct retraining evidence. With --domain_prompt unset
        (default ""), sequence length is 1+K and behavior is unchanged from before this
        option existed.

        Framing note (confirmed via diag_force_intensities_v2.py on a checkpoint trained
        with router+degradation-prompt frozen after Phase 1 — see memory
        lgdp-prompt-injection-inert-at-inference): earlier injection schemes (offset added
        into a base prompt) were a TRAINING-TIME conditioning signal into the jointly-
        optimized LoRA weights, not a live test-time magnitude controller. This direct-feed
        design has not yet been re-tested for that property; do not assume it is different
        without evidence.

        Args:
            intensities_pred: [B, K] router degradation intensities in [0, 1], used directly
                               as the per-degradation weight.

        Returns:
            dynamic_prompt: [B, 1+K, D]
            debug: dict with key "prompt_weights".
        """
        K = self.num_deg
        weights = intensities_pred.float()  # [B, K]
        B = weights.shape[0]

        # No F.normalize: restoration_tokens are raw (non-null-centered) CLIP EOS-pooled
        # sentence embeddings, so they already carry a natural, CLIP-consistent magnitude
        # (measured ~34.5 for this text encoder — same order as BOS's ~17.25, actually
        # somewhat larger) — no need to discard that direction+magnitude and reconstruct an
        # artificial one via unit-normalize + BOS-norm anchoring. Discarding the natural
        # magnitude (via F.normalize) and replacing it with a uniform scale/anchor is what
        # made every token's audibility identical regardless of its own semantic content;
        # using the raw embedding keeps both direction AND magnitude as CLIP produced them.
        bos = self.null_prompt_embeds[:, 0:1, :].expand(B, -1, -1)  # [B, 1, D]

        scale    = F.softplus(self.restoration_token_scale)  # [K]
        strength = float(getattr(args, "lgdp_prompt_strength", 1.0))

        if self.lgdp_variable_tokens:
            # Each degradation's router intensity/scale is broadcast across its OWN
            # variable-length token block (self.token_segment_lengths[i] tokens), not one
            # weight per token — a degradation with a longer description gets more tokens
            # in the prompt, all carrying that SAME degradation's single intensity value.
            weights_expanded = torch.cat(
                [weights[:, i:i + 1].expand(-1, n) for i, n in enumerate(self.token_segment_lengths)],
                dim=1,
            )  # [B, L_total]
            scale_expanded = torch.cat(
                [scale[i:i + 1].expand(n) for i, n in enumerate(self.token_segment_lengths)]
            )  # [L_total]
            tokens = (strength * weights_expanded.unsqueeze(-1) * scale_expanded.view(1, -1, 1).to(weights.dtype)
                      * self.restoration_tokens.to(weights.dtype))  # [B, L_total, D]
        else:
            tokens = (strength * weights.unsqueeze(-1) * scale.view(1, K, 1).to(weights.dtype)
                      * self.restoration_tokens.to(weights.dtype))  # [B, K, D]

        parts = [bos.to(tokens.dtype), tokens]
        # Optional domain prompt: [BOS] + [degrad tokens] + [domain content] + [EOS].
        # domain_prompt_content has 0 length when --domain_prompt is unset/empty, so this
        # is a no-op and the sequence is bit-identical to the no-domain-prompt case.
        if self.domain_prompt_content.shape[1] > 0:
            parts.append(self.domain_prompt_content.to(tokens.dtype).expand(B, -1, -1))
            parts.append(self.domain_prompt_eos.to(tokens.dtype).expand(B, -1, -1))

        dynamic_prompt = torch.cat(parts, dim=1).float()  # [B, 1+K(+domain+1), D]
        dynamic_prompt = self._pad_lgdp_prompt(dynamic_prompt, args)

        return dynamic_prompt, {"prompt_weights": weights}

    def _pad_lgdp_prompt(self, dynamic_prompt, args):
        """Optionally pad [BOS + K tokens] out to CLIP's native 77-token context length,
        using the tail of the null ("") prompt's own padding/EOS embeddings. A length-7
        sequence is out-of-distribution for cross-attention layers whose positional
        embeddings and attention statistics were calibrated on length-77 sequences during
        SD pretraining; --lgdp_prompt_length 77 tests whether that mismatch matters.
        Default (7) preserves the original direct-feed behavior exactly."""
        prompt_length = int(getattr(args, "lgdp_prompt_length", 7))
        if prompt_length == 77 and dynamic_prompt.shape[1] < 77:
            B = dynamic_prompt.shape[0]
            pad_tail = self.null_prompt_embeds[:, dynamic_prompt.shape[1]:, :].expand(B, -1, -1)
            dynamic_prompt = torch.cat([dynamic_prompt, pad_tail.to(dynamic_prompt.dtype)], dim=1)
        return dynamic_prompt

    def _build_lgdp_text_prompt(self, intensities_pred, mode):
        """Build one literal natural-language sentence per batch sample from the router's
        [B, K] intensities -- --lgdp_injection_mode text_ordinal / text_numeric ablation.
        Unlike _compose_lgdp_prompt (a differentiable weighted blend of learned direction
        vectors), this verbalizes the intensity into words/numbers via python string
        formatting, which is inherently non-differentiable -- the SR loss's gradient to
        the router is intentionally cut here (.detach() below); the router still trains
        normally through Phase-1 pretrain / CF / dose-loss, which never call this method.

        mode="text_ordinal": bucket each of the K values into slight/middle/strong and
            join as "restore {bucket} {phrase}, {bucket} {phrase}, ...".
        mode="text_numeric": join as "{phrase} level {value:.2f}, {phrase} level {value:.2f}, ...".

        Returns a list of B strings, one per batch sample, in self.deg_names order.
        """
        phrase_style = getattr(self, "lgdp_text_phrase_style", "degradation_noun")
        if phrase_style == "restoration_verb":
            phrases = RESTORATION_VERB_PHRASES_FULL6 if self.lgdp_token_scheme == "full6" \
                else RESTORATION_VERB_PHRASES_MERGED3
        elif phrase_style == "degradation_noun":
            phrases = NOUN_PHRASES_FULL6 if self.lgdp_token_scheme == "full6" else NOUN_PHRASES_MERGED3
        else:
            raise ValueError(f"Unknown --lgdp_text_phrase_style {phrase_style!r} "
                              "(expected 'degradation_noun' or 'restoration_verb')")
        values = intensities_pred.detach().float().cpu()  # [B, K]
        texts = []
        for b in range(values.shape[0]):
            if mode == "text_ordinal":
                clauses = []
                for k, phrase in enumerate(phrases):
                    v = values[b, k].item()
                    bucket = "slight" if v < 1.0 / 3.0 else ("middle" if v < 2.0 / 3.0 else "strong")
                    clauses.append(f"{bucket} {phrase}")
                texts.append("restore " + ", ".join(clauses))
            elif mode == "text_numeric":
                clauses = [f"{phrase} level {values[b, k].item():.2f}" for k, phrase in enumerate(phrases)]
                texts.append(", ".join(clauses))
            else:
                raise ValueError(f"Unknown text prompt mode {mode!r} (expected 'text_ordinal' or 'text_numeric')")
        return texts

    def _compose_lgdp_prompt_text(self, intensities_pred, args, mode):
        """Alternative to _compose_lgdp_prompt: instead of a learned-vector blend, build a
        real sentence per sample (_build_lgdp_text_prompt) and run it through the frozen
        CLIP text encoder -- this sentence IS the entire UNet prompt (no BOS/token
        splicing, no --lgdp_prompt_length/--domain_prompt, the tokenizer already pads to
        its native 77-token length). Mirrors the null_prompt_embeds / neg_ids pattern
        elsewhere in this file (tokenize -> frozen text_encoder -> [B,77,D], no_grad since
        text_encoder is frozen and the intensity->text step already cut the graph).

        Returns (dynamic_prompt [B,77,D], debug) with the same contract as
        _compose_lgdp_prompt so callers/logging need no changes.
        """
        texts = self._build_lgdp_text_prompt(intensities_pred, mode)
        with torch.no_grad():
            ids = self.tokenizer(
                texts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            dynamic_prompt = self.text_encoder(ids)[0].float()  # [B, 77, D]
        return dynamic_prompt, {"prompt_weights": intensities_pred.float()}

    def _compose_lgdp_prompt_ram(self, x_img):
        """--lgdp_injection_mode ram_tag: real per-image RAM tags of x_img (NOT router
        intensities) become the entire UNet prompt, run through the frozen CLIP text
        encoder (see build_ram_tag_prompt). Same (dynamic_prompt, debug) contract as
        _compose_lgdp_prompt/_compose_lgdp_prompt_text so callers need no other changes;
        prompt_weights has no router-intensity meaning here so it's returned as zeros
        purely to keep self.last_prompt_weights logging from crashing.
        """
        texts = build_ram_tag_prompt(self.ram_model, x_img)
        with torch.no_grad():
            ids = self.tokenizer(
                texts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            dynamic_prompt = self.text_encoder(ids)[0].float()  # [B, 77, D]
        prompt_weights = torch.zeros(x_img.shape[0], self.num_deg, device=x_img.device)
        return dynamic_prompt, {"prompt_weights": prompt_weights}

    def encode_degradation_prediction(self, x_lr):
        """Encode an LR image into degradation intensities using the same visual path as forward.

        Args:
            x_lr: [B, C, H, W] LR image tensor in pixel space (range [-1, 1]).

        Returns:
            intensities: [B, K] predicted degradation intensities in [0, 1].
            None: kept for call-site compatibility (both callers discard this slot);
                  no longer a meaningful shared feature once --router_arch=conv skips
                  the base architecture's pooled-feature stage entirely.
        """
        # mode() (posterior mean), not sample(): this feature path only feeds the router, so
        # any stochastic encoder noise here is pure label/feature noise with no informational
        # content about the actual degradation — sample() would make the SAME image yield a
        # different intensity prediction on every call, which is wrong for something meant to
        # measure degradation magnitude (and would confound any sensitivity/dose-response
        # analysis of the router). Unlike PiSASR.forward()'s SR path, this has no regularization
        # rationale for keeping the noise.
        with torch.amp.autocast('cuda', enabled=False):
            latent = self.vae_fix.encode(x_lr.float()).latent_dist.mode() * self.vae_fix.config.scaling_factor
        lat_fp = latent.to(self._router_param_dtype())
        intensities = self._compute_intensities(lat_fp)  # [B, K]
        return intensities, None

    def forward(self, c_t, c_tgt, batch=None, args=None):
        bs = c_t.shape[0]

        # sample() for encoded_control (SR path, kept as training-time regularization) but
        # mode() for the router's visual feature (latent_fp below) — sharing the sampled
        # draw between both would feed the router VAE-encoder noise with no informational
        # content, on top of a Phase 1 pretraining (compute_router_loss, mode()) / eval
        # (PiSASR_eval.forward, mode()) that are both already deterministic. That mismatch
        # would mean the SAME LR image gets a different intensity prediction / prompt weight
        # on every Phase 2 step, and Phase 2's router input distribution wouldn't match what
        # it was calibrated on in Phase 1 or evaluated with later. One posterior, two draws —
        # no extra VAE encode() call.
        with torch.amp.autocast('cuda', enabled=False):
            posterior = self.vae_fix.encode(c_t.float()).latent_dist
            encoded_src = posterior.sample() * self.vae_fix.config.scaling_factor
        encoded_src = encoded_src.to(dtype=c_t.dtype)
        encoded_control = encoded_src.detach().requires_grad_(True)  # UNet control latent (separate leaf)

        force_val = getattr(args, "force_router_intensity", None)
        if force_val is not None:
            # "No router" ablation: bypass vis_projector/vis_norm/spatial_pool/router
            # entirely (not called at all, no wasted compute/gradient) — every degradation
            # intensity is a fixed constant, not a per-image prediction.
            intensities_pred = torch.full(
                (bs, self.num_deg), float(force_val), device=c_t.device, dtype=torch.float32)
        else:
            # Router: predict per-degradation intensities from the LR visual evidence.
            with torch.amp.autocast('cuda', enabled=False):
                encoded_router = posterior.mode() * self.vae_fix.config.scaling_factor
            latent_fp = encoded_router.to(self._router_param_dtype())
            intensities_pred = self._compute_intensities(latent_fp)  # [B, 6], already in [0, 1]

        # CFG-style dropout: occasionally zero out the degradation weights entirely (trains
        # the unconditional/BOS-only path). Otherwise the 6 degradation tokens, weighted by
        # router intensity, ARE the prompt — the only text condition the UNet ever sees.
        # Gated on self.training: train_pisasr.py's periodic validation snapshot calls this
        # same forward() after unwrapped_model.eval() (under torch.no_grad()) — without this
        # gate, ~null_text_ratio of those "eval" images would silently get the BOS-only path
        # instead of the checkpoint's actual predicted prompt, making validation output
        # non-deterministic and not representative of the checkpoint being evaluated.
        if self.training and random.random() < getattr(args, "null_text_ratio", 0.1):
            intensities_for_prompt = torch.zeros_like(intensities_pred)
        else:
            intensities_for_prompt = intensities_pred
        injection_mode = getattr(args, "lgdp_injection_mode", "direct_feed")
        if injection_mode == "ram_tag":
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_ram(c_t)
        elif injection_mode in ("text_ordinal", "text_numeric"):
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_text(intensities_for_prompt, args, injection_mode)
        else:
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt(intensities_for_prompt, args)

        with torch.no_grad():
            neg_ids = self.tokenizer(
                "depth blur, background haze, signal attenuation, speckle noise, "
                "electronic noise, compression artifact, low quality ultrasound",
                max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            # Truncated to match dynamic_prompt's actual sequence length (needed for CSD's
            # torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) to work) — length
            # varies with lgdp_prompt_length (7 vs 77) AND whether --domain_prompt is set.
            _prompt_len = dynamic_prompt.shape[1]
            neg_prompt_embeds = self.text_encoder(neg_ids)[0][:, :_prompt_len, :].expand(bs, -1, -1)

        self._activate_pix_sem_adapters()
        model_pred = self.unet(
            encoded_control, self.timesteps1,
            encoder_hidden_states=dynamic_prompt, cross_attention_kwargs=None,
        ).sample

        self.last_prompt_weights = lgdp_debug["prompt_weights"][0].detach().cpu()
        self.last_intensities    = intensities_pred[0].detach().cpu()

        x_denoised = encoded_control - model_pred

        with torch.amp.autocast('cuda', enabled=False):
            output_image = (self.vae_fix.decode(x_denoised.float() / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image.to(dtype=c_t.dtype), x_denoised, dynamic_prompt, neg_prompt_embeds, intensities_pred

    def forward_curriculum(self, c_tgt, args):
        """Curriculum-driven main SR forward (2026-07-16 v2-Mix recipe — see memory
        pisasr-dapmr-phase0-audit). Unlike forward() (which conditions on a real paired LR
        image with unknown/unlabeled degradation composition), this SYNTHESIZES the LR
        input from c_tgt — every training step has an exact, known multi-degradation
        label, and the router is not needed to produce a training signal at all during the
        early (beta=1) portion of training.

        --online_degrade_source selects the synthesis sampler:
          "curriculum" (default) - _synthesize_mixed_curriculum_batch, the independently
              re-implemented degradation port (fast, but a from-scratch run on this alone
              caused a real Cardiac/BUSI regression — likely a train/eval degradation-
              distribution mismatch vs the generator that actually produced the static
              training/eval LR files).
          "esrgan_physic" - _synthesize_online_esrgan_batch, an exact online port of THAT
              generator (see src/datasets/esrgan_physic_online.py) — same distribution the
              static data and eval sets come from, freshly re-sampled every step.

        This directly gives the LoRA weights (not just the degradation-prompt tokens)
        supervised exposure to SIMULTANEOUS multi-slot corrections with a KNOWN-correct
        prompt — the finding that motivated this method: a narrower fix that froze LoRA and
        only retuned the degradation-prompt tokens on the same curriculum completely failed
        to move the mixed-degradation interaction/oracle-gap numbers (see the "Stage 1
        controller" experiment in that memory) — the correction-PERFORMING weights
        themselves, not just the prompt directions, need this signal.

        beta (via _teacher_forcing_alpha, driven by --teacher_forcing_anneal_frac) blends
        the prompt from the ground-truth label (beta=1, early training) toward the router's
        own prediction (beta=0, late training). router_pred is computed (WITH gradient,
        never detached) whenever beta<1 OR the caller wants it for an explicit dose loss
        (self.last_router_pred / self.last_labels_syn, mirroring the existing
        self.last_intensities/self.last_prompt_weights convention) — whether the router's
        own gradient actually accumulates is controlled purely by router_params.requires_grad
        at call time (Phase 1 pretrain + --router_finetune_steps' unfreeze-at-the-end
        schedule gate this exactly as forward()'s own router usage does).

        Returns the same 5-tuple as forward(): (output_image, x_denoised, dynamic_prompt,
        neg_prompt_embeds, intensities) so the training loop's L2/CSD/logging code doesn't
        need to change beyond swapping which method it calls.
        """
        source = getattr(args, "online_degrade_source", "curriculum")
        if source == "esrgan_physic":
            degraded_batch, labels_syn = self._synthesize_online_esrgan_batch(c_tgt, args)
        else:
            degraded_batch, labels_syn = self._synthesize_mixed_curriculum_batch(c_tgt, args)
        bs = degraded_batch.shape[0]

        with torch.amp.autocast('cuda', enabled=False):
            posterior = self.vae_fix.encode(degraded_batch.float()).latent_dist
            encoded_src = posterior.sample() * self.vae_fix.config.scaling_factor
        encoded_src = encoded_src.to(dtype=degraded_batch.dtype)
        encoded_control = encoded_src.detach().requires_grad_(True)

        force_val = getattr(args, "force_router_intensity", None)
        if force_val is not None:
            # "No router" ablation, mirrors forward()'s own force_router_intensity bypass
            # (pisasr.py's other forward()): every degradation intensity is a fixed
            # constant regardless of image content, router never called, no gradient.
            intensities = torch.full(
                (bs, self.num_deg), float(force_val), device=degraded_batch.device, dtype=torch.float32)
            router_pred = None
        else:
            beta = self._teacher_forcing_alpha(args)
            router_pred = None
            if beta < 1.0 or getattr(args, "lambda_router_dose_online", 0.0) > 0:
                with torch.amp.autocast('cuda', enabled=False):
                    encoded_router = posterior.mode() * self.vae_fix.config.scaling_factor
                latent_fp = encoded_router.to(self._router_param_dtype())
                router_pred = self._compute_intensities(latent_fp)
            if beta < 1.0:
                intensities = beta * labels_syn + (1.0 - beta) * router_pred
            else:
                intensities = labels_syn

        injection_mode = getattr(args, "lgdp_injection_mode", "direct_feed")
        if injection_mode == "ram_tag":
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_ram(degraded_batch)
        elif injection_mode in ("text_ordinal", "text_numeric"):
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_text(intensities, args, injection_mode)
        else:
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt(intensities, args)

        with torch.no_grad():
            neg_ids = self.tokenizer(
                "depth blur, background haze, signal attenuation, speckle noise, "
                "electronic noise, compression artifact, low quality ultrasound",
                max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            _prompt_len = dynamic_prompt.shape[1]
            neg_prompt_embeds = self.text_encoder(neg_ids)[0][:, :_prompt_len, :].expand(bs, -1, -1)

        self._activate_pix_sem_adapters()
        model_pred = self.unet(
            encoded_control, self.timesteps1,
            encoder_hidden_states=dynamic_prompt, cross_attention_kwargs=None,
        ).sample

        self.last_prompt_weights = lgdp_debug["prompt_weights"][0].detach().cpu()
        self.last_intensities    = intensities[0].detach().cpu()
        self.last_labels_syn     = labels_syn
        self.last_router_pred    = router_pred

        x_denoised = encoded_control - model_pred
        with torch.amp.autocast('cuda', enabled=False):
            output_image = (self.vae_fix.decode(x_denoised.float() / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image.to(dtype=c_tgt.dtype), x_denoised, dynamic_prompt, neg_prompt_embeds, intensities

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"] =\
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"] =\
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem
        sd["lora_rank_unet_pix"] = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}

        # None in --lgdp_variable_tokens mode (no single pooled token to store).
        sd["text_prototypes"] = self.text_prototypes.data.cpu() if self.text_prototypes is not None else None

        # router_arch is saved so load_ckpt_from_state_dict (resume) and PiSASR_eval (inference)
        # know which of the two key layouts below is present. "base" keeps the original
        # separate-key layout untouched (any external tooling that reads sd["router"] etc.
        # directly keeps working); "conv" adds its own single key instead.
        sd["router_arch"] = self.router_arch
        if self.router_arch == "conv":
            sd["conv_router"] = self.conv_router.state_dict()
        else:
            sd["router"] = self.router.state_dict()
            sd["vis_projector"] = self.vis_projector.state_dict()
            sd["vis_norm"] = self.vis_norm.state_dict()
            sd["spatial_pool"] = self.spatial_pool.state_dict()

        # LGDP learnable restoration tokens. token_segment_lengths is None unless
        # --lgdp_variable_tokens (then it's the list of per-degradation token counts needed
        # to know how to broadcast router intensities across restoration_tokens on load).
        sd["restoration_tokens"]       = self.restoration_tokens.detach().cpu()
        sd["restoration_token_scale"]  = self.restoration_token_scale.detach().cpu()
        sd["prompt_bank_mode"]         = getattr(self.args, "prompt_bank_mode", "target_state")
        sd["lgdp_variable_tokens"]     = self.lgdp_variable_tokens
        sd["token_segment_lengths"]    = self.token_segment_lengths
        sd["lgdp_token_scheme"]        = self.lgdp_token_scheme

        # Learnable domain prompt (0-length content when --domain_prompt is unset/empty).
        sd["domain_prompt_content"] = self.domain_prompt_content.detach().cpu()
        sd["domain_prompt_eos"]     = self.domain_prompt_eos.detach().cpu()
        sd["domain_prompt_text"]    = getattr(self.args, "domain_prompt", "")
        torch.save(sd, outf)

    def _tensor_to_np01(self, img):
        """[1,C,H,W] tensor [-1,1] → float32 numpy [H,W] (grayscale mean of channels)."""
        import numpy as _np
        np_img = img[0].float().cpu().numpy()          # [C, H, W]
        return _np.clip(np_img.mean(0) * 0.5 + 0.5, 0.0, 1.0).astype(_np.float32)

    def _np01_to_tensor(self, arr, ref_img):
        """float32 numpy [H,W] → [1,C,H,W] tensor [-1,1] matching ref_img device/dtype."""
        t = torch.from_numpy(arr * 2.0 - 1.0).to(ref_img.device, dtype=ref_img.dtype)
        return t.unsqueeze(0).unsqueeze(0).expand_as(ref_img)

    def _new_label_vec(self):
        """A fresh all-zero per-sample label list of length self.num_deg (6 for
        --lgdp_token_scheme full6, 3 for merged3). Pair with _set_label to fill it in from
        physical deg_type-indexed severities without every synthesis function needing to
        know the active token scheme."""
        return [0.0] * self.num_deg

    def _set_label(self, lbl, deg_type, value):
        """In-place write of one physical degradation's severity into its token-scheme slot.
        lbl: list of length self.num_deg (from _new_label_vec), modified in place.
        deg_type: 0-5, index into DEGRADATION_NAMES / _apply_single_degradation.
        value: severity in [0, 1] for that physical degradation.

        Under full6, self.deg_group_map is the identity so this is a plain lbl[deg_type] =
        value. Under merged3, several deg_types share one group slot (e.g. haze and
        attenuation both feed the "photometric" slot) — MAX rather than sum/overwrite, so a
        slot's magnitude reflects whichever underlying degradation is currently most severe,
        not an inflated double-count when two grouped degradations are simultaneously active.
        """
        slot = self.deg_group_map[deg_type]
        if value > lbl[slot]:
            lbl[slot] = value

    def _fold_label6(self, label6):
        """Fold a full 6-dim physical-severity vector (e.g. from an external generator like
        generate_labeled_lr, where multiple physical degradations can be simultaneously
        nonzero) into a length-self.num_deg label via _set_label's max-pool-per-group rule."""
        lbl = self._new_label_vec()
        for j in range(6):
            self._set_label(lbl, j, float(label6[j]))
        return lbl

    # Per-type max physical parameter at severity=1.0 (severity=0.0 always maps to the
    # exact identity value: sigma=0, strength=0, atten_ratio=1.0, add_sigma=0, scale=1.0).
    _SEVERITY_MAX = {
        0: {"sigma": 2.0},                                   # psf_blur
        1: {"haze_strength": 0.12},                          # haze
        2: {"min_atten_ratio": 0.40},                        # attenuation (ratio: 1.0 -> 0.40)
        3: {"strength": 0.35},                                # speckle
        4: {"add_sigma": 0.06},                               # additive_noise
        5: {"max_scale": 4.0},                                # compression (scale: 1.0 -> 4.0)
    }

    def _sample_aux_params(self, deg_type):
        """Sample the non-severity randoms for ONE degradation type (noise seed, spatial
        correlation lengths, Nakagami shape) — everything EXCEPT the severity-controlled
        magnitude. Kept separate from severity so the SAME realization (aux) can be
        replayed at a DIFFERENT severity (see _severity_to_params /
        _synthesize_controllable_triplet's continuous-user-control target)."""
        if deg_type in (0, 2, 5):
            return {}
        if deg_type == 1:
            return {"haze_sigma": random.uniform(5.0, 8.0),
                    "seed": random.randint(0, 2 ** 31 - 1)}
        if deg_type == 3:
            return {"m": random.uniform(0.4, 0.9),
                    "sigma_ax": random.uniform(0.4, 0.7), "sigma_lat": random.uniform(1.2, 2.2),
                    "seed": random.randint(0, 2 ** 31 - 1)}
        if deg_type == 4:
            return {"sigma_ax": random.uniform(0.4, 0.7), "sigma_lat": random.uniform(1.2, 2.2),
                    "seed": random.randint(0, 2 ** 31 - 1)}
        raise ValueError(f"Unknown deg_type {deg_type}")

    def _severity_to_params(self, deg_type, severity, aux):
        """Pure deterministic map severity in [0,1] -> full params dict (merges in aux).
        Guarantees severity=0.0 -> the exact identity value for every type (D_k(x,0)=x),
        by construction (not by an ad-hoc reverse-normalized label afterwards, which is
        what let attenuation/speckle/compression have a non-identity "zero" before this
        refactor: attenuation's old floor was ratio=0.80 not 1.0, speckle's was
        strength=0.12 not 0, compression's was scale=1.5 not 1.0). `severity` IS the label
        (lbl_delta[deg_type] = severity), no reverse-normalization needed downstream.
        """
        severity = float(max(0.0, min(1.0, severity)))
        mx = self._SEVERITY_MAX[deg_type]
        p = dict(aux)
        p["severity"] = severity
        if deg_type == 0:
            p["sigma"] = severity * mx["sigma"]
        elif deg_type == 1:
            p["haze_strength"] = severity * mx["haze_strength"]
        elif deg_type == 2:
            p["atten_ratio"] = 1.0 - severity * (1.0 - mx["min_atten_ratio"])
        elif deg_type == 3:
            p["strength"] = severity * mx["strength"]
        elif deg_type == 4:
            p["add_sigma"] = severity * mx["add_sigma"]
        elif deg_type == 5:
            p["scale"] = 1.0 + severity * (mx["max_scale"] - 1.0)
        else:
            raise ValueError(f"Unknown deg_type {deg_type}")
        return p

    def _sample_degradation_params(self, deg_type, severity=None):
        """Sample the random params for ONE degradation type, as a dict. Pulled out of
        _apply_single_degradation so the exact same realization can be REPLAYED on a
        different base image or at a different severity — needed to build
        controllable-restoration targets where the "kept" degradations must be identical
        between the input and the (possibly partially-restored) target (see
        _synthesize_controllable_triplet).

        severity=None (default, backward-compatible): sample severity ~ U(0,1) internally.
        This reproduces the exact old marginal label distribution for blur/haze/additive_noise
        (already U(0,1) end to end) and WIDENS attenuation/speckle/compression's old ranges
        to finally include true severity=0 identity (see _SEVERITY_MAX / _severity_to_params
        docstring) — those three previously could never reach identity even at "zero".
        """
        aux = self._sample_aux_params(deg_type)
        if severity is None:
            severity = random.uniform(0.0, 1.0)
        return self._severity_to_params(deg_type, severity, aux)

    def _apply_single_degradation(self, img, deg_type, H, W, params=None):
        """Apply one degradation type to img [1,C,H,W] in [-1,1]. Returns (img_deg, lbl_delta).

        params: optional dict from _sample_degradation_params(deg_type). If None (default,
        backward-compatible), fresh params are sampled internally — identical behavior to
        before this arg existed. If provided, the degradation is applied deterministically
        with those params so the SAME realization can be replayed on another base image.

        Degradation types — ONE physics function per label dimension (1:1), so every type can
        be exercised in isolation without confounding another. Calibrated to match the ACTUAL
        generator that produced the training data, `Real_Ultrasound/generate_lr_esrgan_physic.py`
        (ESRGAN backbone + a thin US-physics layer): that generator has NO acoustic-shadow-column
        degradation at all — its 3 measured-real US-physics signatures are depth attenuation
        (linear ramp, not exponential decay), anisotropic native-grid Nakagami speckle (not an
        FFT approximation), and anechoic haze/clutter (previously entirely missing from this
        6-dim scheme). Compression there is ESRGAN-style down/up-resize with NO JPEG
        (`drop_jpeg=True` — real LR has zero JPEG blockiness), so JPEG is dropped here too.
            0 = psf_blur       – simple Gaussian blur (approximates the ESRGAN kernel-blur
                                 stage; not an exact kernel-sampling match, see backbone in the
                                 real generator for the full RealESRGAN blur/resize/noise model)
            1 = haze           – anechoic clutter: smooth random field raises the brightness
                                 floor in dark regions (real generator's `us_physics` step 3)
            2 = attenuation    – linear top->bottom brightness ramp, Gaussian-smoothed (real
                                 generator's `us_physics` step 1; NOT the old exponential decay)
            3 = speckle        – anisotropic multiplicative Nakagami speckle at native grid,
                                 lateral > axial correlation (real generator's `us_physics`
                                 step 2; NOT the old FFT-based approximation)
            4 = additive_noise – anisotropically-correlated (NOT i.i.d. white) Gaussian noise,
                                 same spatial correlation as speckle so it doesn't whiten the ACF
            5 = compression    – bicubic down × scale∈[1.5,4.0] + upsample, NO JPEG

        Label vector indices (6-dim, same order):
            [0]=psf_blur [1]=haze [2]=attenuation [3]=speckle [4]=additive_noise [5]=compression

        Zero-severity identity guarantee: lbl_delta[deg_type] = params["severity"] directly
        (no reverse-normalization from the physical parameter — see _severity_to_params).
        severity=0.0 short-circuits to returning img UNCHANGED for every type, guaranteeing
        D_k(x, 0) = x exactly (not just approximately, and not subject to each branch's own
        floor-value quirks — e.g. old attenuation/speckle/compression could never actually
        reach identity even at their nominal "zero" label).
        """
        lbl_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if params is None:
            params = self._sample_degradation_params(deg_type)

        severity = params.get("severity")
        if severity is not None:
            lbl_delta[deg_type] = severity
            if severity == 0.0:
                return img, lbl_delta

        if deg_type == 0:  # Simple uniform Gaussian PSF (approximates ESRGAN kernel-blur stage)
            import numpy as _np
            from scipy.ndimage import gaussian_filter as _gf
            arr    = self._tensor_to_np01(img)
            sigma  = params["sigma"]
            result = _gf(arr.astype(_np.float32), sigma=sigma)
            img = self._np01_to_tensor(_np.clip(result, 0.0, 1.0).astype(_np.float32), img)

        elif deg_type == 1:  # Anechoic haze / clutter — matches generate_lr_esrgan_physic.py's
            # us_physics step 3: a smooth random field raises the brightness floor in dark
            # (anechoic) regions. Applied in [0,1] space (multiplicative/additive on envelope).
            import numpy as _np
            from scipy.ndimage import gaussian_filter as _gf
            arr = self._tensor_to_np01(img)  # [H,W] grayscale [0,1]
            haze_strength = params["haze_strength"]
            haze_sigma    = params["haze_sigma"]
            haze_ref      = 0.45
            _rng = _np.random.default_rng(params["seed"])
            field = _gf(_rng.standard_normal((H, W)).astype(_np.float32), sigma=(haze_sigma, haze_sigma))
            field = (field - field.min()) / (_np.ptp(field) + 1e-9)
            darkness = _np.clip(1.0 - arr / haze_ref, 0.0, 1.0)
            result = _np.clip(arr + haze_strength * field * darkness, 0.0, 1.0).astype(_np.float32)
            img = self._np01_to_tensor(result, img)

        elif deg_type == 2:  # Depth attenuation ONLY — matches generate_lr_esrgan_physic.py's
            # us_physics step 1: linear top->bottom brightness ramp, Gaussian-smoothed (NOT
            # the old exponential decay).
            import numpy as _np
            from scipy.ndimage import gaussian_filter as _gf
            arr = self._tensor_to_np01(img)
            atten_ratio = params["atten_ratio"]  # bottom/top brightness ratio
            ramp = _np.linspace(1.0, atten_ratio, H, dtype=_np.float32)[:, None] * _np.ones((1, W), dtype=_np.float32)
            ramp = _gf(ramp, sigma=(H * 0.04, 0))
            result = _np.clip(arr * ramp, 0.0, 1.0).astype(_np.float32)
            img = self._np01_to_tensor(result, img)

        elif deg_type == 3:  # Anisotropic multiplicative Nakagami speckle at native grid —
            # matches generate_lr_esrgan_physic.py's us_physics step 2 (NOT the old FFT-based
            # approximation). Lateral correlation > axial (anisotropic US speckle).
            import numpy as _np
            from scipy.ndimage import gaussian_filter as _gf
            arr = self._tensor_to_np01(img)
            m          = params["m"]          # Nakagami shape
            strength   = params["strength"]   # multiplicative contrast
            sigma_ax   = params["sigma_ax"]   # axial correlation (px)
            sigma_lat  = params["sigma_lat"]  # lateral correlation (px) > axial
            rng = _np.random.default_rng(params["seed"])
            g   = rng.gamma(shape=m, scale=1.0 / m, size=arr.shape)
            amp = _np.sqrt(g)
            amp = _gf(amp, sigma=(sigma_ax, sigma_lat))
            amp = amp / amp.mean()
            result = _np.clip(arr * (1.0 + (amp - 1.0) * strength), 0.0, 1.0).astype(_np.float32)
            img = self._np01_to_tensor(result, img)

        elif deg_type == 4:  # Anisotropically-correlated additive thermal/electronic noise —
            # matches generate_lr_esrgan_physic.py's us_physics additive-noise term: uses the
            # SAME spatial correlation as speckle so it does NOT whiten the ACF (pure i.i.d.
            # Gaussian would make the grain look artificially fine).
            import numpy as _np
            from scipy.ndimage import gaussian_filter as _gf
            arr = self._tensor_to_np01(img)
            add_sigma = params["add_sigma"]
            sigma_ax  = params["sigma_ax"]
            sigma_lat = params["sigma_lat"]
            fg = arr > (8.0 / 255.0)
            _rng = _np.random.default_rng(params["seed"])
            nz = _gf(_rng.standard_normal((H, W)).astype(_np.float32), sigma=(sigma_ax, sigma_lat))
            nz = nz / (nz.std() + 1e-9)
            result = arr.copy()
            result[fg] = _np.clip(arr[fg] + nz[fg] * add_sigma, 0.0, 1.0)
            img = self._np01_to_tensor(result.astype(_np.float32), img)

        elif deg_type == 5:  # Bicubic down × scale∈[1.5,4.0] + upsample — matches ESRGAN
            # backbone's net resolution loss. NO JPEG (real generator sets drop_jpeg=True;
            # real LR has ~zero JPEG blockiness, so injecting it here would mismatch the data).
            scale   = params["scale"]
            h_small = max(1, int(round(H / scale)))
            w_small = max(1, int(round(W / scale)))
            img_s   = F.interpolate(img.float(), size=(h_small, w_small),
                                    mode='bicubic', align_corners=False).clamp(-1.0, 1.0)
            img = F.interpolate(img_s.to(img.dtype), size=(H, W),
                                mode='bicubic', align_corners=False).clamp(-1.0, 1.0)

        return img, lbl_delta

    def _synthesize_degraded_batch(self, c_tgt, single_degradation_prob=0.5):
        """c_tgt: HR clean images [B,C,H,W] in [-1,1] → degrade to approximate
        Real_Ultrasound/generate_lr_esrgan_physic.py's ESRGAN-backbone + US-physics pipeline
        (see _apply_single_degradation for the per-function physics/parameter calibration).

        Per-image, two mutually-exclusive modes:
          - ISOLATED (prob=single_degradation_prob, default 50%): apply exactly ONE of the 6
            physical degradation functions (0=psf_blur, 1=haze, 2=attenuation, 3=speckle,
            4=additive_noise, 5=compression — each its own independent function, see
            _apply_single_degradation). This lets EACH restoration token get a training signal
            where it is the ONLY active degradation (including compression alone, or pure
            haze with zero attenuation, etc.) — needed to individually calibrate each token's
            magnitude response instead of always confounding it with whatever else is combined
            in (the old scheme's shadowing+attenuation were inseparable — a single combined
            function always applied attenuation and only optionally shadowing; the real data
            generator has no shadowing degradation at all, only haze/attenuation/speckle).
          - COMBINED (the rest): sample k ∈ {1..4} types from {0,1,2,3,4} (psf_blur, haze,
            attenuation, speckle, additive_noise), apply in random order, then always apply
            compression (5) last.

        Shared by compute_router_loss (router supervision) and compute_synthetic_sr_loss
        (UNet degradation-magnitude-conditioning supervision) — both need the same
        (degraded LR, KNOWN 6-dim label) pairs, just used for different losses.

        Returns: (degraded_batch [B,C,H,W] in [-1,1], labels_syn [B,num_deg] in [0,1]).
        Labels: 6-dim [psf_blur, haze, attenuation, speckle, additive_noise, compression]
        under --lgdp_token_scheme full6 (default), or the merged3 group-slot layout (see
        _set_label / DEG_TYPE_TO_GROUP_MERGED3) otherwise.
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device

        degraded_images = []
        labels = []

        for i in range(B):
            img = c_tgt[i:i+1]  # [1, C, H, W]
            lbl = self._new_label_vec()

            if random.random() < single_degradation_prob:
                # Isolated: exactly one degradation type, alone (may be compression alone,
                # or pure haze/attenuation alone).
                deg_type = random.randint(0, 5)
                img, lbl_delta = self._apply_single_degradation(img, deg_type, H, W)
                self._set_label(lbl, deg_type, lbl_delta[deg_type])
            else:
                # Combined: sample k ∈ {1..4} from {0,1,2,3,4} (psf_blur, haze,
                # attenuation, speckle, additive_noise).
                num_degs = random.randint(1, 4)
                for deg_type in random.sample(range(5), k=num_degs):
                    img, lbl_delta = self._apply_single_degradation(img, deg_type, H, W)
                    self._set_label(lbl, deg_type, lbl_delta[deg_type])

                # Compression always applied last
                img, lbl_delta = self._apply_single_degradation(img, 5, H, W)
                self._set_label(lbl, 5, lbl_delta[5])

            degraded_images.append(torch.clamp(img, -1.0, 1.0))
            labels.append(lbl)

        degraded_batch = torch.cat(degraded_images, dim=0)
        labels_syn     = torch.tensor(labels, dtype=torch.float32, device=device)  # [B, num_deg]
        return degraded_batch, labels_syn

    def compute_router_loss(self, c_tgt, c_src=None):
        """Self-supervised router regression against synthetic degradation labels.

        --use_mixed_curriculum (2026-07-16 v2-Mix recipe) swaps the sampler from
        _synthesize_degraded_batch (uniform random 1-4-of-5 + forced compression) to
        whichever sampler --online_degrade_source selects — _synthesize_mixed_curriculum_batch
        (independently re-implemented port) or _synthesize_online_esrgan_batch (exact online
        port of the generator that actually produced the static training/eval LR files) —
        the SAME dispatch forward_curriculum uses, so the router is calibrated on the SAME
        distribution it will actually see conditioning requests from later.
        """
        if not getattr(self.args, 'use_degradation', True):
            return torch.tensor(0.0, device=c_tgt.device, requires_grad=True)

        if getattr(self.args, "use_mixed_curriculum", False):
            if getattr(self.args, "online_degrade_source", "curriculum") == "esrgan_physic":
                degraded_batch, labels_syn = self._synthesize_online_esrgan_batch(c_tgt, self.args)
            else:
                degraded_batch, labels_syn = self._synthesize_mixed_curriculum_batch(c_tgt, self.args)
        else:
            degraded_batch, labels_syn = self._synthesize_degraded_batch(
                c_tgt, single_degradation_prob=getattr(self.args, "single_degradation_prob", 0.5))

        # mode(), not sample(): this encoding is purely the router's INPUT feature for a
        # supervised regression against the known synthetic label — sample() injects encoder
        # noise into that input->label mapping with no informational content, adding avoidable
        # variance to the Phase 1 pretraining signal. Already under no_grad (vae_fix is frozen),
        # so this only affects what pseudo-label features the router sees, not gradient flow.
        with torch.no_grad():
            latent = self.vae_fix.encode(degraded_batch).latent_dist.mode() * self.vae_fix.config.scaling_factor

        lat_fp = latent.to(self._router_param_dtype())
        intensities_pred = self._compute_intensities(lat_fp)  # [B, 6], already in [0, 1]

        loss = F.smooth_l1_loss(intensities_pred, labels_syn)
        lambda_rank = float(getattr(self.args, "lambda_rank", 0.0))
        if lambda_rank > 0:
            loss = loss + lambda_rank * active_inactive_ranking_loss(
                intensities_pred, labels_syn, margin=float(getattr(self.args, "rank_margin", 0.1)))
        return loss

    def _forward_with_prompt(self, input_batch, intensities, args, sample_encode=True):
        """Shared encode -> compose LGDP prompt -> UNet -> decode pipeline. Factored out of
        compute_synthetic_sr_loss so the specificity-batch/delta-response/dose-response
        losses (spec C.8/10-12) don't each hand-roll the same six lines with their own
        copy-paste risk of drifting from forward()'s actual pipeline.

        intensities: [B,6] prompt weights — NOT necessarily the router's own prediction;
        callers pass ground-truth labels, controllable-request vectors, all-zeros, or a
        teacher-forcing blend with the router's own prediction.
        sample_encode=True uses the VAE posterior SAMPLE (matches forward()'s training-time
        regularization for the main SR path); False uses a deterministic mode() encode.
        """
        with torch.amp.autocast('cuda', enabled=False):
            posterior = self.vae_fix.encode(input_batch.float()).latent_dist
            encoded_src = (posterior.sample() if sample_encode else posterior.mode()) * self.vae_fix.config.scaling_factor
        encoded_src = encoded_src.to(dtype=input_batch.dtype)
        encoded_control = encoded_src.detach().requires_grad_(True)

        injection_mode = getattr(args, "lgdp_injection_mode", "direct_feed")
        if injection_mode == "ram_tag":
            dynamic_prompt, _ = self._compose_lgdp_prompt_ram(input_batch)
        elif injection_mode in ("text_ordinal", "text_numeric"):
            dynamic_prompt, _ = self._compose_lgdp_prompt_text(intensities, args, injection_mode)
        else:
            dynamic_prompt, _ = self._compose_lgdp_prompt(intensities, args)

        self._activate_pix_sem_adapters()
        model_pred = self.unet(
            encoded_control, self.timesteps1,
            encoder_hidden_states=dynamic_prompt, cross_attention_kwargs=None,
        ).sample

        x_denoised = encoded_control - model_pred
        with torch.amp.autocast('cuda', enabled=False):
            output_image = (self.vae_fix.decode(x_denoised.float() / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image.to(dtype=input_batch.dtype)

    def _teacher_forcing_alpha(self, args):
        """Spec C.9 (teacher forcing before router): alpha=1.0 -> the synthetic SR loss's
        prompt uses the GROUND-TRUTH degradation label exclusively; alpha=0.0 -> it uses the
        router's OWN prediction on that same synthesized image exclusively. Linearly annealed
        1.0 -> 0.0 over the LAST --teacher_forcing_anneal_frac fraction of training (mirrors
        dataset.py's domain_rand_anneal_frac schedule). Rationale (spec C.9): mixing router
        error with prompt-specificity error from the start confounds the two; calibrate the
        prompt pathway against clean labels first, then hand off to the router gradually.
        anneal_frac=0.0 (default) disables annealing -> alpha stays 1.0 for the whole run,
        i.e. the original always-GT-label behavior, zero effect on existing runs.
        """
        override = getattr(args, "teacher_forcing_alpha_override", None)
        if override is not None:
            return float(override)
        anneal_frac = float(getattr(args, "teacher_forcing_anneal_frac", 0.0))
        if anneal_frac <= 0.0:
            return 1.0
        step = float(getattr(args, "_current_step", 0))
        total = float(getattr(args, "max_train_steps", 1))
        anneal_start = total * (1.0 - anneal_frac)
        if step <= anneal_start:
            return 1.0
        frac_into_anneal = (step - anneal_start) / max(1.0, total - anneal_start)
        return max(0.0, 1.0 - frac_into_anneal)

    def compute_synthetic_sr_loss(self, c_tgt, args):
        """Degradation-magnitude-supervised SR training: synthesize a degraded LR view of
        c_tgt with a KNOWN 6-dim label, then run the SAME SR pipeline as forward() but
        weight the degradation prompt by the GROUND-TRUTH label instead of the router's own
        (possibly noisy) prediction. This gives the reconstruction loss a direct, correctly-
        labeled magnitude -> output gradient that real-data training (where the prompt
        weight is always the router's own guess) cannot provide on its own.

        Teacher-forcing anneal (spec C.9, see _teacher_forcing_alpha): once
        --teacher_forcing_anneal_frac > 0, the prompt is gradually blended toward the
        router's OWN (detached) prediction on this synthesized image over the last fraction
        of training, so the LoRA backbone isn't trained exclusively against a signal
        (ground truth) it will never see again once the router takes over full time.

        Returns: SR output image [B,C,H,W] in [-1,1] — caller computes the reconstruction
        loss against c_tgt (the same HR the degradation was synthesized from).
        """
        degraded_batch, labels_syn = self._synthesize_degraded_batch(
            c_tgt, single_degradation_prob=getattr(args, "single_degradation_prob", 0.5))

        alpha = self._teacher_forcing_alpha(args)
        if alpha < 1.0:
            with torch.no_grad():
                router_pred, _ = self.encode_degradation_prediction(degraded_batch)
            intensities_for_prompt = alpha * labels_syn + (1.0 - alpha) * router_pred.detach()
        else:
            intensities_for_prompt = labels_syn

        return self._forward_with_prompt(degraded_batch, intensities_for_prompt, args, sample_encode=True)

    # Canonical order in which degradations are composited onto the clean HR (compression
    # always last, as in _synthesize_degraded_batch). Both the input and the controllable
    # target apply their degradations in THIS order so the "kept" ones are consistent.
    _CONTROLLABLE_ORDER = [0, 1, 2, 3, 4, 5]  # blur, haze, atten, speckle, add_noise, compress

    # Canonical evaluation points for the requested-removal fraction u (spec C.8) — half of
    # _sample_user_multiplier's draws snap here, half draw continuously, so training covers
    # both the exact points used at eval time and the dose-response continuum between them.
    _USER_CONTROL_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

    def _sample_user_multiplier(self):
        if random.random() < 0.5:
            return random.choice(self._USER_CONTROL_GRID)
        return random.random()

    def _synthesize_controllable_triplet(self, c_tgt, args):
        """CONTINUOUS controllable-restoration supervision (spec C.8): same input, DIFFERENT
        target depending on how MUCH of each present degradation the prompt requests
        removing. Generalizes the original binary "remove subset R or keep it" scheme
        (u in {0,1} per slot) to a continuous per-slot user multiplier u in [0,1]:
        remaining severity d_remain = (1-u)*d, target = D_{d_remain}(y), prompt r = u*d.
        This is what forces the prompt to be causally used — the input latent alone cannot
        disambiguate which subset (or how much) to remove.

        Per image:
          1. Pick a set S of 1..controllable_max_degradations types. For each d in S sample
             its AUX params ONCE (seed / correlation lengths / Nakagami shape — the
             non-severity randoms, see _sample_aux_params) and its own full-strength
             severity d ~ U(controllable_min_severity, 1.0) (kept above a floor so there is
             always something meaningful to remove).
          2. input = clean HR with ALL of S applied at full severity d (canonical order).
          3. For each d in S, sample a user multiplier u_d (_sample_user_multiplier): 0 =
             "keep this degradation as-is", 1 = "fully remove it", continuous values in
             between = partial removal.
          4. target = clean HR with each d in S applied at severity d*(1-u_d) (canonical
             order, SAME aux/seed as the input's realization — only severity changes; spec
             B.7's "keep the same noise realization between input and target").
          5. prompt intensity vector p[6]: p[d] = u_d * d for d in S (the requested-removal
             magnitude r = u⊙d), 0 for d not in S.

        Returns a dict (all in [-1,1] except severity/u):
          input    [B,C,H,W]
          target   [B,C,H,W]   y_u
          prompt   [B,6]       r = u*d, drives _compose_lgdp_prompt
          severity [B,6]       d, the full input-side magnitude per slot (0 for d not in S)
          u        [B,6]       requested-removal fraction per slot (0 for d not in S)
        Non-commutativity note: applying S in canonical order at reduced severity is an
        approximate ground truth (exact only if degradations commute); the multiplicative
        US-physics ops (haze/atten/speckle/noise) approximately commute, blur/compression
        less so — acceptable as a training signal, flagged for the paper.
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        max_s = int(getattr(args, "controllable_max_degradations", 4))
        min_severity = float(getattr(args, "controllable_min_severity", 0.3))

        inputs, targets, prompts, severities, mults = [], [], [], [], []
        for i in range(B):
            clean = c_tgt[i:i + 1]  # [1,C,H,W]
            k = random.randint(1, max_s)
            S = random.sample(range(6), k=k)
            S_ordered = [d for d in self._CONTROLLABLE_ORDER if d in S]
            aux      = {d: self._sample_aux_params(d) for d in S}
            severity = {d: random.uniform(min_severity, 1.0) for d in S}
            u        = {d: self._sample_user_multiplier() for d in S}

            # input = apply all of S at full severity
            img_in = clean.clone()
            for d in S_ordered:
                params_in = self._severity_to_params(d, severity[d], aux[d])
                img_in, _ = self._apply_single_degradation(img_in, d, H, W, params=params_in)
            img_in = torch.clamp(img_in, -1.0, 1.0)

            # target = apply all of S at reduced severity (1-u)*d, SAME aux/seed
            img_tgt = clean.clone()
            for d in S_ordered:
                remaining = severity[d] * (1.0 - u[d])
                params_tgt = self._severity_to_params(d, remaining, aux[d])
                img_tgt, _ = self._apply_single_degradation(img_tgt, d, H, W, params=params_tgt)
            img_tgt = torch.clamp(img_tgt, -1.0, 1.0)

            p, sev_full, u_full = self._new_label_vec(), self._new_label_vec(), self._new_label_vec()
            for d in S:
                self._set_label(p, d, u[d] * severity[d])
                self._set_label(sev_full, d, severity[d])
                self._set_label(u_full, d, u[d])

            inputs.append(img_in); targets.append(img_tgt)
            prompts.append(p); severities.append(sev_full); mults.append(u_full)

        return {
            "input":    torch.cat(inputs, dim=0),
            "target":   torch.cat(targets, dim=0),
            "prompt":   torch.tensor(prompts, dtype=torch.float32, device=device),
            "severity": torch.tensor(severities, dtype=torch.float32, device=device),
            "u":        torch.tensor(mults, dtype=torch.float32, device=device),
        }

    def _synthesize_specificity_batch(self, c_tgt, args):
        """Cheap single-forward matched/mismatched specificity + continuous
        partial-restoration supervision. Supersedes the old compute_controllable_sr_loss +
        compute_matched_mismatched_loss pair (which needed up to 4 separate UNet forward/
        backward passes per step — 1 for the controllable triplet, up to 3 for match/
        mismatch's own matched/zero/wrong forwards). Here every sample in the batch gets
        exactly ONE degradation instance and ONE (matched-or-mismatched) control/target
        pair, so the whole batch costs exactly ONE forward pass — same as the plain
        reconstruction loss.

        Per sample:
          1. Pick ONE degradation type k, severity d ~ U(controllable_min_severity, 1.0),
             strength u sampled from the discrete grid {0,.25,.5,.75,1} (spec C.8's
             canonical evaluation points). Degrade clean -> x = D_k(y, d).
          2. Matched (prob specificity_matched_prob, default 0.7): control[k] = u*d (the
             correct slot), target = partial restoration D_k(y, (1-u)*d) — SAME aux/seed as
             x's realization (spec B.7).
          3. Mismatched (else): control[wrong j != k] = u*d (a slot that ISN'T present),
             target = x UNCHANGED — requesting removal of something that isn't there must
             not generic-enhance the image. No extra reference forward needed: the
             degraded input itself IS the correct target for "nothing should change".

        Returns (input_batch, control_batch, target_batch), ready for
        self._forward_with_prompt(input_batch, control_batch, args) vs target_batch.
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        min_severity = float(getattr(args, "controllable_min_severity", 0.2))
        matched_prob = float(getattr(args, "specificity_matched_prob", 0.7))

        inputs, controls, targets = [], [], []
        for i in range(B):
            clean = c_tgt[i:i + 1]
            deg_type = random.randint(0, 5)
            severity = random.uniform(min_severity, 1.0)
            strength = random.choice(self._USER_CONTROL_GRID)

            aux = self._sample_aux_params(deg_type)
            params_in = self._severity_to_params(deg_type, severity, aux)
            degraded, _ = self._apply_single_degradation(clean.clone(), deg_type, H, W, params=params_in)
            degraded = torch.clamp(degraded, -1.0, 1.0)

            control = self._new_label_vec()
            if random.random() < matched_prob:
                self._set_label(control, deg_type, strength * severity)
                params_tgt = self._severity_to_params(deg_type, (1.0 - strength) * severity, aux)
                target, _ = self._apply_single_degradation(clean.clone(), deg_type, H, W, params=params_tgt)
                target = torch.clamp(target, -1.0, 1.0)
            else:
                # "Wrong" must land on a DIFFERENT token slot than deg_type's, not just a
                # different physical deg_type — under merged3 several deg_types share one
                # slot (self.deg_group_map), and requesting that same slot (even via a
                # different underlying deg_type) is not a mismatch in prompt-token space.
                wrong_candidates = [j for j in range(6)
                                     if self.deg_group_map[j] != self.deg_group_map[deg_type]]
                wrong_type = random.choice(wrong_candidates)
                self._set_label(control, wrong_type, strength * severity)
                target = degraded

            inputs.append(degraded); controls.append(control); targets.append(target)

        return (torch.cat(inputs, dim=0),
                torch.tensor(controls, dtype=torch.float32, device=device),
                torch.cat(targets, dim=0))

    def compute_specificity_batch_loss(self, c_tgt, args):
        """One-forward matched/mismatched + continuous partial-restoration loss — see
        _synthesize_specificity_batch. Returns (output_image, target_batch); caller applies
        L1 (+ optionally LPIPS), mirroring the other synthetic-loss methods' convention of
        returning raw tensors and letting train_pisasr.py own the actual loss formula.
        """
        input_batch, control_batch, target_batch = self._synthesize_specificity_batch(c_tgt, args)
        output_image = self._forward_with_prompt(input_batch, control_batch, args)
        return output_image, target_batch

    def compute_delta_response_loss(self, c_tgt, args):
        """Delta-response loss (spec C.11): the CHANGE the model makes in response to a
        control vector must match the actual degradation removed, not just push the whole
        output toward "looks nicer" (a global volume knob — the Phase 0 audit's exact
        failure mode: `attenuation` acts as a near-constant broad brightness/contrast
        correction regardless of image content). Compares [F(x,c) - F(x,0)] against
        [y_c^GT - x] on a multi-degradation controllable triplet
        (_synthesize_controllable_triplet). Both branches keep gradient (no stop-grad) — the
        ground-truth delta is an external anchor, so there is no risk of the two live
        predictions collapsing onto each other without matching it.
        """
        triplet = self._synthesize_controllable_triplet(c_tgt, args)
        x, r, y = triplet["input"], triplet["prompt"], triplet["target"]

        pred_c = self._forward_with_prompt(x, r, args)
        pred_0 = self._forward_with_prompt(x, torch.zeros_like(r), args)

        return F.l1_loss(pred_c - pred_0, y - x)

    def compute_dose_response_loss(self, c_tgt, args):
        """Dose-response loss (spec C.12): for two different requested-removal fractions
        u_a < u_b on the SAME input, the model's output difference must track the
        ground-truth target difference — makes low-strength requests (e.g. u=0.25) produce
        a response clearly above the noise floor. The Phase 0 audit's noise-floor finding
        (memory pisasr-dapmr-phase0-audit) found several tokens' low-strength response was
        only ~1.2-2x above repeat-inference noise, barely distinguishable from nothing.
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        max_s = int(getattr(args, "controllable_max_degradations", 4))
        min_severity = float(getattr(args, "controllable_min_severity", 0.3))

        inputs, targets_a, targets_b, prompts_a, prompts_b = [], [], [], [], []
        for i in range(B):
            clean = c_tgt[i:i + 1]
            k = random.randint(1, max_s)
            S = random.sample(range(6), k=k)
            S_ordered = [d for d in self._CONTROLLABLE_ORDER if d in S]
            aux      = {d: self._sample_aux_params(d) for d in S}
            severity = {d: random.uniform(min_severity, 1.0) for d in S}
            u_a = {d: random.uniform(0.0, 0.7) for d in S}
            u_b = {d: random.uniform(u_a[d] + 0.1, 1.0) for d in S}  # strictly above u_a

            img_in = clean.clone()
            for d in S_ordered:
                p_in = self._severity_to_params(d, severity[d], aux[d])
                img_in, _ = self._apply_single_degradation(img_in, d, H, W, params=p_in)
            img_in = torch.clamp(img_in, -1.0, 1.0)

            img_a = clean.clone()
            for d in S_ordered:
                p_a = self._severity_to_params(d, severity[d] * (1.0 - u_a[d]), aux[d])
                img_a, _ = self._apply_single_degradation(img_a, d, H, W, params=p_a)
            img_a = torch.clamp(img_a, -1.0, 1.0)

            img_b = clean.clone()
            for d in S_ordered:
                p_b = self._severity_to_params(d, severity[d] * (1.0 - u_b[d]), aux[d])
                img_b, _ = self._apply_single_degradation(img_b, d, H, W, params=p_b)
            img_b = torch.clamp(img_b, -1.0, 1.0)

            pa, pb = self._new_label_vec(), self._new_label_vec()
            for d in S:
                self._set_label(pa, d, u_a[d] * severity[d])
                self._set_label(pb, d, u_b[d] * severity[d])

            inputs.append(img_in); targets_a.append(img_a); targets_b.append(img_b)
            prompts_a.append(pa); prompts_b.append(pb)

        input_batch = torch.cat(inputs, dim=0)
        target_a = torch.cat(targets_a, dim=0)
        target_b = torch.cat(targets_b, dim=0)
        r_a = torch.tensor(prompts_a, dtype=torch.float32, device=device)
        r_b = torch.tensor(prompts_b, dtype=torch.float32, device=device)

        pred_a = self._forward_with_prompt(input_batch, r_a, args)
        pred_b = self._forward_with_prompt(input_batch, r_b, args)

        return F.l1_loss(pred_b - pred_a, target_b - target_a)

    def _make_two_cf_views(self, img, H, W):
        """Build two counterfactual views from ONE clean HR image with DIFFERENT
        dominant physics degradations. Reuses the physics-grounded
        _apply_single_degradation (blur, haze, attenuation, anisotropic Nakagami speckle,
        correlated sensor noise, compression — 6 independent functions).

        Returns: (view_a, label_a, view_b, label_b) — views in [-1,1], labels [num_deg].
        """
        # Pick two distinct degradation types (0..5): same content, different degradation.
        type_a, type_b = random.sample(range(6), k=2)
        view_a, lbl_delta_a = self._apply_single_degradation(img.clone(), type_a, H, W)
        view_b, lbl_delta_b = self._apply_single_degradation(img.clone(), type_b, H, W)
        view_a = torch.clamp(view_a, -1.0, 1.0)
        view_b = torch.clamp(view_b, -1.0, 1.0)
        lbl_a = self._new_label_vec()
        self._set_label(lbl_a, type_a, lbl_delta_a[type_a])
        lbl_b = self._new_label_vec()
        self._set_label(lbl_b, type_b, lbl_delta_b[type_b])
        return view_a, lbl_a, view_b, lbl_b

    def compute_counterfactual_router_loss(self, c_tgt):
        """Counterfactual degradation supervision (LGDP section 9), generated on-the-fly
        from the HR target so NO dataset/collate changes are needed.

        For each HR image we synthesize two views with different physics degradations and
        known labels. The router is supervised to:
          (1) predict each view's absolute degradation label (smooth_l1), and
          (2) reproduce the label DIFFERENCE between the two same-content views
              (contrast term) — since content is identical, only the degradation
              evidence drives (inten_a - inten_b), which is the counterfactual signal.

        Args:
            c_tgt: [B,C,H,W] clean HR images in [-1,1].
        Returns:
            scalar loss (already weighted-ready; caller multiplies by lambda_cf_router).
        """
        if not getattr(self.args, "use_degradation", True):
            return torch.tensor(0.0, device=c_tgt.device, requires_grad=True)

        B, C, H, W = c_tgt.shape
        views_a, views_b, labels_a, labels_b = [], [], [], []
        for i in range(B):
            va, la, vb, lb = self._make_two_cf_views(c_tgt[i:i + 1], H, W)
            views_a.append(va); views_b.append(vb)
            labels_a.append(la); labels_b.append(lb)

        view_a = torch.cat(views_a, dim=0)
        view_b = torch.cat(views_b, dim=0)
        y_a = torch.tensor(labels_a, dtype=torch.float32, device=c_tgt.device)  # [B,num_deg]
        y_b = torch.tensor(labels_b, dtype=torch.float32, device=c_tgt.device)  # [B,num_deg]

        inten_a, _ = self.encode_degradation_prediction(view_a)
        inten_b, _ = self.encode_degradation_prediction(view_b)

        loss = (F.smooth_l1_loss(inten_a, y_a) + F.smooth_l1_loss(inten_b, y_b))

        lambda_contrast = float(getattr(self.args, "lambda_cf_contrast", 0.5))
        if lambda_contrast > 0:
            loss_contrast = F.smooth_l1_loss(inten_a - inten_b, y_a - y_b)
            loss = loss + lambda_contrast * loss_contrast

        return loss

    # ------------------------------------------------------------------
    # v2-Mix fine-tune curriculum (2026-07-16 follow-up to the mixed-degradation +
    # oracle-router audit): the audit found v2's near-zero router gain on mixed-degradation
    # images (+0.13dB, CI includes 0) traces to two separable problems — (1) severe
    # controller/token interaction (every restoration token loses 1-6.5dB of effectiveness
    # when another degradation is simultaneously present, since v2's specificity-batch loss
    # only ever shows the model ONE active degradation per sample) and (2) a smaller router-
    # calibration gap concentrated in speckle-containing combos. Two-stage fix: Stage 1
    # ("controller") retrains ONLY the degradation-prompt directions/scale on mixed-
    # degradation reconstruction with the prompt fed directly from ground truth (bypassing
    # the router entirely) to fix interaction; Stage 2 ("router") then retrains ONLY the
    # router on the same curriculum with a dose + active/inactive-ranking loss, annealing
    # the reconstruction loss's conditioning from ground truth to the router's own
    # prediction (_teacher_forcing_alpha). See run_finetune_stage in train_pisasr.py.
    # ------------------------------------------------------------------

    # 2-degradation combos that include speckle, and the 1 suggested 3-degradation combo —
    # oversampled (not exclusively used) within the mixed pool so speckle gets much more
    # multi-degradation exposure than a uniform random pairing would give it, without making
    # speckle appear in nearly every mixed sample (which would just recreate a NEW always-on
    # bias, this time for speckle instead of attenuation).
    _SPECKLE_COMBOS_2 = [[2, 3], [1, 3], [0, 3], [4, 3]]  # atten+speckle, haze+speckle, blur+speckle, noise+speckle
    _SPECKLE_COMBO_3 = [2, 0, 3]                            # atten+blur+speckle

    def _synthesize_mixed_curriculum_batch(self, c_tgt, args):
        """Per-sample category curriculum (spec, 2026-07-16 follow-up):
          10% clean or very-light single degradation (severity in [0, 0.2], or literally
              none) — keeps the "zero/near-zero dose" behavior in-distribution WITHOUT a
              separate zero-dose forward branch (just samples of the same mixed batch).
          20% single degradation, moderate-to-high severity — preserves v2's existing
              specificity behavior (the compute_specificity_batch_loss branch already
              covers this in more depth; this fraction keeps the RECONSTRUCTION loss itself
              exposed to isolated cases too, not just mixed ones).
          45% two-degradation mixture (speckle-oversampled at speckle_oversample_prob).
          25% three-degradation mixture (speckle-oversampled at speckle_oversample_prob,
              using the one suggested atten+blur+speckle combo when oversampled).

        Returns: (degraded_batch [B,C,H,W] in [-1,1], labels_syn [B,6] in [0,1]) — same
        contract as _synthesize_degraded_batch, drop-in usable anywhere that expects it.
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        min_sev = float(getattr(args, "controllable_min_severity", 0.2))
        speckle_p = float(getattr(args, "speckle_oversample_prob", 0.45))

        degraded_images, labels = [], []
        for i in range(B):
            img = c_tgt[i:i + 1]
            r = random.random()
            if r < 0.10:                                    # clean / very light
                types = [] if random.random() < 0.5 else [random.randint(0, 5)]
                sev_range = (0.0, 0.2)
            elif r < 0.30:                                   # single, moderate-high
                types = [random.randint(0, 5)]
                sev_range = (min_sev, 1.0)
            elif r < 0.75:                                   # two-mixture
                types = (list(random.choice(self._SPECKLE_COMBOS_2)) if random.random() < speckle_p
                         else random.sample(range(6), k=2))
                sev_range = (min_sev, 1.0)
            else:                                             # three-mixture
                types = (list(self._SPECKLE_COMBO_3) if random.random() < speckle_p
                         else random.sample(range(6), k=3))
                sev_range = (min_sev, 1.0)

            lbl = self._new_label_vec()
            img_out = img.clone()
            for d in [t for t in self._CONTROLLABLE_ORDER if t in types]:
                severity = random.uniform(*sev_range)
                aux = self._sample_aux_params(d)
                params = self._severity_to_params(d, severity, aux)
                img_out, _ = self._apply_single_degradation(img_out, d, H, W, params=params)
                self._set_label(lbl, d, severity)
            img_out = torch.clamp(img_out, -1.0, 1.0)
            degraded_images.append(img_out)
            labels.append(lbl)

        degraded_batch = torch.cat(degraded_images, dim=0)
        labels_syn = torch.tensor(labels, dtype=torch.float32, device=device)
        return degraded_batch, labels_syn

    def _synthesize_online_esrgan_batch(self, c_tgt, args):
        """Online per-step port of the ACTUAL generator that produced the static training/
        eval LR files (generate_lr_esrgan_physic.py — see src/datasets/esrgan_physic_online.py
        for why: a from-scratch run driven by the simpler _synthesize_mixed_curriculum_batch
        port caused a real Cardiac/BUSI regression, most likely because that port doesn't
        match this generator's exact ESRGAN-kernel + resize/noise pipeline closely enough —
        this uses the SAME RealESRGAN_degradation + params.yml already in this repo, so the
        online training distribution is pixel-for-pixel the same family as what the static
        data (and therefore the eval sets) were built from, just freshly re-sampled every
        step instead of fixed once per image.

        --online_clean_prob (2026-07-17 follow-up, memory pisasr-dapmr-phase0-audit): with
        this probability per sample, BYPASS the degrade generator entirely and use the
        clean HR itself as "LR" with label=[0]*6 — the real esrgan_physic_degrade always
        applies at least the ESRGAN backbone (blur+resize+noise), so without this the
        router (Phase 1 pretrain + the router_finetune window's dose regression) NEVER sees
        a genuinely-low-severity example to calibrate against, and was found to predict
        systematically elevated intensities even on in-domain Cardiac test images (mean
        0.44-0.70 vs. controllable_l3_r1_anneal50's 0.05-0.23) — pushing this checkpoint
        family's own restoration tokens well past their empirically-found PSNR-optimal
        operating range (~0.2-0.4), causing measurable overcorrection on OOD BUSI.

        Safety note (this was explicitly raised and reasoned through, not just assumed):
        this does NOT teach the UNet "control=0 -> make it clean" as a general shortcut,
        because control=0 here ALWAYS co-occurs with a genuinely-already-clean input (the
        GT label IS 0, so target=input is the correct answer, not an artificially-forced
        one) — unlike a hypothetical "force control=0 on a HEAVILY degraded input" case,
        which really would teach the wrong lesson. The complementary lesson — "control=0 on
        a DEGRADED input should leave it unchanged, not clean it" — is already taught by
        compute_specificity_batch_loss's mismatched case (a degraded image, wrong slot
        requested, target=degraded input unchanged); the two signals are complementary, not
        redundant: this teaches the router what "low severity" looks like, that one teaches
        the UNet not to over-generalize zero-control into "always clean".
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        clean_prob = float(getattr(args, "online_clean_prob", 0.1))
        random_order = bool(getattr(args, "random_degrad_order", False))
        degraded_images, labels = [], []
        for i in range(B):
            if random.random() < clean_prob:
                degraded_images.append(c_tgt[i:i + 1].clone())
                labels.append(self._new_label_vec())
                continue
            hr_u8 = self._tensor_to_np01(c_tgt[i:i + 1])
            hr_u8 = (hr_u8 * 255.0).clip(0, 255).astype('uint8')
            if random_order:
                lr_u8, label = generate_labeled_lr_random_order(hr_u8)
            else:
                lr_u8, label = generate_labeled_lr(hr_u8)
            lr01 = lr_u8.astype('float32') / 255.0
            degraded_images.append(self._np01_to_tensor(lr01, c_tgt[i:i + 1]))
            labels.append(self._fold_label6(label))

        degraded_batch = torch.cat(degraded_images, dim=0)
        labels_syn = torch.tensor(labels, dtype=torch.float32, device=device)
        return degraded_batch, labels_syn

    def _synthesize_online_esrgan_dosed_batch(self, c_tgt, args):
        """Merged curriculum + specificity synthesis (2026-07-26, user request): builds
        ONE batch mixing full-dose (bit-identical distribution to today's
        _synthesize_online_esrgan_batch / forward_curriculum), partial-dose, and
        mismatch samples, all rendered through generate_labeled_lr_dosed on the SAME
        esrgan_physic generator the static train/eval data came from (NOT the
        independently re-implemented _apply_single_degradation primitives
        compute_specificity_batch_loss uses) -- so ONE UNet forward can train both the
        main SR-reconstruction objective AND the specificity (partial-dose/mismatch)
        objective every step, at the SAME cost as plain forward_curriculum alone (no
        separate specificity forward pass needed -- see forward_curriculum_dosed).

        Per sample:
          - --online_clean_prob: unchanged bypass (clean HR as "LR", label=0), same as
            _synthesize_online_esrgan_batch.
          - With probability --curriculum_dose_isolate_prob: ISOLATE sample (2026-08-12,
            user request). Picks ONE physical degradation type k and renders the input at
            dose=onehot(k) -- i.e. ALL other 5 types suppressed -- matching the
            direct-control audit's exact regime (figure_direct_eval_*.py: single-degradation
            input, one-hot forced control). Every other branch below always renders its
            input at dose=[1]*6 (all 6 types simultaneously present), so a genuinely
            isolated single-degradation input was previously never seen in training at any
            magnitude; the achieved-magnitude collapse/cross-talk the audit found on
            curriculum_dosed_ratio15_25_endpointfix (u=0/u=1 fix alone didn't resolve it)
            is consistent with that gap. u ~ _sample_user_multiplier() as in PARTIAL below;
            control = onehot(k) * u * label_iso[k]; target = same generator replayed at
            dose=onehot(k)*(1-u) (SAME seed -- only slot k's severity changes, others stay
            at their dose=0 identity). full_label for this sample is label_iso (the TRUE,
            actually-isolated composition), NOT the dose=[1]*6 one, so the router-dose loss
            stays consistent with what was actually fed in.
          - Otherwise, draw a seed and render at dose=[1]*6 to get (input_lr,
            full_label) -- pixel-identical to what _synthesize_online_esrgan_batch
            produces for this seed (verified in
            scripts/audit/test_esrgan_physic_dosed_equivalence.py).
          - With probability --curriculum_dose_mismatch_prob (of the remainder): MISMATCH
            sample. control =
            full_label with one extra slot (one that's actually near-zero, i.e. not
            present) bumped to a nonzero request. target = input UNCHANGED (requesting
            removal of something absent must not generic-enhance the image) -- mirrors
            compute_specificity_batch_loss's mismatched case, no extra generator call.
          - Else with probability --curriculum_dose_partial_prob (of the remainder):
            PARTIAL sample. u ~ _sample_user_multiplier() (per-sample scalar, applied
            uniformly to all 6 physical slots): 50% snapped to {0,.25,.5,.75,1} (spec
            C.8's canonical evaluation grid -- so u=0 "keep as-is" and u=1 "fully
            remove" are EXPLICITLY anchored at ~10% probability each, not excluded --
            2026-08-12 fix, see figure_direct_eval_specificity_sweep.py audit: the old
            u ~ U(curriculum_dose_min_u=0.1, curriculum_dose_max_u=0.9) clamp meant
            u=0/u=1 were never trained here, plausibly the cause of the achieved-
            magnitude collapse observed near s=1 on curriculum_dosed_ratio15_25), 50%
            continuous U(0,1) for dose-response coverage in between. control =
            u * full_label (the requested REMOVAL amount). target =
            generate_labeled_lr_dosed(hr, dose=(1-u)*ones(6), seed=seed) -- SAME
            seed/aux realization, rendered at the REMAINING severity after removing
            fraction u; self-consistent with control by construction.
          - Else: FULL sample (today's forward_curriculum behavior, unchanged). control
            = full_label, target = c_tgt (clean).

        Returns: (input_batch [B,C,H,W], control_batch [B,num_deg], target_batch
        [B,C,H,W], full_label_batch [B,num_deg], is_full_dose [B] bool tensor) --
        full_label is always the TRUE severity (regardless of what was requested), for
        the router-dose loss to keep training against; is_full_dose gates which samples
        CSD may fire on (see forward_curriculum_dosed / train_pisasr.py).
        """
        B, C, H, W = c_tgt.shape
        device = c_tgt.device
        clean_prob = float(getattr(args, "online_clean_prob", 0.1))
        mismatch_prob = float(getattr(args, "curriculum_dose_mismatch_prob", 0.1))
        partial_prob = float(getattr(args, "curriculum_dose_partial_prob", 0.3))
        isolate_prob = float(getattr(args, "curriculum_dose_isolate_prob", 0.0))
        mismatch_thresh = 1e-3

        inputs, controls, targets, full_labels, is_full = [], [], [], [], []
        for i in range(B):
            if random.random() < clean_prob:
                inputs.append(c_tgt[i:i + 1].clone())
                controls.append(self._new_label_vec())
                targets.append(c_tgt[i:i + 1].clone())
                full_labels.append(self._new_label_vec())
                is_full.append(True)
                continue

            hr_u8 = self._tensor_to_np01(c_tgt[i:i + 1])
            hr_u8 = (hr_u8 * 255.0).clip(0, 255).astype('uint8')
            seed = random.randint(0, 2 ** 31 - 1)
            r = random.random()

            if r < isolate_prob:
                k_iso = random.randint(0, 5)
                dose_iso = [0.0] * 6
                dose_iso[k_iso] = 1.0
                lr_iso_u8, label_iso6 = generate_labeled_lr_dosed(hr_u8, dose=dose_iso, seed=seed)
                lr_iso01 = lr_iso_u8.astype('float32') / 255.0
                input_iso_t = self._np01_to_tensor(lr_iso01, c_tgt[i:i + 1])
                label_iso = self._fold_label6(label_iso6)
                full_labels.append(label_iso)

                u = self._sample_user_multiplier()
                control = self._new_label_vec()
                control[k_iso] = u * label_iso[k_iso]
                dose_tgt = [0.0] * 6
                dose_tgt[k_iso] = 1.0 - u
                tgt_iso_u8, _ = generate_labeled_lr_dosed(hr_u8, dose=dose_tgt, seed=seed)
                tgt_iso01 = tgt_iso_u8.astype('float32') / 255.0
                target_iso_t = self._np01_to_tensor(tgt_iso01, c_tgt[i:i + 1])

                inputs.append(input_iso_t)
                controls.append(control)
                targets.append(target_iso_t)
                is_full.append(False)
                continue

            lr_u8, full_label6 = generate_labeled_lr_dosed(hr_u8, dose=[1.0] * 6, seed=seed)
            lr01 = lr_u8.astype('float32') / 255.0
            input_t = self._np01_to_tensor(lr01, c_tgt[i:i + 1])
            full_label = self._fold_label6(full_label6)
            full_labels.append(full_label)

            r -= isolate_prob
            if r < mismatch_prob:
                control = list(full_label)
                absent = [j for j in range(self.num_deg) if full_label[j] < mismatch_thresh]
                if absent:
                    wrong = random.choice(absent)
                    control[wrong] = random.uniform(0.3, 1.0)
                inputs.append(input_t)
                controls.append(control)
                targets.append(input_t.clone())
                is_full.append(False)
            elif r < mismatch_prob + partial_prob:
                u = self._sample_user_multiplier()
                control = [u * v for v in full_label]
                tgt_u8, _ = generate_labeled_lr_dosed(hr_u8, dose=[1.0 - u] * 6, seed=seed)
                tgt01 = tgt_u8.astype('float32') / 255.0
                target_t = self._np01_to_tensor(tgt01, c_tgt[i:i + 1])
                inputs.append(input_t)
                controls.append(control)
                targets.append(target_t)
                is_full.append(False)
            else:
                inputs.append(input_t)
                controls.append(list(full_label))
                targets.append(c_tgt[i:i + 1].clone())
                is_full.append(True)

        input_batch = torch.cat(inputs, dim=0)
        control_batch = torch.tensor(controls, dtype=torch.float32, device=device)
        target_batch = torch.cat(targets, dim=0)
        full_label_batch = torch.tensor(full_labels, dtype=torch.float32, device=device)
        is_full_dose = torch.tensor(is_full, dtype=torch.bool, device=device)
        return input_batch, control_batch, target_batch, full_label_batch, is_full_dose

    def forward_curriculum_dosed(self, c_tgt, args):
        """Merged curriculum+specificity forward (2026-07-26, user request) -- see
        _synthesize_online_esrgan_dosed_batch for the synthesis. Unlike
        forward_curriculum (which ALWAYS requests full restoration against clean
        c_tgt), this trains full, partial-dose, and mismatch requests in the SAME loop
        via ONE synthesis + ONE UNet forward, so a separate
        compute_specificity_batch_loss forward is not needed when
        --curriculum_dose_random is set (leave --lambda_specificity at its default 0
        when using this mode).

        Returns (output_image, target_batch, x_denoised, dynamic_prompt,
        neg_prompt_embeds, intensities, is_full_dose) -- target_batch (NOT c_tgt) is
        what the caller's L2/LPIPS loss must compare against; is_full_dose gates
        whether CSD may fire that step (CSD pulls toward a "clean high quality"
        distribution -- applying it against a deliberately partial/mismatched target
        would fight the L2/LPIPS objective, per user confirmation 2026-07-26).
        """
        source = getattr(args, "online_degrade_source", "curriculum")
        if source != "esrgan_physic":
            raise ValueError("--curriculum_dose_random requires --online_degrade_source "
                              "esrgan_physic (the dosed generator only wraps generate_labeled_lr_dosed).")
        (degraded_batch, control_batch, target_batch,
         full_label_batch, is_full_dose) = self._synthesize_online_esrgan_dosed_batch(c_tgt, args)
        bs = degraded_batch.shape[0]

        with torch.amp.autocast('cuda', enabled=False):
            posterior = self.vae_fix.encode(degraded_batch.float()).latent_dist
            encoded_src = posterior.sample() * self.vae_fix.config.scaling_factor
        encoded_src = encoded_src.to(dtype=degraded_batch.dtype)
        encoded_control = encoded_src.detach().requires_grad_(True)

        force_val = getattr(args, "force_router_intensity", None)
        if force_val is not None:
            intensities = torch.full(
                (bs, self.num_deg), float(force_val), device=degraded_batch.device, dtype=torch.float32)
            router_pred = None
        else:
            beta = self._teacher_forcing_alpha(args)
            router_pred = None
            if beta < 1.0 or getattr(args, "lambda_router_dose_online", 0.0) > 0:
                with torch.amp.autocast('cuda', enabled=False):
                    encoded_router = posterior.mode() * self.vae_fix.config.scaling_factor
                latent_fp = encoded_router.to(self._router_param_dtype())
                router_pred = self._compute_intensities(latent_fp)
            if beta < 1.0 and router_pred is not None:
                blended = beta * control_batch + (1.0 - beta) * router_pred
            else:
                blended = control_batch
            # Teacher-forcing blend with the router's own prediction is only
            # semantically valid on full-dose samples (both represent "how much IS
            # actually wrong" -- router_pred estimates the TRUE severity, matching
            # full-dose's control==full_label). Partial/mismatch samples' control is a
            # DELIBERATE request that differs from the true severity by construction;
            # blending it with router_pred would corrupt that deliberate signal, so
            # those samples always use their exact control vector unblended,
            # regardless of beta.
            full_mask = is_full_dose.unsqueeze(-1).to(control_batch.dtype)
            intensities = full_mask * blended + (1.0 - full_mask) * control_batch

        injection_mode = getattr(args, "lgdp_injection_mode", "direct_feed")
        if injection_mode == "ram_tag":
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_ram(degraded_batch)
        elif injection_mode in ("text_ordinal", "text_numeric"):
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_text(intensities, args, injection_mode)
        else:
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt(intensities, args)

        with torch.no_grad():
            neg_ids = self.tokenizer(
                "depth blur, background haze, signal attenuation, speckle noise, "
                "electronic noise, compression artifact, low quality ultrasound",
                max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            _prompt_len = dynamic_prompt.shape[1]
            neg_prompt_embeds = self.text_encoder(neg_ids)[0][:, :_prompt_len, :].expand(bs, -1, -1)

        self._activate_pix_sem_adapters()
        model_pred = self.unet(
            encoded_control, self.timesteps1,
            encoder_hidden_states=dynamic_prompt, cross_attention_kwargs=None,
        ).sample

        self.last_prompt_weights = lgdp_debug["prompt_weights"][0].detach().cpu()
        self.last_intensities    = intensities[0].detach().cpu()
        self.last_labels_syn     = full_label_batch
        self.last_router_pred    = router_pred

        x_denoised = encoded_control - model_pred
        with torch.amp.autocast('cuda', enabled=False):
            output_image = (self.vae_fix.decode(x_denoised.float() / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return (output_image.to(dtype=c_tgt.dtype), target_batch, x_denoised, dynamic_prompt,
                neg_prompt_embeds, intensities, is_full_dose)

    def compute_mixed_curriculum_sr_loss(self, c_tgt, args, beta=1.0, detach_router=True):
        """Mixed-degradation-curriculum synthetic SR loss with an explicit teacher-forcing
        beta: beta=1.0 (Stage 1 "controller") feeds the prompt directly from ground truth
        (c=d, router never called at all — Stage 1 trains the degradation-prompt directions/
        scale in isolation from router error). beta<1.0 (Stage 2 "router", via
        _teacher_forcing_alpha's anneal) blends toward the router's OWN prediction on this
        same synthesized image; detach_router=False (Stage 2's setting) lets the
        reconstruction loss's gradient flow into router_pred so the router is trained
        end-to-end by the actual downstream task, not only by the separate dose/rank
        regression loss. detach_router=True (default, unused by Stage 1/2 as designed but
        kept as a safe default) mirrors compute_synthetic_sr_loss's existing behavior.

        Returns: (output_image, labels_syn, router_pred_or_None) — router_pred is None
        only when beta=1.0 AND detach_router=True (the router genuinely isn't needed at
        all in that combination). Stage 2 always passes detach_router=False, so router_pred
        is ALWAYS populated there (with gradient) even during the early portion of a
        teacher-forcing anneal where beta is still 1.0 — its own dose/rank loss needs the
        router's live prediction every step, independent of whether that step's
        RECONSTRUCTION conditioning happens to blend it in yet.
        """
        degraded_batch, labels_syn = self._synthesize_mixed_curriculum_batch(c_tgt, args)
        router_pred = None
        if beta < 1.0 or not detach_router:
            router_pred, _ = self.encode_degradation_prediction(degraded_batch)
        if beta < 1.0:
            rp = router_pred.detach() if detach_router else router_pred
            intensities = beta * labels_syn + (1.0 - beta) * rp
        else:
            intensities = labels_syn
        output_image = self._forward_with_prompt(degraded_batch, intensities, args, sample_encode=True)
        return output_image, labels_syn, router_pred


def active_inactive_ranking_loss(router_pred, labels_syn, margin=0.1, active_thresh=1e-4):
    """Margin ranking loss (2026-07-16 follow-up, Stage 2 "router" dose supervision):
    for each sample, every ACTIVE slot's predicted intensity (labels_syn > active_thresh,
    i.e. that degradation is actually present) should exceed every INACTIVE slot's by at
    least `margin`. Directly targets the matched-slot-rank failure mode found across every
    prior audit (memory pisasr-dapmr-phase0-audit): even when the plain SmoothL1 dose loss
    gets each slot's magnitude roughly right on average, it does not by itself enforce that
    present-degradation slots outrank absent ones within the SAME image — which is what a
    downstream ranking-sensitive consumer (argmax-like routing, or the response-matrix
    diagonal-vs-offdiagonal comparison used throughout this audit) actually needs. speckle
    was the single most consistent failure (matched-slot rank 5-6/6 in every checkpoint
    audited so far), motivating this term specifically.

    router_pred, labels_syn: [B, 6]. Returns a scalar (0 with grad if some sample has no
    valid active/inactive split, e.g. all-clean or all-degraded).
    """
    active = (labels_syn > active_thresh)  # [B, K] bool
    per_sample_losses = []
    for b in range(router_pred.shape[0]):
        act_idx = active[b].nonzero(as_tuple=True)[0]
        inact_idx = (~active[b]).nonzero(as_tuple=True)[0]
        if len(act_idx) == 0 or len(inact_idx) == 0:
            continue
        ri = router_pred[b, act_idx].unsqueeze(1)     # [A, 1]
        rj = router_pred[b, inact_idx].unsqueeze(0)    # [1, I]
        per_sample_losses.append(F.relu(margin - ri + rj).mean())
    if not per_sample_losses:
        return torch.zeros((), device=router_pred.device, dtype=router_pred.dtype, requires_grad=True)
    return torch.stack(per_sample_losses).mean()


class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        # --lgdp_injection_mode ram_tag: mirrors PiSASR.__init__ (train class) — see
        # load_ram_tagger/build_ram_tag_prompt.
        if getattr(args, "lgdp_injection_mode", "direct_feed") == "ram_tag":
            self.ram_model = load_ram_tagger(args.ram_pretrained_path, self.device)

        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        # No domain/base prompt: the UNet's only text condition is [BOS] + the 6 degradation-
        # restoration tokens (see _compose_lgdp_prompt). BOS is taken from encoding "" once —
        # the contextualized attention-sink anchor token.
        with torch.no_grad():
            null_ids = self.tokenizer(
                "", max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            self.null_prompt_embeds = self.text_encoder(null_ids)[0]  # [1, 77, D]

        # Optional static domain prompt — mirrors PiSASR.__init__ (train class); learnable, so
        # the checkpoint's TRAINED values (loaded below in _load_and_save_ckpt_from_state_dict)
        # take precedence over this initial text-encoder-derived placeholder whenever the
        # checkpoint has one. --domain_prompt here only matters as a fallback for checkpoints
        # saved before this feature existed (no "domain_prompt_content" key).
        with torch.no_grad():
            domain_content, domain_eos = encode_domain_prompt(
                self.tokenizer, self.text_encoder, self.device,
                getattr(args, "domain_prompt", ""),
            )
        self.domain_prompt_content = nn.Parameter(domain_content, requires_grad=False)
        self.domain_prompt_eos = nn.Parameter(domain_eos, requires_grad=False)

        # LGDP restoration tokens — mirrors PiSASR.__init__ (train class); see there for
        # the two mutually exclusive schemes (default EOS-pooled vs --lgdp_variable_tokens).
        # All of these are overwritten by checkpoint load below regardless of scheme, EXCEPT
        # that --lgdp_token_scheme (like --router_arch) must be passed matching the
        # checkpoint's training-time value BEFORE this point, since the router/restoration-
        # token/scale modules below are shape-constructed from self.num_deg and the
        # checkpoint load uses strict load_state_dict/copy_ (asserted below).
        self.lgdp_variable_tokens = bool(getattr(args, "lgdp_variable_tokens", False))
        self.lgdp_token_scheme = getattr(args, "lgdp_token_scheme", "full6")
        # --lgdp_text_phrase_style: only consulted by _build_lgdp_text_prompt (text_ordinal/
        # text_numeric injection modes) — which vocabulary names each degradation slot with.
        self.lgdp_text_phrase_style = getattr(args, "lgdp_text_phrase_style", "degradation_noun")
        if self.lgdp_token_scheme not in ("full6", "merged3"):
            raise ValueError(f"Unknown --lgdp_token_scheme {self.lgdp_token_scheme!r} "
                              "(expected 'full6' or 'merged3')")
        if self.lgdp_token_scheme == "full6":
            self.num_deg = len(DEGRADATION_NAMES)  # 6
            self.deg_names = list(DEGRADATION_NAMES)
        else:
            self.num_deg = len(DEGRADATION_NAMES_MERGED3)  # 3
            self.deg_names = list(DEGRADATION_NAMES_MERGED3)
        prompt_bank_mode = getattr(args, "prompt_bank_mode", "target_state")
        if self.lgdp_variable_tokens:
            self.text_prototypes = None
            with torch.no_grad():
                concat_tokens, self.token_segment_lengths = compute_degradation_content_tokens(
                    self.text_encoder, self.tokenizer, self.device, mode=prompt_bank_mode,
                    scheme=self.lgdp_token_scheme,
                )
            self.restoration_tokens = nn.Parameter(concat_tokens)  # [1, L_total, D]
        else:
            self.token_segment_lengths = None
            with torch.no_grad():
                self.text_prototypes = nn.Parameter(
                    compute_degradation_text_prototypes(
                        self.text_encoder, self.tokenizer, self.device,
                        mode=prompt_bank_mode, return_null_centered=False,
                        scheme=self.lgdp_token_scheme,
                    )
                )  # [1, num_deg, D]
            self.restoration_tokens = nn.Parameter(self.text_prototypes.detach().clone())  # [1, K, D]

        _rts_init = getattr(args, "restoration_token_init_scale", 1.0)
        _rts_raw  = torch.tensor(float(_rts_init)).expm1().clamp(min=1e-8).log().item()
        self.restoration_token_scale = nn.Parameter(
            torch.full((self.num_deg,), _rts_raw)
        )  # [K]; effective scale = softplus(this) >= 0

        self.ctx_dim = self.text_encoder.config.hidden_size

        # Must match the checkpoint's training-time --router_arch (asserted in
        # _load_and_save_ckpt_from_state_dict below) — see PiSASR.__init__ for rationale.
        self.router_arch = getattr(args, "router_arch", "base")
        if self.router_arch == "conv":
            self.conv_router = ConvRouterHead(in_ch=4, num_deg=self.num_deg)
        else:
            # 4 channels × 3×3 patch = 36 features per token; lets router detect blur/noise spatially
            self.vis_projector = nn.Sequential(
                nn.Linear(36, 128),
                nn.GELU(),
                nn.Linear(128, self.ctx_dim),
            )
            self.vis_norm = nn.LayerNorm(self.ctx_dim)
            self.spatial_pool = nn.Linear(self.ctx_dim * 6, self.ctx_dim)

            self.router = DegradationRouter(self.ctx_dim, num_deg=self.num_deg)

        self._load_pretrained_weights(args.pretrained_path)

        if not args.default:
            self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        self._move_models_to_device_and_dtype()

        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _get_dtype(self, precision):
        if precision == "fp16": return torch.float16
        elif precision == "bf16": return torch.bfloat16
        else: return torch.float32

    def _move_models_to_device_and_dtype(self):
        router_models = [self.conv_router] if self.router_arch == "conv" else \
            [self.vis_projector, self.vis_norm, self.spatial_pool, self.router]
        for model in [self.unet, self.text_encoder] + router_models:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)
        if self.text_prototypes is not None:
            self.text_prototypes.data = self.text_prototypes.data.to(self.device, dtype=self.weight_dtype)
        self.restoration_tokens.data = self.restoration_tokens.data.to(self.device, dtype=self.weight_dtype)
        self.restoration_token_scale.data = self.restoration_token_scale.data.to(self.device, dtype=self.weight_dtype)
        self.null_prompt_embeds = self.null_prompt_embeds.to(self.device, dtype=self.weight_dtype)
        self.domain_prompt_content.data = self.domain_prompt_content.data.to(self.device, dtype=self.weight_dtype)
        self.domain_prompt_eos.data = self.domain_prompt_eos.data.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=torch.float32)  # VAE kept in fp32
        self.vae.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sd = torch.load(pretrained_path, map_location=device)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            base_layer = target.get_base_layer()
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight

                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):
                                output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:
                                output_tensor = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:
                                output_tensor = F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3) * target.scaling[active_adapter]

                            key = key + ".weight"
                            self.lora_deltas_sem[key] = output_tensor.data.to(dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        set_weights_and_activate_adapters(self.unet, ["default_encoder_pix", "default_decoder_pix", "default_others_pix"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone()
            self.ori_unet_weight[name] = self.ori_unet_weight[name].data.to(self.weight_dtype).to("cuda")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        # Diagnostic: override the BOS anchor with the position-0 output of arbitrary text
        # (instead of encoding "").
        _bp = os.environ.get("BASE_PROMPT")
        if _bp is not None:
            _ids = self.tokenizer(_bp, max_length=self.tokenizer.model_max_length,
                                  padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                self.null_prompt_embeds = self.text_encoder(_ids)[0].to(dtype=self.weight_dtype)
            print(f"[BASE_PROMPT override] '{_bp}'", flush=True)

        # Asserted BEFORE text_prototypes/router are touched below: a scheme/arch mismatch
        # must fail with this clear message, not a bare shape-mismatch RuntimeError from
        # whichever copy_/load_state_dict call happens to run into it first.
        ckpt_router_arch = sd.get("router_arch", "base")
        assert ckpt_router_arch == self.router_arch, (
            f"Checkpoint was trained with router_arch={ckpt_router_arch!r} but eval was "
            f"started with --router_arch {self.router_arch!r} — pass the matching flag.")
        ckpt_token_scheme = sd.get("lgdp_token_scheme", "full6")
        assert ckpt_token_scheme == self.lgdp_token_scheme, (
            f"Checkpoint was trained with lgdp_token_scheme={ckpt_token_scheme!r} but eval "
            f"was started with --lgdp_token_scheme {self.lgdp_token_scheme!r} — pass the "
            f"matching flag.")

        # text_prototypes is None in --lgdp_variable_tokens mode (no single pooled token to
        # store) — skip it there.
        if self.text_prototypes is not None and sd.get("text_prototypes") is not None:
            self.text_prototypes.data.copy_(sd["text_prototypes"].to(self.device, dtype=self.weight_dtype))
        if self.router_arch == "conv":
            self.conv_router.load_state_dict(sd["conv_router"])
        else:
            self.router.load_state_dict(sd["router"])
            self.vis_projector.load_state_dict(sd["vis_projector"])
            self.vis_norm.load_state_dict(sd["vis_norm"])
            self.spatial_pool.load_state_dict(sd["spatial_pool"])

        # restoration_tokens' shape depends on the token scheme (fixed [1,6,D] vs variable
        # [1,L_total,D]) — reassign outright (not copy_'d) so eval doesn't need
        # --lgdp_variable_tokens set correctly to match the checkpoint (mirrors
        # domain_prompt_content's pattern below).
        self.restoration_tokens = nn.Parameter(
            sd["restoration_tokens"].to(self.device, dtype=self.weight_dtype), requires_grad=False)
        self.restoration_token_scale.data.copy_(sd["restoration_token_scale"].to(self.device, dtype=self.weight_dtype))
        if "token_segment_lengths" in sd and sd["token_segment_lengths"] is not None:
            self.lgdp_variable_tokens = True
            self.token_segment_lengths = sd["token_segment_lengths"]

        # Learnable domain prompt: prefer the checkpoint's TRAINED values over the
        # --domain_prompt-derived placeholder from __init__. Reassigned outright (not
        # copy_'d) because its length depends on the domain prompt's own token count and
        # may not match __init__'s placeholder shape. Older checkpoints saved before this
        # feature existed simply lack the key, so the __init__ placeholder (content of
        # whatever --domain_prompt was passed, "" by default = 0-length) stays in effect.
        if "domain_prompt_content" in sd:
            self.domain_prompt_content = nn.Parameter(
                sd["domain_prompt_content"].to(self.device, dtype=self.weight_dtype), requires_grad=False)
            self.domain_prompt_eos = nn.Parameter(
                sd["domain_prompt_eos"].to(self.device, dtype=self.weight_dtype), requires_grad=False)

    def _compose_lgdp_prompt(self, intensities_pred, args):
        """Compose dynamic LGDP prompt for eval — mirrors PiSASR._compose_lgdp_prompt exactly."""
        K = self.num_deg
        weights = intensities_pred.float()  # [B, K]
        B = weights.shape[0]

        # No F.normalize: restoration_tokens are raw (non-null-centered) CLIP EOS-pooled
        # sentence embeddings, so they already carry a natural, CLIP-consistent magnitude
        # (measured ~34.5 for this text encoder — same order as BOS's ~17.25, actually
        # somewhat larger) — no need to discard that direction+magnitude and reconstruct an
        # artificial one via unit-normalize + BOS-norm anchoring. Discarding the natural
        # magnitude (via F.normalize) and replacing it with a uniform scale/anchor is what
        # made every token's audibility identical regardless of its own semantic content;
        # using the raw embedding keeps both direction AND magnitude as CLIP produced them.
        bos = self.null_prompt_embeds[:, 0:1, :].expand(B, -1, -1)  # [B, 1, D]

        scale    = F.softplus(self.restoration_token_scale)  # [K]
        strength = float(getattr(args, "lgdp_prompt_strength", 1.0))

        if self.lgdp_variable_tokens:
            # Each degradation's router intensity/scale is broadcast across its OWN
            # variable-length token block (self.token_segment_lengths[i] tokens), not one
            # weight per token — a degradation with a longer description gets more tokens
            # in the prompt, all carrying that SAME degradation's single intensity value.
            weights_expanded = torch.cat(
                [weights[:, i:i + 1].expand(-1, n) for i, n in enumerate(self.token_segment_lengths)],
                dim=1,
            )  # [B, L_total]
            scale_expanded = torch.cat(
                [scale[i:i + 1].expand(n) for i, n in enumerate(self.token_segment_lengths)]
            )  # [L_total]
            tokens = (strength * weights_expanded.unsqueeze(-1) * scale_expanded.view(1, -1, 1).to(weights.dtype)
                      * self.restoration_tokens.to(weights.dtype))  # [B, L_total, D]
        else:
            tokens = (strength * weights.unsqueeze(-1) * scale.view(1, K, 1).to(weights.dtype)
                      * self.restoration_tokens.to(weights.dtype))  # [B, K, D]

        parts = [bos.to(tokens.dtype), tokens]
        # Optional domain prompt: [BOS] + [degrad tokens] + [domain content] + [EOS].
        # domain_prompt_content has 0 length when --domain_prompt is unset/empty, so this
        # is a no-op and the sequence is bit-identical to the no-domain-prompt case.
        if self.domain_prompt_content.shape[1] > 0:
            parts.append(self.domain_prompt_content.to(tokens.dtype).expand(B, -1, -1))
            parts.append(self.domain_prompt_eos.to(tokens.dtype).expand(B, -1, -1))

        dynamic_prompt = torch.cat(parts, dim=1).float()  # [B, 1+K(+domain+1), D]
        dynamic_prompt = self._pad_lgdp_prompt(dynamic_prompt, args)

        return dynamic_prompt, {"prompt_weights": weights}

    def _pad_lgdp_prompt(self, dynamic_prompt, args):
        """Mirrors PiSASR._pad_lgdp_prompt exactly — see there for rationale."""
        prompt_length = int(getattr(args, "lgdp_prompt_length", 7))
        if prompt_length == 77 and dynamic_prompt.shape[1] < 77:
            B = dynamic_prompt.shape[0]
            pad_tail = self.null_prompt_embeds[:, dynamic_prompt.shape[1]:, :].expand(B, -1, -1)
            dynamic_prompt = torch.cat([dynamic_prompt, pad_tail.to(dynamic_prompt.dtype)], dim=1)
        return dynamic_prompt

    def _build_lgdp_text_prompt(self, intensities_pred, mode):
        """Mirrors PiSASR._build_lgdp_text_prompt exactly — see there for rationale."""
        phrase_style = getattr(self, "lgdp_text_phrase_style", "degradation_noun")
        if phrase_style == "restoration_verb":
            phrases = RESTORATION_VERB_PHRASES_FULL6 if self.lgdp_token_scheme == "full6" \
                else RESTORATION_VERB_PHRASES_MERGED3
        elif phrase_style == "degradation_noun":
            phrases = NOUN_PHRASES_FULL6 if self.lgdp_token_scheme == "full6" else NOUN_PHRASES_MERGED3
        else:
            raise ValueError(f"Unknown --lgdp_text_phrase_style {phrase_style!r} "
                              "(expected 'degradation_noun' or 'restoration_verb')")
        values = intensities_pred.detach().float().cpu()  # [B, K]
        texts = []
        for b in range(values.shape[0]):
            if mode == "text_ordinal":
                clauses = []
                for k, phrase in enumerate(phrases):
                    v = values[b, k].item()
                    bucket = "slight" if v < 1.0 / 3.0 else ("middle" if v < 2.0 / 3.0 else "strong")
                    clauses.append(f"{bucket} {phrase}")
                texts.append("restore " + ", ".join(clauses))
            elif mode == "text_numeric":
                clauses = [f"{phrase} level {values[b, k].item():.2f}" for k, phrase in enumerate(phrases)]
                texts.append(", ".join(clauses))
            else:
                raise ValueError(f"Unknown text prompt mode {mode!r} (expected 'text_ordinal' or 'text_numeric')")
        return texts

    def _compose_lgdp_prompt_text(self, intensities_pred, args, mode):
        """Mirrors PiSASR._compose_lgdp_prompt_text exactly — see there for rationale."""
        texts = self._build_lgdp_text_prompt(intensities_pred, mode)
        with torch.no_grad():
            ids = self.tokenizer(
                texts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            dynamic_prompt = self.text_encoder(ids)[0].float()  # [B, 77, D]
        return dynamic_prompt, {"prompt_weights": intensities_pred.float()}

    def _compose_lgdp_prompt_ram(self, x_img):
        """Mirrors PiSASR._compose_lgdp_prompt_ram exactly — see there for rationale."""
        texts = build_ram_tag_prompt(self.ram_model, x_img)
        with torch.no_grad():
            ids = self.tokenizer(
                texts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.text_encoder.device)
            dynamic_prompt = self.text_encoder(ids)[0].float()  # [B, 77, D]
        prompt_weights = torch.zeros(x_img.shape[0], self.num_deg, device=x_img.device)
        return dynamic_prompt, {"prompt_weights": prompt_weights}

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        if self.router_arch == "conv":
            self.conv_router.eval()
        else:
            self.router.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def _router_param_dtype(self):
        if self.router_arch == "conv":
            return next(self.conv_router.parameters()).dtype
        return next(self.vis_projector.parameters()).dtype

    def _compute_intensities(self, latent_fp):
        """Mirrors PiSASR._compute_intensities exactly — see there for rationale."""
        if self.router_arch == "conv":
            return self.conv_router(latent_fp)
        _, _, H_lat, W_lat = latent_fp.shape
        H_tok = (H_lat - 1) // 2 + 1
        W_tok = (W_lat - 1) // 2 + 1
        patches = F.unfold(latent_fp, kernel_size=3, stride=2, padding=1).transpose(1, 2)
        tokens = self.vis_norm(self.vis_projector(patches))
        feat = _structured_spatial_pool(self.spatial_pool, tokens, H_tok, W_tok)
        return self.router(feat)

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        c_t = c_t.to(dtype=self.weight_dtype)

        # Single VAE encode shared for router AND UNet control (encoded_control below is the
        # same tensor). mode(), not sample(): at inference there is no training-time
        # regularization rationale for encoder noise, and this tensor drives both the router's
        # degradation-intensity prediction and the actual SR output — sample() made both
        # non-deterministic across repeated calls on the identical input image.
        with torch.amp.autocast('cuda', enabled=False):
            encoded_src = self.vae.encode(c_t.float()).latent_dist.mode() * self.vae.config.scaling_factor
        encoded_for_router = encoded_src.to(dtype=c_t.dtype)

        bs = encoded_for_router.shape[0]

        # Router: predict per-degradation intensities from the LR visual evidence.
        lat_for_router = encoded_for_router.to(self._router_param_dtype())
        intensities_pred = self._compute_intensities(lat_for_router)  # [B, 6], already in [0, 1]

        # --router_gain (2026-07-17, inference-only, no retraining): a scalar multiplier on
        # the router's OWN per-image prediction (preserves relative per-image/per-slot
        # variation, unlike --force_router_intensity which replaces the prediction with a
        # fixed constant). Motivated by a checkpoint-family fine-grained sweep
        # (memory pisasr-dapmr-phase0-audit) that found these restoration tokens' PSNR peaks
        # around strength 0.2-0.4 and degrades smoothly past that — checkpoints whose router
        # was trained on the online esrgan_physic generator (which never shows the router a
        # near-clean example) predict mean intensities of 0.44-0.79, well past that
        # empirically-optimal range. Default 1.0 = no change.
        router_gain = float(getattr(self.args, "router_gain", 1.0))
        if router_gain != 1.0:
            intensities_pred = torch.clamp(intensities_pred * router_gain, 0.0, 1.0)

        # --router_gain_slot / --router_gain_slot_value: apply a gain multiplier to only ONE
        # of the 6 degradation slots (index into DEGRAD_TYPES), leaving the other 5 at their
        # raw predicted (or --router_gain-scaled) value. Used to isolate which single slot
        # drives the downstream-Dice sensitivity seen when scaling all 6 slots together.
        slot_idx = getattr(self.args, "router_gain_slot", None)
        if slot_idx is not None:
            slot_gain = float(getattr(self.args, "router_gain_slot_value", 1.0))
            intensities_pred = intensities_pred.clone()
            intensities_pred[:, int(slot_idx)] = torch.clamp(
                intensities_pred[:, int(slot_idx)] * slot_gain, 0.0, 1.0)

        # --router_gain_vec: general per-slot multiplicative gain, one value per degradation
        # slot (order: psf_blur, haze, attenuation, speckle, additive_noise, compression).
        # Generalizes --router_gain_slot to arbitrary combinations, e.g. boosting several
        # slots while suppressing others simultaneously.
        gain_vec_str = getattr(self.args, "router_gain_vec", None)
        if gain_vec_str is not None:
            _gv = [float(v) for v in gain_vec_str.split(",")]
            assert len(_gv) == 6, f"--router_gain_vec needs 6 comma-separated values, got {len(_gv)}"
            gain_vec_t = torch.tensor(_gv, device=intensities_pred.device, dtype=intensities_pred.dtype).unsqueeze(0)
            intensities_pred = torch.clamp(intensities_pred * gain_vec_t, 0.0, 1.0)

        # --router_force_slot / --router_force_slot_value: REPLACE (not scale) one slot's
        # value with a fixed constant for every image, leaving the other 5 slots at their
        # real per-image router prediction. Unlike --router_gain_slot (multiplicative, so a
        # slot that's already near-zero stays near-zero) this can push a slot to an absolute
        # value regardless of what the router itself predicted -- used to test whether a
        # slot's OWN calibration matters, by asking "what if it were pinned to value X for
        # every image" (e.g. router_cf_ablation's finding that the standalone router already
        # near-constant-outputs ~0.9-0.97 haze on BUSI regardless of true content: pinning
        # Cardiac's haze slot to 0.9 too tests whether that accidental always-on behavior is
        # itself what makes BUSI eval look good, rather than genuine per-image adaptation).
        force_slot_idx = getattr(self.args, "router_force_slot", None)
        if force_slot_idx is not None:
            force_slot_val = float(getattr(self.args, "router_force_slot_value", 1.0))
            intensities_pred = intensities_pred.clone()
            intensities_pred[:, int(force_slot_idx)] = force_slot_val

        self.last_intensities = intensities_pred[0].detach().cpu()  # store for inspection

        encoded_control = encoded_src.to(dtype=self.weight_dtype)

        # Degradation prompt: [BOS] + 6 restoration tokens weighted by router intensity —
        # the UNet's only text condition. --lgdp_injection_mode text_ordinal/text_numeric
        # instead builds a literal sentence from intensities_pred (see _compose_lgdp_prompt_text).
        injection_mode = getattr(self.args, "lgdp_injection_mode", "direct_feed")
        if injection_mode == "ram_tag":
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_ram(c_t)
        elif injection_mode in ("text_ordinal", "text_numeric"):
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt_text(intensities_pred, self.args, injection_mode)
        else:
            dynamic_prompt, lgdp_debug = self._compose_lgdp_prompt(intensities_pred, self.args)
        self.last_prompt_weights = lgdp_debug["prompt_weights"][0].detach().cpu()

        if getattr(self.args, "debug_router", False):
            _names = list(DEGRADATION_NAMES)
            _it = self.last_intensities.tolist()
            print("[router] " + " ".join(f"{n}={v:.3f}" for n, v in zip(_names, _it)), flush=True)

        model_pred = self._process_latents(
            encoded_control, dynamic_prompt.to(self.weight_dtype), default, cross_attn_kwargs=None)

        x_denoised = encoded_control - model_pred

        with torch.amp.autocast('cuda', enabled=False):
            output_image = self.vae.decode(x_denoised.float() / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        return 0, output_image.to(dtype=c_t.dtype)

    def _process_latents(self, encoded_control, prompt_embeds, default, cross_attn_kwargs=None,
                         lambda_pix_override=None, lambda_sem_override=None):
        # lambda_pix_override / lambda_sem_override: per-image PiSA pix/sem scales, when supplied
        # by a caller. The SAME scalars are used for every tile of one image.
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap
        if h * w <= tile_size * tile_size:
            return self._predict_no_tiling(encoded_control, prompt_embeds, default, cross_attn_kwargs,
                                           lambda_pix_override, lambda_sem_override)
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap,
                                         cross_attn_kwargs, lambda_pix_override, lambda_sem_override)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default, cross_attn_kwargs=None,
                           lambda_pix_override=None, lambda_sem_override=None):
        if default:
            return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
        model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
        self._apply_ori_weight()
        model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
        self._apply_lora_delta()
        model_pred_sem -= model_pred_pix
        lambda_pix_eff = self.lambda_pix if lambda_pix_override is None else lambda_pix_override
        lambda_sem_eff = self.lambda_sem if lambda_sem_override is None else lambda_sem_override
        return lambda_pix_eff * model_pred_pix + lambda_sem_eff * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap, cross_attn_kwargs=None,
                             lambda_pix_override=None, lambda_sem_override=None):
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1
        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    if default:
                        model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
                    else:
                        model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
                        self._apply_ori_weight()
                        model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attn_kwargs).sample
                        self._apply_lora_delta()
                        model_out_sem = model_out_sem - model_out_pix
                        lambda_pix_eff = self.lambda_pix if lambda_pix_override is None else lambda_pix_override
                        lambda_sem_eff = self.lambda_sem if lambda_sem_override is None else lambda_sem_override
                        model_out = lambda_pix_eff * model_out_pix + lambda_sem_eff * model_out_sem
                    input_list = []
                noise_preds.append(model_out)

        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np
        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]
        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        encoder, decoder = self.vae.encoder, self.vae.decoder
        if not hasattr(encoder, 'original_forward'): encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'): decoder.original_forward = decoder.forward
        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
