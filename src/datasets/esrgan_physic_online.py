"""Online (per-step) port of /mnt/HDD2/tuong/LL/data/Real_Ultrasound/generate_lr_esrgan_physic.py
(2026-07-16 follow-up to the mixed-degradation + oracle-router audit, memory
pisasr-dapmr-phase0-audit): that script is what actually generated the STATIC LR files in
Real_Ultrasound_esrgan_physic/*/train.txt — i.e. the training/eval LR distribution was
never "real acquired ultrasound", it's this same ESRGAN-backbone + US-physics generator,
just pre-rendered once. A from-scratch training run driven by a SIMPLER, independently
re-implemented degradation port (PiSASR._apply_single_degradation) turned out to not match
this generator closely enough (no exact kernel sampling, no JPEG/resize-noise pipeline, only
a single Gaussian blur) and caused a real Cardiac/BUSI regression — likely a train/eval
degradation-distribution mismatch, not a "lost real-domain signal" problem per se.

This module re-implements the SAME generator (using the LOCAL RealESRGAN_degradation +
params.yml already in this directory, NOT the original script's stale external path) so
online per-step generation is pixel-for-pixel the same family of degradation the static
training/eval data came from, while additionally exposing a 6-dim ground-truth severity
label (DEGRADATION_NAMES order) per generated image — needed so the SAME image can drive
both the SR reconstruction loss AND router/dose supervision.

One structural wrinkle: unlike the simplified port, this generator does NOT have psf_blur
and compression as two independently controllable knobs — both come from the SAME
continuous ESRGAN backbone pipeline (kernel1 -> resize -> noise -> [kernel2] -> resize2 ->
noise2 -> final resize), driven by one shared `scale_final` for the net resolution loss.
`scale_final` is used directly as the compression label; psf_blur has no scalar exposed by
the kernel-sampling code, so its severity is estimated POST-HOC from the actual sampled
kernel's spatial spread (intensity-weighted second moment) — a legitimate severity measure
of the kernel that was actually used, robust to kernel shape (iso/aniso/generalized/plateau
all reduce to one spread number), not requiring any change to basicsr's kernel functions.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# US-physics parameter ranges — copied verbatim from generate_lr_esrgan_physic.py's `US`
# dict so the online generator matches the static-data generator exactly.
US = dict(
    scale_range=(1.5, 4.0),
    atten=(0.40, 0.80),
    atten_prob=0.9,
    spk_m=(0.4, 0.9),
    spk_strength=(0.12, 0.35),
    spk_sigma_ax=(0.4, 0.7),
    spk_sigma_lat=(1.2, 2.2),
    add_sigma=(0.0, 0.06),
    haze=(0.0, 0.12),
    haze_sigma=(5.0, 8.0),
    haze_ref=0.45,
    drop_jpeg=True, drop_sinc=True,
)

# Calibrated normalization ceilings so each raw quantity maps roughly to [0, 1].
# blur: combined (in-quadrature) kernel1+kernel2 spread ceiling, calibrated empirically
# (2026-07-16 session: 500 sampled kernel pairs -> mean 2.78, median 2.72, p95 4.89,
# max 6.51 -> ceiling set at the p95, so most samples land in the mid-range of [0,1]
# with only the sharpest ~5% tail clipped to 1.0).
_BLUR_SIGMA_EFF_CEIL = 4.9

_deg = None


def _get_deg():
    global _deg
    if _deg is None:
        import sys
        sys.path.insert(0, _CUR_DIR)
        from realesrgan import RealESRGAN_degradation
        _deg = RealESRGAN_degradation('params.yml', device='cpu')
    return _deg


def _sample(rng, v):
    return float(rng.uniform(*v)) if isinstance(v, (tuple, list)) else v


def sample_us_params(rng, R=US):
    keys = ['atten', 'spk_m', 'spk_strength', 'spk_sigma_ax', 'spk_sigma_lat', 'add_sigma', 'haze', 'haze_sigma']
    p = {k: _sample(rng, R[k]) for k in keys}
    p['atten_prob'] = R['atten_prob']
    p['haze_ref'] = R['haze_ref']
    return p


def _kernel_sigma_eff(kernel):
    """Intensity-weighted spatial second moment of a (possibly non-Gaussian: aniso/
    generalized/plateau/sinc) blur kernel — a shape-agnostic severity proxy. kernel: 2D
    numpy array, assumed to sum to ~1 (basicsr's kernels are normalized)."""
    k = np.asarray(kernel, dtype=np.float64)
    k = k / (k.sum() + 1e-12)
    h, w = k.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = (ys * k).sum()
    cx = (xs * k).sum()
    var = ((ys - cy) ** 2 * k).sum() + ((xs - cx) ** 2 * k).sum()
    return float(np.sqrt(max(var, 0.0)))


def esrgan_backbone_labeled(hr, seed, scale_final, drop_jpeg=True, drop_sinc=True):
    """Verbatim port of generate_lr_esrgan_physic.py's esrgan_backbone(), using the LOCAL
    RealESRGAN_degradation, plus a returned blur-severity scalar computed from the actual
    sampled kernel1/kernel2. hr: uint8 HxW. Returns (lq_u8, blur_sigma_eff_raw)."""
    import random
    from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
    from basicsr.utils.img_process_util import filter2D
    from basicsr.utils import img2tensor
    deg = _get_deg()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rgb = np.stack([hr] * 3, -1).astype(np.float32) / 255.
    img_gt = img2tensor([rgb], bgr2rgb=False, float32=True)[0].unsqueeze(0)
    k1, k2, sinc = deg.random_kernels()
    blur_sigma_eff = float(np.sqrt(_kernel_sigma_eff(k1) ** 2 + _kernel_sigma_eff(k2) ** 2))
    img_gt, k1, k2, sinc = (x.to(deg.device) for x in [img_gt, k1, k2, sinc])
    o = deg.opt
    with torch.no_grad():
        oh, ow = img_gt.shape[2:]
        out = filter2D(img_gt, k1)
        ud = random.choices(['up', 'down', 'keep'], o['resize_prob'])[0]
        sc = (np.random.uniform(1, o['resize_range'][1]) if ud == 'up' else
              np.random.uniform(o['resize_range'][0], 1) if ud == 'down' else 1)
        out = F.interpolate(out, scale_factor=sc, mode=random.choice(['area', 'bilinear', 'bicubic']))
        if np.random.uniform() < o['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(out, sigma_range=o['noise_range'], clip=True, rounds=False, gray_prob=o['gray_noise_prob'])
        else:
            out = random_add_poisson_noise_pt(out, scale_range=o['poisson_scale_range'], gray_prob=o['gray_noise_prob'], clip=True, rounds=False)
        if not drop_jpeg:
            jp = out.new_zeros(out.size(0)).uniform_(*o['jpeg_range']); out = deg.jpeger(torch.clamp(out, 0, 1), quality=jp)
        if np.random.uniform() < o['second_blur_prob']:
            out = filter2D(out, k2)
        ud = random.choices(['up', 'down', 'keep'], o['resize_prob2'])[0]
        sc = (np.random.uniform(1, o['resize_range2'][1]) if ud == 'up' else
              np.random.uniform(o['resize_range2'][0], 1) if ud == 'down' else 1)
        out = F.interpolate(out, size=(int(oh / scale_final * sc), int(ow / scale_final * sc)), mode=random.choice(['area', 'bilinear', 'bicubic']))
        if np.random.uniform() < o['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(out, sigma_range=o['noise_range2'], clip=True, rounds=False, gray_prob=o['gray_noise_prob2'])
        else:
            out = random_add_poisson_noise_pt(out, scale_range=o['poisson_scale_range2'], gray_prob=o['gray_noise_prob2'], clip=True, rounds=False)
        if not drop_sinc:
            out = filter2D(out, sinc)
        if not drop_jpeg:
            jp = out.new_zeros(out.size(0)).uniform_(*o['jpeg_range2']); out = deg.jpeger(torch.clamp(out, 0, 1), quality=jp)
        out = F.interpolate(out, size=(oh, ow), mode=random.choice(['area', 'bilinear', 'bicubic']))
        lq = (out.detach().squeeze(0).numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return lq[:, :, 0], blur_sigma_eff


def us_physics_labeled(lr, rng, p):
    """Verbatim port of us_physics(), returning the ACTUALLY-applied attenuation ratio
    (1.0, i.e. no-op, if the atten_prob gate didn't fire this call) alongside the output."""
    H, W = lr.shape
    fg = lr > 8
    if not fg.any():
        return lr, 1.0
    f = lr.astype(np.float32) / 255.

    applied_atten = 1.0
    if rng.random() < p['atten_prob']:
        applied_atten = p['atten']
        ramp = np.linspace(1.0, p['atten'], H)[:, None] * np.ones((1, W), np.float32)
        ramp = gaussian_filter(ramp, sigma=(H * 0.04, 0))
        f = f * ramp

    g = rng.gamma(shape=p['spk_m'], scale=1.0 / p['spk_m'], size=lr.shape)
    amp = np.sqrt(g)
    amp = gaussian_filter(amp, sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
    amp /= amp.mean()
    f = f * (1.0 + (amp - 1.0) * p['spk_strength'])

    if p.get('add_sigma', 0) > 0:
        nz = gaussian_filter(rng.standard_normal(lr.shape), sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
        nz /= (nz.std() + 1e-9)
        f = f + nz * p['add_sigma'] * fg

    haze = gaussian_filter(rng.standard_normal(lr.shape), sigma=p['haze_sigma'])
    haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
    darkness = np.clip(1.0 - f / p['haze_ref'], 0, 1)
    f = f + p['haze'] * haze * darkness

    out = np.clip(f * 255., 0, 255)
    out[~fg] = lr[~fg]
    return out.astype(np.uint8), applied_atten


def us_physics_labeled_dosed(lr, rng, p, dose):
    """Like us_physics_labeled, but each of the 4 US-physics effects is scaled toward
    identity by its own dose fraction instead of always applying at full strength:
    dose[1]=haze, dose[2]=attenuation, dose[3]=speckle, dose[4]=additive_noise (same
    slot indices as DEGRADATION_NAMES / generate_labeled_lr's label order; dose[0]/
    dose[5] (psf_blur/compression) are unused here, handled by the caller's backbone
    blend instead). dose=1.0 for a slot reproduces us_physics_labeled's original
    behavior for that slot EXACTLY (bit-for-bit, verified); dose=0.0 fully suppresses
    that slot's effect (identity for that term).

    Draws rng in the EXACT SAME sequence as us_physics_labeled (atten gate -> speckle
    gamma/noise -> conditional additive noise -> haze), so the underlying realization's
    "style" (speckle grain pattern, haze texture placement) stays IDENTICAL across
    different doses of the same seed/aux draw -- only each effect's magnitude changes,
    mirroring _apply_single_degradation/_severity_to_params's "replay the same
    realization at a different severity" contract (2026-07-26, user request: extend
    esrgan_physic with continuous severity control so specificity-style partial-dose/
    mismatch supervision can be built on the SAME generator that produced the static
    train/eval data, instead of only the independently-reimplemented _apply_single_
    degradation primitives).
    """
    H, W = lr.shape
    fg = lr > 8
    if not fg.any():
        return lr, 1.0
    f = lr.astype(np.float32) / 255.
    dose_haze, dose_atten, dose_speckle, dose_noise = dose[1], dose[2], dose[3], dose[4]

    applied_atten = 1.0
    if rng.random() < p['atten_prob']:
        full_atten = p['atten']
        applied_atten = 1.0 - dose_atten * (1.0 - full_atten)
        ramp = np.linspace(1.0, applied_atten, H)[:, None] * np.ones((1, W), np.float32)
        ramp = gaussian_filter(ramp, sigma=(H * 0.04, 0))
        f = f * ramp

    g = rng.gamma(shape=p['spk_m'], scale=1.0 / p['spk_m'], size=lr.shape)
    amp = np.sqrt(g)
    amp = gaussian_filter(amp, sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
    amp /= amp.mean()
    f = f * (1.0 + (amp - 1.0) * p['spk_strength'] * dose_speckle)

    if p.get('add_sigma', 0) > 0:
        nz = gaussian_filter(rng.standard_normal(lr.shape), sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
        nz /= (nz.std() + 1e-9)
        f = f + nz * p['add_sigma'] * dose_noise * fg

    haze = gaussian_filter(rng.standard_normal(lr.shape), sigma=p['haze_sigma'])
    haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
    darkness = np.clip(1.0 - f / p['haze_ref'], 0, 1)
    f = f + p['haze'] * dose_haze * haze * darkness

    out = np.clip(f * 255., 0, 255)
    out[~fg] = lr[~fg]
    return out.astype(np.uint8), applied_atten


def generate_labeled_lr_dosed(hr, dose, seed=None, R=US):
    """Continuous-severity-control extension of generate_labeled_lr (2026-07-26, user
    request), for merging specificity-style partial-dose/mismatch supervision directly
    into the esrgan_physic curriculum instead of needing a separate synthesis/forward
    pass. hr: uint8 HxW. dose: array-like[6] in [0,1], DEGRADATION_NAMES order
    [psf_blur, haze, attenuation, speckle, additive_noise, compression] -- 1.0 = that
    slot's full severity (matches generate_labeled_lr's default call), 0.0 = that slot's
    effect fully suppressed.

    IMPORTANT GUARANTEE (verified by test_esrgan_physic_dosed_equivalence.py): calling
    this with dose=[1,1,1,1,1,1] and the same seed reproduces generate_labeled_lr's
    output BIT-FOR-BIT -- zero risk of silently shifting the static train/eval data's
    distribution for any caller that always requests full dose.

    psf_blur and compression are NOT independently controllable in this generator (both
    come from the SAME sequential kernel1->resize->noise->kernel2->resize2->noise2
    ESRGAN backbone pass -- see this module's docstring) -- unlike the 4 US-physics
    effects (which have their OWN continuous parameters and are dosed via
    us_physics_labeled_dosed, an exact replay of the same realization at reduced
    magnitude), there is no clean way to re-sample a "50%-as-blurry" kernel or a
    "50%-as-compressed" resize from this pipeline. Instead, psf_blur/compression share
    ONE severity dial (bc_frac = max(dose[0], dose[5])) applied as a PIXEL-SPACE ALPHA
    BLEND between the clean HR and the full-severity backbone output -- an approximation
    (not a true re-synthesis at reduced kernel/resize magnitude), but exact at the
    dose=1.0 (full, bit-identical to today) and dose=0.0 (identity) endpoints, and
    monotonic/reasonable in between.

    Returns (lr_dosed uint8 HxW, applied_label float32[6]) -- applied_label reflects the
    ACTUALLY-dosed severity that was rendered (not the pre-dose full severity), so a
    caller building a restoration TARGET from this label gets a self-consistent
    (image, label) pair.
    """
    dose = np.asarray(dose, dtype=np.float64)
    assert dose.shape == (6,), f"dose must be length-6 (DEGRADATION_NAMES order), got shape {dose.shape}"
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    rng = np.random.default_rng(seed)
    scale_final = _sample(rng, R['scale_range'])
    lr_full, blur_sigma_eff = esrgan_backbone_labeled(hr, seed, scale_final, R['drop_jpeg'], R['drop_sinc'])

    bc_frac = float(np.clip(max(dose[0], dose[5]), 0.0, 1.0))
    if bc_frac >= 1.0 - 1e-9:
        lr = lr_full
    elif bc_frac <= 1e-9:
        lr = hr.copy()
    else:
        lr = (bc_frac * lr_full.astype(np.float32)
              + (1.0 - bc_frac) * hr.astype(np.float32)).clip(0, 255).astype(np.uint8)

    p = sample_us_params(rng, R)
    out, applied_atten = us_physics_labeled_dosed(lr, rng, p, dose)
    out[hr <= 8] = 0

    applied_label = np.array([
        bc_frac * np.clip(blur_sigma_eff / _BLUR_SIGMA_EFF_CEIL, 0.0, 1.0),
        dose[1] * np.clip(p['haze'] / R['haze'][1], 0.0, 1.0),
        np.clip((R['atten'][1] - applied_atten) / (R['atten'][1] - R['atten'][0]), 0.0, 1.0),
        dose[3] * np.clip((p['spk_strength'] - R['spk_strength'][0]) / (R['spk_strength'][1] - R['spk_strength'][0]), 0.0, 1.0),
        dose[4] * np.clip(p['add_sigma'] / R['add_sigma'][1], 0.0, 1.0),
        bc_frac * np.clip((scale_final - R['scale_range'][0]) / (R['scale_range'][1] - R['scale_range'][0]), 0.0, 1.0),
    ], dtype=np.float32)
    return out, applied_label


def _apply_atten(f, rng, p, H, W):
    applied_atten = 1.0
    if rng.random() < p['atten_prob']:
        applied_atten = p['atten']
        ramp = np.linspace(1.0, p['atten'], H)[:, None] * np.ones((1, W), np.float32)
        ramp = gaussian_filter(ramp, sigma=(H * 0.04, 0))
        f = f * ramp
    return f, applied_atten


def _apply_speckle(f, rng, p, shape):
    g = rng.gamma(shape=p['spk_m'], scale=1.0 / p['spk_m'], size=shape)
    amp = np.sqrt(g)
    amp = gaussian_filter(amp, sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
    amp /= amp.mean()
    f = f * (1.0 + (amp - 1.0) * p['spk_strength'])
    return f


def _apply_add_noise(f, rng, p, shape, fg):
    if p.get('add_sigma', 0) > 0:
        nz = gaussian_filter(rng.standard_normal(shape), sigma=(p['spk_sigma_ax'], p['spk_sigma_lat']))
        nz /= (nz.std() + 1e-9)
        f = f + nz * p['add_sigma'] * fg
    return f


def _apply_haze(f, rng, p, shape):
    haze = gaussian_filter(rng.standard_normal(shape), sigma=p['haze_sigma'])
    haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
    darkness = np.clip(1.0 - f / p['haze_ref'], 0, 1)
    f = f + p['haze'] * haze * darkness
    return f


def us_physics_labeled_random_order(lr, rng, p):
    """Random ORDER + random QUANTITY variant of us_physics_labeled (2026-08-09, user
    request): instead of always composing all 4 US-physics effects in the fixed
    atten->speckle->noise->haze order (only attenuation independently gated, via
    atten_prob), each call independently samples (a) how many of the 4 effects fire
    (1..4, uniform over the count), (b) which subset, and (c) the order they're
    composed in. haze's darkness-dependent term and attenuation's multiplicative ramp
    both read the CURRENT f, so composing them in a different order genuinely changes
    the rendered result, not just bookkeeping -- this is the point (mimics different
    real acquisition-pipeline orderings), not an approximation of the fixed-order path.
    attenuation still additionally respects atten_prob on top of being selected into the
    subset (so it can be "selected but a no-op this draw", same semantics as before).

    Returns (out uint8 HxW, applied_atten float, applied_ops list[str] subset actually
    composed, in the order they were applied) -- applied_ops lets the caller zero out
    the label for any of {haze, speckle, additive_noise} that wasn't selected this call
    (attenuation's applied_atten already self-reports via the 1.0 no-op sentinel).
    """
    H, W = lr.shape
    fg = lr > 8
    if not fg.any():
        return lr, 1.0, []
    f = lr.astype(np.float32) / 255.

    ops = ['attenuation', 'speckle', 'additive_noise', 'haze']
    n_apply = int(rng.integers(1, len(ops) + 1))
    order = rng.permutation(ops)[:n_apply].tolist()

    applied_atten = 1.0
    for op in order:
        if op == 'attenuation':
            f, applied_atten = _apply_atten(f, rng, p, H, W)
        elif op == 'speckle':
            f = _apply_speckle(f, rng, p, lr.shape)
        elif op == 'additive_noise':
            f = _apply_add_noise(f, rng, p, lr.shape, fg)
        elif op == 'haze':
            f = _apply_haze(f, rng, p, lr.shape)

    out = np.clip(f * 255., 0, 255)
    out[~fg] = lr[~fg]
    return out.astype(np.uint8), applied_atten, order


def generate_labeled_lr_random_order(hr, seed=None, R=US):
    """Random-order/quantity/severity variant of generate_labeled_lr (2026-08-09, user
    request: "both phase 1 and phase 2, online degradation should have randomized
    order, quantity, and severity"). Severity was already randomized per-image (every
    US-physics param and every ESRGAN-backbone kernel/resize/noise choice in R is
    resampled fresh each call) -- what this adds is randomizing WHICH of the 4
    US-physics effects (haze, attenuation, speckle, additive_noise) fire this call and
    in WHAT ORDER they're composed, via us_physics_labeled_random_order.

    The ESRGAN-backbone stage (psf_blur + compression: kernel1->resize->noise->
    [kernel2]->resize2->noise2->final-resize) keeps its OWN fixed internal step order --
    generate_lr_esrgan_physic.py's docstring already documents that other backbone
    orderings introduce "twisted lines", and psf_blur/compression aren't independently
    controllable knobs to begin with (see this module's top docstring: both come from
    the SAME sequential kernel pass) -- so backbone-order randomization isn't
    structurally meaningful here, only the 4 independently-parameterized US-physics
    effects are.

    hr: uint8 HxW. Returns (lr uint8 HxW, label float32[6] in DEGRADATION_NAMES order
    [psf_blur, haze, attenuation, speckle, additive_noise, compression]) -- label for
    any US-physics effect NOT selected this call is 0.0 (identity), matching
    generate_labeled_lr_dosed's "applied severity, not requested severity" contract.
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    rng = np.random.default_rng(seed)
    scale_final = _sample(rng, R['scale_range'])
    lr, blur_sigma_eff = esrgan_backbone_labeled(hr, seed, scale_final, R['drop_jpeg'], R['drop_sinc'])
    p = sample_us_params(rng, R)
    out, applied_atten, applied_ops = us_physics_labeled_random_order(lr, rng, p)
    out[hr <= 8] = 0

    label = np.array([
        np.clip(blur_sigma_eff / _BLUR_SIGMA_EFF_CEIL, 0.0, 1.0),
        np.clip(p['haze'] / R['haze'][1], 0.0, 1.0) if 'haze' in applied_ops else 0.0,
        np.clip((R['atten'][1] - applied_atten) / (R['atten'][1] - R['atten'][0]), 0.0, 1.0),
        np.clip((p['spk_strength'] - R['spk_strength'][0]) / (R['spk_strength'][1] - R['spk_strength'][0]), 0.0, 1.0) if 'speckle' in applied_ops else 0.0,
        np.clip(p['add_sigma'] / R['add_sigma'][1], 0.0, 1.0) if 'additive_noise' in applied_ops else 0.0,
        np.clip((scale_final - R['scale_range'][0]) / (R['scale_range'][1] - R['scale_range'][0]), 0.0, 1.0),
    ], dtype=np.float32)
    return out, label


def generate_labeled_lr(hr, seed=None, R=US):
    """Top-level: hr uint8 HxW -> (lr uint8 HxW, label float32[6] in DEGRADATION_NAMES
    order [psf_blur, haze, attenuation, speckle, additive_noise, compression], each in
    [0, 1]). seed=None draws a fresh random seed (true online/per-step generation,
    unlike the static dataset's one-fixed-seed-per-image).
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    rng = np.random.default_rng(seed)
    scale_final = _sample(rng, R['scale_range'])
    lr, blur_sigma_eff = esrgan_backbone_labeled(hr, seed, scale_final, R['drop_jpeg'], R['drop_sinc'])
    p = sample_us_params(rng, R)
    out, applied_atten = us_physics_labeled(lr, rng, p)
    out[hr <= 8] = 0

    label = np.array([
        np.clip(blur_sigma_eff / _BLUR_SIGMA_EFF_CEIL, 0.0, 1.0),                          # psf_blur
        np.clip(p['haze'] / R['haze'][1], 0.0, 1.0),                                        # haze
        np.clip((R['atten'][1] - applied_atten) / (R['atten'][1] - R['atten'][0]), 0.0, 1.0),  # attenuation
        np.clip((p['spk_strength'] - R['spk_strength'][0]) / (R['spk_strength'][1] - R['spk_strength'][0]), 0.0, 1.0),  # speckle
        np.clip(p['add_sigma'] / R['add_sigma'][1], 0.0, 1.0),                              # additive_noise
        np.clip((scale_final - R['scale_range'][0]) / (R['scale_range'][1] - R['scale_range'][0]), 0.0, 1.0),  # compression
    ], dtype=np.float32)
    return out, label
