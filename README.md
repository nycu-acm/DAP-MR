# DAP-MR: Degradation-Aware Prompting for Medical (Ultrasound) SR

## Directory structure

```
pisasr.py               # Model: PiSASR / PiSASR_eval, LGDP router + prompt injection,
                         #        synthetic degradation functions, CSD/ranking losses
train_pisasr.py          # Training loop (Phase 1 router pretrain + Phase 2 joint train)
test_pisasr.py            # Inference: LR image(s) -> SR output(s)
test_metrics.py            # Post-inference evaluation: PSNR/SSIM/LPIPS/DISTS vs GT

src/
  models/                 # autoencoder_kl.py, unet_2d_condition.py (diffusers, tiling-hook enabled)
  datasets/                # dataset.py (paired dataset), esrgan_physic_online.py (online
                            #   synthetic LR generator + curriculum), realesrgan.py, params.yml
  my_utils/                # training_utils.py (argparse), vaehook.py (tiled VAE),
                            # wavelet_color_fix.py, devices.py

ram/                      # Vendored RAM (Recognize Anything Model) — used for image
                          #   captioning / DAPE tagging in some prompt-bank modes

scripts/
  get_path.py             # Generates the ground-truth image-path .txt used by --dataset_txt_paths
  train/*.sh               # Training launch scripts (one per experiment configuration)
  test/*.sh                 # Inference + metrics launch scripts
  eval/*.sh                  # Downstream evaluation (e.g. SAMUS segmentation referee) scripts
```

## Installation

```shell
conda create -n pisasr python=3.10
conda activate pisasr
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers the core model stack. `test_metrics.py` additionally
needs `lpips`, `pyiqa`, `opencv-python`, and `scipy`/`timm` (used by
`esrgan_physic_online.py` and the RAM Swin backbone) — install these if missing.

## Data preparation

1. Fill in `folder_path` / `txt_path` in [`scripts/get_path.py`](scripts/get_path.py)
   and run it to produce a `.txt` file listing all HR ground-truth image paths.
2. Your training data folder needs paired `LR`/`HR` subfolders per domain, e.g.
   `data/<Dataset>/<Domain>/{train.txt, test/{LR,HR}}` (see any `scripts/train/*.sh`
   for the exact `--dataset_txt_paths` / `--dataset_test_folder` layout expected).

## Training

Every experiment configuration lives in `scripts/train/*.sh` — copy the one closest
to what you want and change the flags you need (GPU id, port, `OUT=` output dir,
`--tracker_project_name`) rather than hand-building a new command. Example:

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_pisasr.py \
    --pretrained_model_path     preset/models/stable-diffusion-2-1-base \
    --pretrained_model_path_csd preset/models/stable-diffusion-2-1-base \
    --dataset_txt_paths         data/<Dataset>/<Domain>/train.txt \
    --dataset_test_folder       data/<Dataset>/<Domain>/test \
    --online_degrade_source     esrgan_physic \
    --prompt_bank_mode          target_state \
    --lora_rank_unet_pix        4 \
    --lora_rank_unet_sem        4 \
    --router_pretrain_steps     1000 \
    --max_train_steps           8500 \
    --checkpointing_steps       8500 \
    --learning_rate             5e-5 \
    --train_batch_size          1 \
    --gradient_accumulation_steps 1 \
    --enable_xformers_memory_efficient_attention \
    --seed 123 \
    --output_dir experiments/<name>
```

Key LGDP-specific flags (full list + docstrings in `src/my_utils/training_utils.py`;
architecture context for each in [`Claude.md`](Claude.md) §2/§4):
- `--router_arch`: `base` (patch-unfold + MLP) or `conv` (CNN over the VAE latent).
- `--prompt_bank_mode`: which text prompts seed the degradation prototypes
  (`target_state` is recommended/default).
- `--lgdp_injection_mode`: `direct_feed` (recommended/default) builds the whole UNet
  prompt from the router-weighted restoration tokens; `text_ordinal`/`text_numeric`/
  `ram_tag` are ablation-only alternatives.
- `--online_degrade_source`: `esrgan_physic` (recommended) vs `curriculum`; combine
  with `--use_mixed_curriculum` to synthesize LR on-the-fly every step.
- `--lambda_cf_router` / `--lambda_cf_contrast`: counterfactual router supervision
  during Phase 1 (paired same-content, different-degradation views).
- `--lambda_specificity` / `--lambda_lpips_specificity`: teaches the router/prompt to
  restore only the requested degradation slot (controllable restoration).
- `--lambda_dose_response` / `--lambda_delta_response`: teaches restoration strength
  to scale monotonically with requested degradation intensity.
- `--domain_randomization`, `--domain_rand_strength`, `--domain_rand_prob`:
  photometric domain-randomization augmentation — usually worth keeping on.
- `--router_pretrain_steps` / `--router_finetune_steps` / `--max_train_steps`:
  Phase 1 (router-only) vs Phase 2 (joint) step budget — see `Claude.md` §3.

## Inference

```shell
python test_pisasr.py \
    --pretrained_model_path preset/models/stable-diffusion-2-1-base \
    --pretrained_path experiments/<name>/checkpoints/model_8500.pkl \
    --process_size 512 \
    --upscale 1 \
    --input_image data/<Dataset>/<Domain>/test/LR \
    --output_dir experiments/<name>/output \
    --default
```

Add `--lambda_pix`/`--lambda_sem` instead of `--default` to manually control
pixel-level vs semantic-level restoration strength. Add
`--latent_tiled_size/--vae_encoder_tiled_size/--vae_decoder_tiled_size` to reduce
VRAM via tiled VAE/latent processing.

Then evaluate against ground truth:

```shell
python test_metrics.py \
    --inp_imgs experiments/<name>/output \
    --gt_imgs  data/<Dataset>/<Domain>/test/HR \
    --log      experiments/<name>
```

`scripts/test/*.sh` chains both commands together per experiment; `scripts/eval/*.sh`
adds downstream-task evaluation (e.g. SAMUS segmentation Dice) on top of the SR output.

## Citations

This code builds on PiSA-SR. If it helps your work, please cite the original paper:

```
@article{sun2024pisasr,
  title={Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach},
  author={Sun, Lingchen and Wu, Rongyuan and Ma, Zhiyuan and Liu, Shuaizheng and Yi, Qiaosi and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2025}
}
```
