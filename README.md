# DAP-MR: Degradation-Aware Prompt–Magnitude Routing for Ultrasound Image Super-Resolution

> Under Review

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Task](https://img.shields.io/badge/Task-Ultrasound%20Super--Resolution-6f42c1)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)

</div>

---

## 🧪 Abstract

When super-resolution models designed for natural images are used on ultrasound scans, relying on semantic text prompts can lead to unreliable results. The tags these models generate often do not match the actual restoration needs, and specialized medical terms may be misunderstood by general-purpose text encoders. Instead, prompts focused on common image degradations are more directly useful for guiding restoration. However, using a single prompt to describe both the type and severity of image degradation can entangle these factors, making it hard for the model to know how strong the degradation is. To address this limitation, we introduce Degradation-Aware Prompt--Magnitude Routing (DAP-MR). Each degradation type is represented by a dedicated token initialized from the contextualized EOS embedding of a degradation-oriented prompt and subsequently adapted during ultrasound SR training. A separate magnitude router predicts a continuous, input-dependent magnitude that modulates the corresponding token. Thus, language provides a degradation-aware initialization for \emph{what} should be restored, while magnitude routing supplies an explicit signal for \emph{how strongly} it should influence restoration. DAP-MR can be added to both one-step and multi-step diffusion super-resolution frameworks without changing their core structures. Our experiments on synthetic datasets under both in-domain and anatomical-domain-shift settings, together with evaluations on unpaired real-world ultrasound data show that DAP-MR delivers better restoration quality and generalizes more consistently across different domains.

---

## 🏗️ Framework

DAP-MR consists of three main components:

1. **Degradation Prompt Bank**  
   Degradation-specific tokens encode the identity of ultrasound artifacts.

2. **Magnitude Router**  
   A visual router predicts continuous degradation severity from the degraded
   input.

3. **Prompt–Magnitude Routing**  
   Router outputs modulate the corresponding degradation tokens before they
   are passed to the diffusion SR model.

The current implementation is built on **PiSA-SR** and supports training with
synthetic ultrasound degradations and controllable restoration objectives.

<!--
Add the architecture figure when available:

<p align="center">
  <img src="assets/Fig2.png" alt="DAP-MR Architecture" width="900">
</p>
-->

---

## 📁 Directory Structure

```text
DAP-MR/
├── pisasr.py
│   └── PiSASR / PiSASR_eval, degradation router, prompt injection,
│       synthetic degradation functions, CSD and ranking losses
│
├── train_pisasr.py
│   └── Training loop:
│       Phase 1 - router pretraining
│       Phase 2 - joint optimization
│
├── test_pisasr.py
│   └── Inference: LR image(s) → SR output(s)
│
├── test_metrics.py
│   └── Evaluation: PSNR / SSIM / LPIPS / DISTS
│
├── src/
│   ├── models/
│   │   ├── autoencoder_kl.py
│   │   └── unet_2d_condition.py
│   │       # Diffusers modules with tiling support
│   │
│   ├── datasets/
│   │   ├── dataset.py
│   │   ├── esrgan_physic_online.py
│   │   │   # Online synthetic ultrasound degradation + curriculum
│   │   ├── realesrgan.py
│   │   └── params.yml
│   │
│   └── my_utils/
│       ├── training_utils.py
│       ├── vaehook.py
│       ├── wavelet_color_fix.py
│       └── devices.py
│
├── ram/
│   └── Vendored RAM (Recognize Anything Model), used for image
│       captioning / DAPE tagging in selected prompt-bank modes
│
└── scripts/
    ├── get_path.py
    │   └── Generates HR image-path lists for --dataset_txt_paths
    │
    ├── train/
    │   └── Training launch scripts for individual experiments
    │
    ├── test/
    │   └── Inference + metric evaluation scripts
    │
    └── eval/
        └── Downstream evaluation scripts
            (e.g. SAMUS segmentation evaluation)
```

---

## ⚙️ Installation

Create the environment:

```bash
conda create -n pisasr python=3.10
conda activate pisasr

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers the core model stack.

For metric evaluation and some degradation / RAM components, additional
packages may be required:

```bash
pip install lpips pyiqa opencv-python scipy timm
```

A CUDA-enabled GPU is recommended for training and inference.

---

## 📚 Data Preparation

### 1. Generate the HR image-path list

Set `folder_path` and `txt_path` in:

```text
scripts/get_path.py
```

Then run:

```bash
python scripts/get_path.py
```

This produces the `.txt` file used by:

```text
--dataset_txt_paths
```

### 2. Prepare the dataset

A typical dataset layout is:

```text
data/
└── <Dataset>/
    └── <Domain>/
        ├── train.txt
        └── test/
            ├── LR/
            └── HR/
```

The exact paths are specified through:

```text
--dataset_txt_paths
--dataset_test_folder
```

Example experiment configurations are available under:

```text
scripts/train/
```

---

## 🏋️ Training

Training configurations are provided in:

```text
scripts/train/*.sh
```

We recommend copying the configuration closest to the desired experiment and
modifying only the necessary options such as:

- GPU ID
- distributed training port
- output directory
- experiment name
- dataset paths
- training steps
- DAP-MR configuration

rather than constructing a training command from scratch.

### Example

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_pisasr.py \
    --pretrained_model_path preset/models/stable-diffusion-2-1-base \
    --pretrained_model_path_csd preset/models/stable-diffusion-2-1-base \
    --dataset_txt_paths data/<Dataset>/<Domain>/train.txt \
    --dataset_test_folder data/<Dataset>/<Domain>/test \
    --online_degrade_source esrgan_physic \
    --prompt_bank_mode target_state \
    --lora_rank_unet_pix 4 \
    --lora_rank_unet_sem 4 \
    --router_pretrain_steps 3000 \
    --max_train_steps 8500 \
    --checkpointing_steps 8500 \
    --learning_rate 5e-5 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --enable_xformers_memory_efficient_attention \
    --seed 123 \
    --output_dir experiments/<name>
```

---

## 🎛️ DAP-MR Configuration

The complete argument list and documentation are available in:

```text
src/my_utils/training_utils.py
```

> Some implementation arguments retain the legacy `lgdp_` prefix.

### Router Architecture

```text
--router_arch
```

Available options:

- `base`: patch-unfold representation followed by an MLP.
- `conv`: CNN-based router operating on the VAE latent representation.

---

### Prompt Bank

```text
--prompt_bank_mode
```

Controls the textual representations used to initialize the degradation
prototypes.

Recommended configuration:

```text
target_state
```

---

### Prompt Injection

```text
--lgdp_injection_mode
```

Available modes include:

- `direct_feed` — recommended/default; constructs the UNet conditioning from
  router-weighted restoration tokens.
- `text_ordinal` — ordinal text-based degradation conditioning.
- `text_numeric` — numeric severity expressed through text.
- `ram_tag` — RAM/DAPE-tag-based conditioning.

The alternative modes are primarily intended for ablation experiments.

---

### Online Degradation

```text
--online_degrade_source
```

Supported modes include:

```text
esrgan_physic
curriculum
```

`esrgan_physic` is the recommended configuration.

It can be combined with:

```text
--use_mixed_curriculum
```

to generate degraded LR observations on-the-fly during training.

---

### Counterfactual Router Supervision

```text
--lambda_cf_router
--lambda_cf_contrast
```

These objectives supervise the router using paired views with the same image
content but different degradation characteristics during the router
pretraining stage.

---

### Degradation Specificity

```text
--lambda_specificity
--lambda_lpips_specificity
```

These losses encourage the model to respond selectively to the requested
degradation instead of unnecessarily modifying unrelated degradation
components.

---

### Magnitude-Aware Restoration

```text
--lambda_dose_response
--lambda_delta_response
```

These objectives encourage restoration strength to change consistently with
the requested degradation magnitude.

---

### Domain Randomization

```text
--domain_randomization
--domain_rand_strength
--domain_rand_prob
```

Photometric domain randomization can be applied during training to improve
robustness to acquisition-domain variation.

---

### Training Stages

```text
--router_pretrain_steps
--router_finetune_steps
--max_train_steps
```

Training is divided into:

**Phase 1 — Router Pretraining**

The degradation router is optimized before joint SR training.

**Phase 2 — Joint Training**

The router, degradation-aware conditioning, and trainable SR components are
optimized jointly.

---

## 🔍 Inference

Run inference with:

```bash
python test_pisasr.py \
    --pretrained_model_path preset/models/stable-diffusion-2-1-base \
    --pretrained_path experiments/<name>/checkpoints/model_8500.pkl \
    --process_size 512 \
    --upscale 1 \
    --input_image data/<Dataset>/<Domain>/test/LR \
    --output_dir experiments/<name>/output \
    --default
```

By default, the predefined PiSA-SR restoration configuration is used.

Pixel-level and semantic-level restoration strengths can also be controlled
manually using:

```text
--lambda_pix
--lambda_sem
```

instead of:

```text
--default
```

For high-resolution inputs or limited GPU memory, tiled processing is
available through:

```text
--latent_tiled_size
--vae_encoder_tiled_size
--vae_decoder_tiled_size
```

---

## ✅ Evaluation

After inference, evaluate the SR images against the corresponding HR ground
truth:

```bash
python test_metrics.py \
    --inp_imgs experiments/<name>/output \
    --gt_imgs data/<Dataset>/<Domain>/test/HR \
    --log experiments/<name>
```

The current evaluation pipeline includes:

- **PSNR**
- **SSIM**
- **LPIPS**
- **DISTS**

Experiment-specific inference and evaluation pipelines are provided in:

```text
scripts/test/*.sh
```

These scripts typically run inference and quantitative evaluation
sequentially.

Additional downstream evaluation is available in:

```text
scripts/eval/*.sh
```

including ultrasound segmentation evaluation using models such as SAMUS.

---

## 📄 Paper

The DAP-MR manuscript is currently **under review**.

Paper, supplementary material, pretrained checkpoints, and citation
information will be released after publication.

---

## 🙏 Acknowledgements

This repository builds upon **PiSA-SR: Pixel-level and Semantic-level
Adjustable Super-resolution**.

If you use the underlying PiSA-SR implementation, please also cite the original
work:

```bibtex
@inproceedings{sun2025pisasr,
  title     = {Pixel-level and Semantic-level Adjustable Super-resolution:
               A Dual-LoRA Approach},
  author    = {Sun, Lingchen and Wu, Rongyuan and Ma, Zhiyuan and
               Liu, Shuaizheng and Yi, Qiaosi and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition},
  year      = {2025}
}
```

---

## 📢 Updates

- **2026** — DAP-MR repository initialized.
- **2026** — Manuscript under review.
- Code, checkpoints, paper, and supplementary material will be updated
  following publication.