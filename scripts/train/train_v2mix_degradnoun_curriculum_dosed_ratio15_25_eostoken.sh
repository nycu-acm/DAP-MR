#!/bin/bash
# Variant 1/4 of a token/prompt-scheme ablation (2026-08-15, user request), all trained
# with the SAME base recipe as dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25 (see
# that script for the ratio15_25 combined-dose rationale). This variant: LEARNABLE
# restoration tokens ("learnable cho EOS token") -- keeps --lgdp_injection_mode at its
# default direct_feed (EOS-pooled restoration tokens weighted by router intensity), but
# flips --degrad_prompt_learnable false -> true so those tokens are trained jointly with
# LoRA throughout Phase 2 instead of staying frozen at their text-prototype init. The
# other 3 variants swap the injection mode entirely (text_ordinal/text_numeric/ram_tag)
# -- see the sibling _linguistic/_numerical/_ramtag scripts.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

GPU_ID=${1:-0}
PORT=${2:-22690}
cd "$(dirname "$0")/../.."
OUT=experiments/dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken
mkdir -p "${OUT}"
exec > >(tee "${OUT}/training.log") 2>&1
echo "[$(date '+%H:%M:%S')] === dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken (from scratch) 8.5k -> GPU ${GPU_ID} port ${PORT} -> ${OUT} (DAP_new) ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} /home/tuong/.conda/envs/pisasr/bin/accelerate launch \
    --gpu_ids "all" --num_processes=1 --main_process_port ${PORT} \
    train_pisasr.py \
    --pretrained_model_path     "/mnt/HDD1/tuong/SR/PiSA-SR/preset/sd-2.1-base" \
    --pretrained_model_path_csd "/mnt/HDD1/tuong/SR/PiSA-SR/preset/sd-2.1-base" \
    --dataset_txt_paths         "/mnt/HDD2/tuong/LL/data/Real_Ultrasound_esrgan_physic/Cardiac/train.txt" \
    --dataset_test_folder       "/mnt/HDD2/tuong/LL/data/Real_Ultrasound_esrgan_physic/Cardiac/test" \
    --use_mixed_curriculum       true \
    --online_degrade_source      esrgan_physic \
    --curriculum_dose_random     true \
    --curriculum_dose_partial_prob   0.15 \
    --curriculum_dose_mismatch_prob  0.25 \
    --curriculum_dose_min_u      0.1 \
    --curriculum_dose_max_u      0.9 \
    --prompt_bank_mode          degradation_noun \
    --lgdp_injection_mode        direct_feed \
    --lgdp_prompt_length        77 \
    --lgdp_prompt_strength      1.0 \
    --degrad_prompt_learnable   true \
    --lambda_cf_router          0.5 \
    --lambda_cf_contrast        0.0 \
    --lora_rank_unet_pix        4 \
    --lora_rank_unet_sem        4 \
    --pix_steps                 4000 \
    --router_pretrain_steps     3000 \
    --router_finetune_steps     1000 \
    --max_train_steps           8500 \
    --checkpointing_steps       8500 \
    --eval_freq                 9999 \
    --learning_rate             5e-5 \
    --lr_scheduler               constant \
    --train_batch_size          1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --lambda_l2                 1.0 \
    --lambda_lpips              2.0 \
    --lambda_csd                1.0 \
    --cfg_csd                   7.5 \
    --timesteps1                1 \
    --align_method               adain \
    --domain_randomization      true \
    --domain_rand_strength      0.2 \
    --domain_rand_prob          1.0 \
    --domain_rand_anneal_frac   0.5 \
    --null_text_ratio           0.0 \
    --lambda_router_dose_online  1.0 \
    --teacher_forcing_anneal_frac 0.1176 \
    --controllable_min_severity  0.2 \
    --online_clean_prob          0.1 \
    --deg_file_path             params.yml \
    --prob                      0.0 \
    --seed                      123 \
    --output_dir                "${OUT}" \
    --tracker_project_name      PiSASR_dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken

echo "[$(date '+%H:%M:%S')] === dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken (from scratch) 8.5k DONE ==="
