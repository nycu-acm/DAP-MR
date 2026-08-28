#!/bin/bash
# Eval dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken (variant 1/4 of the
# token/prompt-scheme ablation -- direct_feed injection + learnable restoration tokens)
# on Cardiac (test/) + BUSI (full/) at --lambda_pix 0.9 --lambda_sem 1.1 (user request,
# matches the base ratio15_25 recipe's own lambda_pix0.9_sem1.1 sweep point).
WORKDIR="/mnt/HDD2/tuong/LL/PiSA-SR_learnable_layer_final/DAP_MR"
cd "${WORKDIR}"
GPU_ID="${1:-0}"
PY="/home/tuong/.conda/envs/pisasr/bin/python"
DATAROOT="/mnt/HDD2/tuong/LL/data/Real_Ultrasound_esrgan_physic"
MODEL_PATH="/mnt/HDD1/tuong/SR/PiSA-SR/preset/sd-2.1-base"
NAME="dapmr_v2mix_degradnoun_curriculum_dosed_ratio15_25_eostoken"
EXP="${WORKDIR}/experiments/${NAME}"
CKPT="${EXP}/checkpoints/model_8500.pkl"
SUFFIX="lambda_pix0.9_sem1.1"

echo "[$(date '+%H:%M:%S')] === Eval ${NAME} (${SUFFIX}) on GPU ${GPU_ID} ==="
for DOMAIN_SPEC in "Cardiac:test" "BUSI:full"; do
    DOMAIN="${DOMAIN_SPEC%%:*}"
    SPLIT="${DOMAIN_SPEC##*:}"
    LR="${DATAROOT}/${DOMAIN}/${SPLIT}/LR"
    HR="${DATAROOT}/${DOMAIN}/${SPLIT}/HR"
    OUT="${EXP}/${DOMAIN}_${SUFFIX}"
    LOG="${EXP}/test_${DOMAIN}_${SUFFIX}.log"
    mkdir -p "${OUT}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PY} test_pisasr.py \
        --pretrained_model_path "${MODEL_PATH}" \
        --pretrained_path       "${CKPT}" \
        --input_image           "${LR}" \
        --output_dir            "${OUT}" \
        --prompt_bank_mode      degradation_noun \
        --lgdp_prompt_length    77 \
        --lambda_pix            0.9 \
        --lambda_sem            1.1 \
        --process_size          512 --upscale 1 --align_method adain --default --seed 123 > "${LOG}" 2>&1
    echo "[$(date '+%H:%M:%S')] === ${NAME}/${DOMAIN} gen DONE ($(ls "${OUT}" | wc -l) imgs) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PY} test_metrics.py \
        --inp_imgs "${OUT}" --gt_imgs "${HR}" --log "${EXP}" >> "${LOG}" 2>&1
    echo "[$(date '+%H:%M:%S')] === ${NAME}/${DOMAIN} metrics DONE ==="
done
echo "[$(date '+%H:%M:%S')] === ${NAME} (${SUFFIX}): all done ==="
