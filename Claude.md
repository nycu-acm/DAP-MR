# DAP-MR × PiSA-SR — Kiến trúc & tích hợp

File này giải thích **DAP-MR (LGDP — Language-Grounded Degradation Prompting)** được
ghép vào codebase **PiSA-SR** gốc như thế nào: phần nào là của PiSA-SR, phần nào là
DAP-MR thêm mới, và thêm vào chỗ nào trong code. Đọc file này trước khi sửa
`pisasr.py`/`train_pisasr.py`. Hướng dẫn chạy train/inference cụ thể xem `README.md`.

(Đây không phải log lịch sử thí nghiệm — nội dung đó nằm trong
`experiments/<tên>/training.log` của từng run. File này chỉ mô tả kiến trúc hiện tại.)

---

## 1. PiSA-SR gốc là gì (phần nền, giữ nguyên không đổi)

PiSA-SR là mô hình super-resolution one-step, distill từ SD-2.1-base:

- 1 UNet (trọng số gốc đóng băng) + 2 LoRA adapter chồng lên: **pix-LoRA** (phục hồi
  mức pixel) và **sem-LoRA** (phục hồi mức ngữ nghĩa), rank 4 mỗi cái —
  `initialize_unet()` (`pisasr.py:529`)
- Khử nhiễu **1 bước duy nhất** (`timesteps1 = [1]`), không có sampling loop nhiều bước
- Loss gốc: L2 + LPIPS + CSD (`CSDLoss`, `pisasr.py:574` — CSD là distillation loss so
  với chính SD gốc qua `--pretrained_model_path_csd`)
- Wavelet/AdaIN color-fix sau khi decode (`src/my_utils/wavelet_color_fix.py`)
- VAE/UNet là bản diffusers có sửa để hỗ trợ tiled encode/decode, tiết kiệm VRAM
  (`src/models/autoencoder_kl.py`, `src/models/unet_2d_condition.py`,
  `src/my_utils/vaehook.py`)

Toàn bộ phần này **không phải đóng góp của DAP-MR** — nguyên bản từ
[PiSA-SR](https://github.com/csslc/PiSA-SR) (CVPR 2025).

---

## 2. DAP-MR / LGDP thêm gì

**Ý tưởng cốt lõi**: thay vì dùng 1 prompt văn bản cố định (hoặc caption RAM chung
chung) cho mọi ảnh, cho một **router nhỏ nhìn ảnh LR** để đoán mức độ của 6 loại suy
giảm đặc trưng ảnh siêu âm, rồi **dùng ngay dự đoán đó để dựng prompt** đưa vào UNet —
không tốn thêm lần forward UNet nào (router chỉ là vài phép nhân ma trận nhỏ trước khi
gọi UNet, vẫn đúng 1 lần như PiSA-SR gốc).

6 loại suy giảm (`DEGRADATION_NAMES`, `pisasr.py:25`):
`psf_blur, haze, attenuation, speckle, additive_noise, compression`

### 2.1 Router — đoán cường độ suy giảm từ ảnh LR

- Input: **latent VAE** của ảnh LR (không phải ảnh RGB thô)
- 2 kiến trúc, chọn qua `--router_arch`:
  - `base` (mặc định): unfold patch 3×3 → `vis_projector` → `_structured_spatial_pool`
    (`pisasr.py:179`) → MLP `DegradationRouter` (`pisasr.py:462`)
  - `conv`: `ConvRouterHead` (`pisasr.py:482`) — CNN thẳng trên latent, ít tham số hơn
- Output: `intensities_pred` ∈ [0,1]⁶ — một giá trị cho mỗi degradation
- Gọi qua `PiSASR._compute_intensities` / `encode_degradation_prediction`
  (`pisasr.py:1152`) bên trong `forward()` (`pisasr.py:1177`)

### 2.2 Prompt bank — "restoration tokens" học được

- Mỗi degradation gắn với 1 câu mô tả *trạng thái mong muốn sau phục hồi*
  (`TARGET_STATE_PROMPTS`, `pisasr.py:34` — mặc định qua `--prompt_bank_mode
  target_state`; còn `instruction`, `degradation_noun`, `random` để đối chứng)
- Câu này qua CLIP text encoder (đóng băng), lấy embedding tại vị trí EOS →
  `text_prototypes` (`compute_degradation_text_prototypes`, `pisasr.py:221`)
- `restoration_tokens` (nn.Parameter) khởi tạo = bản sao `text_prototypes`, học tiếp
  cùng LoRA trong Phase 2 — đây là **hướng ngữ nghĩa** (direction) của mỗi degradation
- `restoration_token_scale` (nn.Parameter, qua softplus) là **độ lớn** (magnitude) học
  riêng cho từng degradation — tách direction/magnitude để gradient sạch hơn

### 2.3 Ghép prompt — thứ UNet thực sự nhìn thấy

`_compose_lgdp_prompt` (`pisasr.py:981`, bản inference tương ứng ở `pisasr.py:2906`)
dựng **toàn bộ** prompt của UNet:

```
prompt = [BOS] + [6 restoration_tokens × router_intensity × scale]  (+ domain_prompt tuỳ chọn, mặc định tắt)
```

Không có prompt nền/carrier nào khác cộng vào — đây là khác biệt lớn nhất so với
PiSA-SR gốc (dùng caption cố định). File còn giữ 3 biến thể `--lgdp_injection_mode`
khác chỉ để so sánh/ablation, **không khuyến nghị dùng**:
- `text_ordinal` / `text_numeric`: verbalize cường độ thành câu rồi encode lại — không
  differentiable, cắt gradient tới router
- `ram_tag`: dùng caption RAM thật thay cho router (tắt hẳn cơ chế DAP-MR)

Mặc định và khuyến nghị: `direct_feed`.

### 2.4 Nguồn suy giảm tổng hợp (synthetic degradation)

`_apply_single_degradation` (`pisasr.py:1531`) hiện thực 6 hàm suy giảm dựa trên vật lý
ảnh siêu âm (PSF blur, shadow/attenuation, haze, speckle qua FFT, additive noise, JPEG +
downsample). Hàm này dùng ở **3 chỗ**:

1. Sinh nhãn giả để pretrain router — Phase 1, `compute_router_loss` (`pisasr.py:1731`)
2. Sinh cặp ảnh counterfactual (cùng nội dung, khác mức suy giảm) để giám sát thêm cho
   router — `compute_counterfactual_router_loss` (`pisasr.py:2138`)
3. Sinh batch LR "online" ngay trong training loop thay vì đọc LR có sẵn từ đĩa —
   `_synthesize_online_esrgan_batch` / `_synthesize_online_esrgan_dosed_batch`
   (`pisasr.py:2261` / `2319`, bật qua `--online_degrade_source esrgan_physic`)

### 2.5 Loss phụ trợ cho router/prompt (tuỳ chọn, mặc định tắt = 0)

Tất cả **cộng thêm** vào loss gốc của PiSA-SR (L2+LPIPS+CSD), mỗi loss backward riêng
(tránh lỗi DDP "mark variable ready twice" khi chia sẻ `lora_params`):

- `--lambda_synth_sr` (xác suất `synth_sr_ratio`): SR loss trên LR tổng hợp có **nhãn
  suy giảm chính xác** (không qua router đoán) — cho LoRA gradient trực tiếp đúng nhãn
- `--lambda_specificity` (+ `--lambda_lpips_specificity`, xác suất
  `specificity_batch_ratio`, hoặc cửa sổ `specificity_window_start/_end`): dạy "chỉ phục
  hồi đúng slot được yêu cầu, giữ nguyên phần còn lại" (controllable restoration) — có
  thể **thay thế** forward chính của step đó (`--specificity_replace_main`) thay vì
  cộng thêm
- `--lambda_delta_response`, `--lambda_dose_response`: dạy độ mạnh phục hồi tỉ lệ đúng
  theo cường độ suy giảm yêu cầu (dose-response monotonicity)
- `--domain_randomization` (+ `_strength/_prob/_anneal_frac`): photometric augmentation
  trên cả LR/HR — không phải loss, nhưng nên luôn bật; đóng góp lớn nhất cho khả năng
  tổng quát ngoài phân phối (OOD)

---

## 3. Vòng lặp huấn luyện 2 pha (`train_pisasr.py`)

```
Phase 1 — Router Pre-train (--router_pretrain_steps, mặc định 1000 step)
  Chỉ train router (+ CF loss nếu --lambda_cf_router > 0); LoRA bị đóng băng.
  Sau Phase 1: router bị đóng băng (unwrapped.set_router_eval()).
                              |
                              v
Phase 2 — Joint train (đến --max_train_steps, vd 8500)
  Train LoRA (pix trước --pix_steps step, rồi sem) qua loss SR chính:
    forward()                    — LR thật đọc từ đĩa (mặc định)
    forward_curriculum()         — LR tổng hợp on-the-fly, multi-degradation
                                    (--use_mixed_curriculum)
    forward_curriculum_dosed()   — biến thể "dosed" (severity liên tục thay vì on/off)
  Mỗi step, các loss phụ ở mục 2.5 có thể fire (theo xác suất/cửa sổ riêng), mỗi cái
  backward tách biệt, TRƯỚC forward chính (có thể thay luôn forward chính nếu
  --specificity_replace_main hoặc trong cửa sổ specificity).
  --router_finetune_steps step CUỐI (nếu > 0): mở lại router, train chung với LoRA qua
  chính loss SR thật — bù lệch phân phối giữa synthetic (Phase 1) và LR thật.
```

---

## 4. Bảng cờ (flags) quan trọng theo nhóm

| Nhóm | Cờ | Ghi chú |
|---|---|---|
| Kiến trúc router | `--router_arch` | `base` hoặc `conv` |
| Prompt bank | `--prompt_bank_mode` | `target_state` (khuyến nghị), `instruction`, `degradation_noun`, `random` |
| Token scheme | `--lgdp_token_scheme` | `full6` (mặc định) hoặc `merged3` (gộp 6→3 nhóm) |
| Cách ghép prompt | `--lgdp_injection_mode` | `direct_feed` (khuyến nghị); `text_ordinal/numeric`, `ram_tag` chỉ để ablation |
| Độ dài prompt | `--lgdp_prompt_length` | `7` (mặc định) hoặc `77` |
| Nguồn LR | `--online_degrade_source` | `esrgan_physic` (khuyến nghị) hoặc `curriculum` |
| Curriculum | `--use_mixed_curriculum` | bật `forward_curriculum` thay vì `forward()` |
| Router loss phụ | `--lambda_cf_router`, `--lambda_cf_contrast` | counterfactual supervision, chỉ ở Phase 1 |
| Controllable | `--lambda_specificity`, `--lambda_lpips_specificity`, `--specificity_replace_main` | chỉ phục hồi đúng slot yêu cầu |
| Dose-response | `--lambda_dose_response`, `--lambda_delta_response` | độ mạnh phục hồi tỉ lệ đúng cường độ |
| Domain-rand | `--domain_randomization`, `--domain_rand_strength/_prob/_anneal_frac` | augmentation, nên luôn bật |
| Hai pha | `--router_pretrain_steps`, `--router_finetune_steps` | xem mục 3 |

Danh sách đầy đủ + docstring chi tiết cho từng cờ: `src/my_utils/training_utils.py`
(hàm `parse_args`).

---

## 5. Vị trí code (tra cứu nhanh)

| Thành phần | File | Vị trí |
|---|---|---|
| Model (train) | `pisasr.py` | `class PiSASR` (664+) |
| Model (inference) | `pisasr.py` | `class PiSASR_eval` (2629+) — mirror `PiSASR`, không có phần tính loss |
| Router | `pisasr.py` | `DegradationRouter` (462), `ConvRouterHead` (482) |
| Prompt bank / text prototypes | `pisasr.py` | bank text (25-72), `compute_degradation_text_prototypes` (221) |
| Ghép prompt | `pisasr.py` | `_compose_lgdp_prompt` (981 train / 2906 eval) |
| Suy giảm tổng hợp | `pisasr.py` | `_apply_single_degradation` (1531) |
| Router loss (Phase 1 + CF) | `pisasr.py` | `compute_router_loss` (1731), `compute_counterfactual_router_loss` (2138) |
| Curriculum forward | `pisasr.py` | `forward_curriculum` (1258), `forward_curriculum_dosed` (2470) |
| Vòng lặp train | `train_pisasr.py` | `main()` (192); Phase 1 ≈ 356-431; Phase 2 ≈ 433 trở đi |
| Toàn bộ cờ CLI | `src/my_utils/training_utils.py` | `parse_args()` |
| RAM captioning (tuỳ chọn) | `ram/` | vendored, chỉ dùng khi `--lgdp_injection_mode ram_tag` |

---

## 6. Chạy train/inference

Xem `README.md` để biết lệnh chạy cụ thể, chuẩn bị data, và evaluate metrics.
`scripts/train/*.sh` / `scripts/test/*.sh` là các cấu hình thí nghiệm cụ thể (mỗi file
= 1 tổ hợp cờ đã chạy thật) — nên copy file gần nhất với nhu cầu rồi sửa, thay vì viết
lệnh từ đầu.
