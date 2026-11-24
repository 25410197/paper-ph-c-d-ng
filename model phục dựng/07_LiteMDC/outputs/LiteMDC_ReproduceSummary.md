# Lite-MDC — Reproduce Summary
**Paper:** Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop
**Tác giả:** Sandesh Bhagat, Manesh Kokare, Vineet Haswani, Praful Hambarde, Trupti Taori, P.H. Ghante, D.K. Patil
**Nguồn:** Smart Agricultural Technology 7 (2024) 100408, Elsevier — DOI: 10.1016/j.atech.2024.100408
**Ngày phục dựng:** 2026-06-18
**Notebook xuất:** `papers/07- Advancing real-time plant disease detection/outputs/LiteMDC_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop |
| Tác giả | Bhagat, Kokare, Haswani, Hambarde, Taori, Ghante, Patil |
| Journal | Smart Agricultural Technology 7 (2024) 100408, Elsevier |
| Năm | 2024 |
| DOI | 10.1016/j.atech.2024.100408 |
| Nhóm phân loại | Nhóm A (đề xuất kiến trúc mới, có bảng layer đầy đủ) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `Lite-MDC`
**Loại:** CNN nhẹ — Multi-kernel Depthwise Separable Convolution (MDsConv)
**Framework:** Keras / TensorFlow (paper Mục 5.2)
**Module đặc biệt:** MDsConv — Channel Split → [DWConv3x3 | DWConv5x5] → Concat → BN → LeakyReLU → Pointwise Conv1x1 → BN → LeakyReLU.

### MDsConv module (paper Mục 4.2 / Algorithm 1 / Fig 7c, 8b)
1. Input H×W×C → Channel Split thành 2 nhóm bằng nhau [C/2, C/2]
2. Nhánh trái: DepthwiseConv2D 3×3 (stride s, padding same)
3. Nhánh phải: DepthwiseConv2D 5×5 (stride s, padding same)
4. Concatenate → BN → LeakyReLU(0.1)
5. Pointwise Conv2D 1×1 (N filters) → BN → LeakyReLU(0.1) → [Dropout nếu có]

Downsampling (stride 2) đặt ở **bước depthwise của MDConv** trong các block 1, 3, 6, 9, 12 — không dùng MaxPooling.

### Bảng kiến trúc (Table 5 — channel theo cột Output shape)

| Block | Stride | Out channels | Output shape | Dropout |
|-------|--------|-------------|-------------|---------|
| Conv0 (3×3, 64) | 1 | 64 | 224×224×64 | — |
| Block 1 (MDConv_1 + PConv_1) | 2 | 64 | 112×112×64 | — |
| Block 2 (MDConv_2 + PConv_2) | 1 | 128 | 112×112×128 | — |
| Block 3 (MDConv_3 + PConv_3) | 2 | 128 | 56×56×128 | — |
| Block 4 (MDConv_4 + PConv_4) | 1 | 256 | 56×56×256 | — |
| Block 5 (MDConv_5 + PConv_5) | 1 | 256 | 56×56×256 | — |
| Block 6 (MDConv_6 + PConv_6) | 2 | 256 | 28×28×256 | — |
| Block 7 (MDConv_7 + PConv_7) | 1 | 512 | 28×28×512 | — |
| Block 8 (MDConv_8 + PConv_8) | 1 | 512 | 28×28×512 | — |
| Block 9 (MDConv_9 + PConv_9) | 2 | 512 | 14×14×512 | **0.1** |
| Block 10 (MDConv_10 + PConv_10) | 1 | 512 | 14×14×512 | **0.2** |
| Block 11 (MDConv_11 + PConv_11) | 1 | 512 | 14×14×512 | **0.3** |
| Block 12 (MDConv_12 + PConv_12) | 2 | 512 | 7×7×512 | **0.4** |
| DwConv_13 (depthwise 7×7, valid) | 1 | 512 | 1×1×512 | — |
| PConv_13 (1×1) | 1 | 512 | 1×1×512 | — |
| Conv_14 (1×1, **no activation**) | 1 | 512 | 1×1×512 | — |
| Global Average Pooling | — | 512 | 512 | — |
| Softmax | — | NUM_CLASSES | NUM_CLASSES | — |

**Total params (đã verify analytically):** **2,288,965** (~2.29M) — Trainable 2,270,021 / Non-trainable 18,944. Khớp với mục tiêu paper ~2.2M.

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper
| Thông số | Giá trị | Vị trí |
|---------|--------|--------|
| Framework | Keras / TensorFlow | Mục 5.2 |
| Optimizer | Adam | Mục 5.2 |
| Loss function | Categorical cross-entropy (dùng sparse_categorical_crossentropy với nhãn int) | Mục 5.2 |
| Batch size | 16 | Mục 5.2 |
| Input size | 224×224×3 | Mục 4.3 / Table 5 |
| LR schedule | ReduceLROnPlateau(monitor=val_loss, factor=0.3, patience=5) | Mục 5.2 |
| Train/Val/Test split | 60/20/20 (chỉ áp dụng nếu cần; dataset người dùng đã pre-split) | Mục 5.2 |
| Activation | LeakyReLU (mọi BN trừ layer cuối) | Mục 4.3 |
| Downsampling | stride 2 thay max pooling (blocks 1,3,6,9,12) | Mục 4.3 |

### ✅ User-confirmed (không có trong paper)
| Thông số | Giá trị dùng | Lý do |
|---------|------------|-------|
| Learning rate khởi đầu | 3e-4 | Người dùng xác nhận |
| Epochs | 100 | Người dùng xác nhận |
| LeakyReLU alpha | 0.1 | Người dùng xác nhận |
| Dropout schedule | 0.1 / 0.2 / 0.3 / 0.4 sau PConv_9/10/11/12 | Người dùng xác nhận (paper chỉ nói "tăng dần 0.05→0.4") |
| Normalization | ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] | Người dùng xác nhận (paper chỉ nói per-channel mean subtraction) |
| Seed | 42 | Convention dự án |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ tham khảo)*
| Mục | Chi tiết |
|-----|---------|
| Tên dataset gốc | Pigeon-pea (novel dataset của paper) |
| Số classes (paper) | 5 |
| Train/Val/Test (paper) | 60 / 20 / 20 |
| Input size | 224×224 |
| Normalization (paper) | Per-channel mean subtraction |

### 4b. Dataset Thực Tế *(dataset người dùng)*
| Mục | Chi tiết |
|-----|---------|
| Đường dẫn | `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` |
| Cấu trúc | Pre-split: train/ val/ test (auto-detect) |
| Số classes | AUTO-DETECT từ thư mục train |
| Image size | 224×224 |
| Normalization | ImageNet mean/std |
| Split | Dùng trực tiếp 3 folder, không random split |

> NUM_CLASSES được auto-detect từ thư mục train; softmax head khớp số class thực tế.

---

## 5. AUGMENTATION (Scenario 2)

Notebook dùng đúng 04 kỹ thuật theo ràng buộc dự án (tham số tham khảo từ paper Mục 5.2):

| Kỹ thuật | Tham số | Keras layer |
|---------|--------|-------------|
| Rotation | ±30° | RandomRotation(30/360) |
| Zoom / Scale | 0.2 | RandomZoom(0.2) |
| Horizontal Flip | có | RandomFlip('horizontal') |
| Brightness | factor 0.4, value_range (0,1) | RandomBrightness |

Augmentation chỉ áp dụng cho tập **train** của Scenario 2. Val/Test luôn dùng baseline transform (Resize + ImageNet normalize).

> Lưu ý: paper còn dùng width/height shift, shear, vertical flip, rotation 360°, brightness 0.1–1.0. Notebook giới hạn ở 04 kỹ thuật theo ràng buộc CLAUDE.md rule 3.

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ tham khảo)*

> ⚠️ Kết quả dưới đây trên dataset gốc của tác giả — không so sánh trực tiếp được.

| Dataset (paper) | Accuracy | Params |
|-----------------|----------|--------|
| Pigeon-pea | 94.14% (Table 9) / 95.01% (Table 7) | 2.2M |
| PlantVillage | 99.78% | 2.2M |
| Cassava | 86.4% | 2.2M |
| Apple leaf | 97.2% | 2.2M |

FPS: 33–34 (NVIDIA P100). FLOPS ~0.61G.

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Macro | Recall-Macro | Parameters |
|----------|---------|---------|--------------|-----------|
| Baseline (No-Aug) | — | — | — | 2,288,965 |
| Augmented | — | — | — | 2,288,965 |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization (không có trong paper → không thêm)
- [ ] Ablation Study (không thêm)
- [ ] Transfer Learning (Lite-MDC train from scratch)
- [ ] Inference Benchmark (không thêm)

Notebook chỉ chứa 2 scenario bắt buộc + Excel export. Không thêm cell ngoài phạm vi paper.

---

## 8. OUTPUT FILES

```
papers/07- Advancing real-time plant disease detection/
├── inputs/
│   ├── 1-s2.0-S2772375524000133-main.pdf
│   ├── paper_content.md
│   ├── implement_plan.md
│   └── figures/
└── outputs/
    ├── LiteMDC_Experiment.ipynb
    └── LiteMDC_ReproduceSummary.md
```

Artifacts sinh khi chạy notebook (trong `/kaggle/working`): `best_model_s1.keras`, `best_model_s2.keras`, `results_summary.xlsx`, các file `.png` (learning curve, confusion matrix, class distribution).

---

## 9. GHI CHÚ IMPLEMENTATION

1. **Channel sequence (nguồn chân lý = cột Output shape của Table 5):** Conv0=64 → 64 → 128 → 128 → 256 → 256 → 256 → 512 → 512 → 512 → 512 → 512 → 512. Có mâu thuẫn nội tại trong cột "Filters size" của Table 5 (vd PConv_2) — đã ưu tiên cột Output shape theo implement_plan ghi chú 5.
2. **Param count verify:** Tính analytic = 2,288,965 (~2.29M), khớp mục tiêu paper ~2.2M. Nếu `model.summary()` trên Kaggle lệch nhiều → rà lại số filter pointwise.
3. **DwConv_13:** depthwise 7×7 padding='valid' trên feature map 7×7 → ra 1×1×512 (đúng paper, giữ để khớp param count, không thay bằng GAP sớm).
4. **Tầng cuối không activation:** Conv_14 có BN nhưng KHÔNG LeakyReLU (paper: "LeakyReLU except the last one").
5. **MDsConv split:** dùng `Lambda` slice kênh thành 2 nửa; DepthwiseConv2D depth_multiplier=1 giữ nguyên C/2 kênh mỗi nhánh. Tất cả tầng có số kênh chẵn (64/128/256/512) → split an toàn.
6. **Loss:** dùng `sparse_categorical_crossentropy` (nhãn int từ image_dataset_from_directory) — tương đương categorical cross-entropy của paper.
7. **Class imbalance:** auto-detect trên tập train; nếu max/min ≥ 1.5 → bật `class_weight` trong `model.fit()` (project rule 11).
8. **Best checkpoint reload:** ModelCheckpoint theo val_accuracy; reload trước test eval cả 2 scenario (project rule 10).
9. **GPU memory:** clear_session + gc.collect giữa 2 scenario (project rule 13).
10. **Khác biệt so với paper:** (a) augmentation rút gọn còn 04 kỹ thuật; (b) normalization ImageNet thay vì mean subtraction; (c) LR/epochs/dropout là user-confirmed. Đây là các điểm có thể ảnh hưởng kết quả so với paper.
