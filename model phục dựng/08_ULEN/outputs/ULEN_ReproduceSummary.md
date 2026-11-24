# ULEN — Reproduce Summary
**Paper:** An ultra-lightweight efficient network for image-based plant disease and pest infection detection
**Tác giả:** Beibei Wang, Chenxiao Zhang, Yanyan Li, Chunxia Cao, Daye Huang, Yan Gong
**Nguồn:** Precision Agriculture (2023) 24:1836–1861 — DOI: 10.1007/s11119-023-10020-0
**Ngày phục dựng:** 2026-06-18
**Notebook xuất:** `papers/08 - An ultra-lightweight efficient network/outputs/ULEN_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | An ultra-lightweight efficient network for image-based plant disease and pest infection detection |
| Tác giả | Beibei Wang, Chenxiao Zhang, Yanyan Li, Chunxia Cao, Daye Huang, Yan Gong |
| Journal | Precision Agriculture, Vol. 24, pp. 1836–1861 |
| Năm | 2023 |
| DOI | 10.1007/s11119-023-10020-0 |
| Nhóm phân loại | Nhóm C (topology rõ, thiếu số kênh/filters từng block) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `ULEN`
**Loại:** CNN lightweight (depth-wise separable + ShuffleNet-style residual/channel-shuffle + SPP)
**Framework:** PyTorch
**Module đặc biệt:** RDWConv (residual depth-wise conv với channel shuffle, kiểu ShuffleNet v2), Spatial Pyramid Pooling (SPP).

### Bảng kiến trúc (đã build)

| # | Layer / Block | Kernel / Filters | Stride | Output Shape (256² in) | Ghi chú |
|---|--------------|------------------|--------|------------------------|---------|
| 1 | Input | — | — | 3×256×256 | — |
| 2 | ConvBlock3x3 (Conv+BN+ReLU+MaxPool) | 3×3 → C=64 | 1 (conv), 2 (pool) | 64×128×128 | Conv chuẩn (full conv) — first layer |
| 3 | MaxPool ×2 | 2×2 | 2 mỗi lớp | 64×32×32 | Hai lớp MaxPool2d liên tiếp |
| 4 | RDWConv ×3 | 3×3 DWConv | 1 | 64×32×32 | ShuffleNet v2 split; channels không đổi |
| 5 | ConvBlock1x1 (Conv+BN+ReLU) | 1×1 → C₂=128 | 1 | 128×32×32 | Conv chuẩn — last layer |
| 6 | SPP {4×4, 2×2, 1×1} | — | — | 21×128 = 2688 | AdaptiveAvgPool, flatten, concat |
| 7 | Linear | 2688 → num_classes | — | num_classes | FC duy nhất, không dropout |

**RDWConv (stride=1, Fig. 7a):** split C → C/2 (skip) + C/2 (main). Main path: DWConv3×3 → BN → ReLU → Conv1×1 → BN. Concat(skip, main) → Channel Shuffle (groups=2). Channels giữ nguyên.

**Total params (notebook, num_classes=38):** ~116,806 (so với paper 111,758 — xem mục 6)

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper
| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework | PyTorch | Mục "Model training" |
| Optimizer | Adam | Mục "Model training" |
| Loss function | Cross-Entropy | Mục "Model training", Eq. (1) |
| Batch size | 16 | Mục "Model training" |
| Epochs | 20 | Mục "Model training" |
| LR schedule | epoch 1–5 = 1e-3; epoch 6–20 = 1e-4 (MultiStepLR milestones=[5], gamma=0.1) | Mục "Model training" |
| Input size | 256×256 | Mục "The Cassava dataset" |
| Train/Val/Test split | 70 / 20 / 10, stratified per class | Mục "The Cassava dataset" + "Model training" |
| Pretrained | Không — train from scratch | Toàn bài |

### ⚠️ Giá trị giả định (không có trong paper)
| Thông số | Giá trị dùng | Lý do giả định |
|---------|------------|--------------|
| Channels C (block đầu + 3 RDWConv) | 64 | Dò ngược để khớp gần 111,758 params; paper không công bố số kênh |
| Channels C₂ (sau conv 1×1) | 128 | Như trên |
| Conv block đầu stride/padding | stride=1, padding=1 | Giữ spatial trước maxpool — convention chuẩn |
| Pooling trong conv block đầu | MaxPool2d(2,2) | Paper chỉ nói "a pooling layer"; gợi ý người dùng |
| MaxPool ×2 | kernel 2×2, stride 2 mỗi lớp | Gợi ý người dùng |
| Activation conv đầu/cuối | ReLU | Fig.7 chỉ ghi ReLU trong RDWConv; áp dụng đồng nhất |
| Channel Shuffle groups | 2 | Block tách 2 nhánh concat |
| Linear head | 1 FC, không dropout | Gợi ý người dùng |
| Normalization mean/std | ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225] | Paper không nêu |
| Weight decay | Không dùng (Adam default) | Paper không nêu |

---

## 4. DATASET

### 4a. Dataset trong Paper *(tham khảo)*

| Mục | Chi tiết từ Paper |
|-----|---------|
| Dataset gốc | PlantVillage (38 lớp) và Cassava (5 lớp) |
| Input size | 256×256 |
| Split | 70 / 20 / 10 stratified per class |
| Normalization | Không nêu |

### 4b. Dataset Thực Tế *(dataset của người dùng)*

| Mục | Chi tiết |
|-----|---------|
| Đường dẫn Kaggle | `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` |
| Số classes | auto-detect (`NUM_CLASSES = len(class_names)`) |
| Danh sách classes | auto-detect từ `ImageFolder.classes` |
| Train / Val / Test | 70 / 20 / 10 stratified (hoặc dùng folder train/val/test nếu có sẵn) |
| Image size | 256×256 |
| Normalization | ImageNet mean/std |

---

## 5. AUGMENTATION (Scenario 2)

> Paper KHÔNG dùng augmentation. Scenario 2 là yêu cầu workspace (core rule #3).

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | `RandomRotation(30)` | Workspace |
| Zoom / Scale | `RandomResizedCrop(256, scale=(0.8,1.0))` | Workspace |
| Horizontal Flip | `RandomHorizontalFlip(p=0.5)` | Workspace |
| ColorJitter | `brightness=0.2, contrast=0.2, saturation=0.2` | Workspace |

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(tham khảo)*

> ⚠️ Kết quả của tác giả trên dataset gốc — không so sánh trực tiếp với dataset người dùng.

| Dataset (paper) | Precision | Recall | F1 | Parameters |
|-----------------|-----------|--------|-----|-----------|
| PlantVillage (38 lớp) | 0.9813 | 0.9749 | 0.9776 | 111,758 |
| Cassava (5 lớp) | 0.5497 | 0.5322 | 0.5337 | 111,758 |

### So sánh Parameters (build vs paper)
- **Build (C=64, C₂=128, num_classes=38):** ~116,806 params, lệch +5,048 so với 111,758.
- Phần lệch nằm gần như toàn bộ ở lớp Linear cuối (21×128×38 = 102,182), chiếm phần lớn tổng params. Vì paper không công bố số kênh, cấu hình kênh ở đây là **ước lượng**. Notebook auto-detect `NUM_CLASSES` từ dataset người dùng → tổng params sẽ khác cả 111,758 lẫn 116,806 tùy số lớp thực tế.
- Notebook in `torchinfo.summary()` và tự so sánh với 111,758 ở cell 1.10.

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Macro | Recall-Macro | Parameters |
|----------|---------|---------|--------------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization
- [ ] Ablation Study
- [ ] Transfer Learning Comparison
- [ ] Inference Benchmark

(Paper không yêu cầu các mục trên → không thêm.)

---

## 8. OUTPUT FILES

```
papers/08 - An ultra-lightweight efficient network/
├── inputs/
│   ├── paper_content.md
│   └── implement_plan.md
└── outputs/
    ├── ULEN_Experiment.ipynb
    └── ULEN_ReproduceSummary.md
```

Notebook tự xuất khi chạy (vào `/kaggle/working`): `best_model_s1.pth`, `best_model_s2.pth`, `lc_s1.png`, `lc_s2.png`, `cm_s1.png`, `cm_s2.png`, `class_distribution.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION

- **Số kênh là ước lượng (không từ paper).** C=64, C₂=128 chọn để khớp gần 111,758. torchinfo.summary() trong notebook so sánh trực tiếp với con số này.
- **RDWConv = ShuffleNet v2 unit:** concat 2 nhánh + channel shuffle (KHÔNG cộng residual element-wise), đúng theo Fig. 7a. Tên gọi "residual" nhưng cơ chế là concat.
- **Conv đầu/cuối là conv chuẩn**, các RDWConv giữa dùng depth-wise — đúng câu paper "except for the first and last convolutional layer".
- **SPP dùng AdaptiveAvgPool** → in_features Linear cố định = 21×C₂ bất kể spatial size; vẫn resize 256×256 theo paper.
- **LR schedule 2 mức** triển khai bằng `MultiStepLR(milestones=[5], gamma=0.1)` (1e-3 → 1e-4 sau epoch 5).
- **Class weighting là DEVIATION từ paper:** paper chỉ dùng stratified split, không weighting. Notebook auto-bật class_weights khi max/min ≥ 1.5 (core rule #11) và in cảnh báo rõ đây là khác biệt so với paper.
- **Best checkpoint theo val_accuracy** (core rule #10), reload trước khi eval test.
- **GPU memory:** `torch.cuda.empty_cache()` sau Scenario 1; DataLoader `num_workers=2, pin_memory=True` (core rule #13).
