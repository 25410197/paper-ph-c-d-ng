# PDLC-ViT — Reproduce Summary
**Paper:** A Multitask Learning-Based Vision Transformer for Plant Disease Localization and Classification
**Tác giả:** S. Hemalatha, Jai Jaganath Babu Jayachandran
**Nguồn:** International Journal of Computational Intelligence Systems (2024) 17:188 — DOI: https://doi.org/10.1007/s44196-024-00597-3
**Ngày phục dựng:** 2026-06-19
**Notebook xuất:** `papers/10 - PDLCViT_Experiment/outputs/PDLCViT_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | A Multitask Learning-Based Vision Transformer for Plant Disease Localization and Classification |
| Tác giả | S. Hemalatha, Jai Jaganath Babu Jayachandran |
| Journal / Conference | International Journal of Computational Intelligence Systems |
| Năm | 2024 (Vol. 17:188) |
| DOI / Link | https://doi.org/10.1007/s44196-024-00597-3 |
| Nhóm phân loại | Nhóm C (model có tên + bảng hyperparameters đầy đủ, nhưng thiếu các thông số định lượng cốt lõi của ViT) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `PDLC-ViT` (Plant Disease Localization and Classification ViT)
**Loại:** Vision Transformer hybrid (co-scale + co-attention + cross-attention) trong khung MTL
**Framework gốc:** MATLAB 2023a — **phục dựng bằng PyTorch** (CORE RULE #2)
**Phạm vi phục dựng:** CHỈ classification branch (bỏ segmentation head / localization)

**Module đặc biệt:**
- **Image Splitter** — chia ảnh thành patch không chồng lấn 16×16 (Conv stride=16).
- **Co-scale Mechanism** — bắt phụ thuộc đa tỉ lệ (DIỄN GIẢI: 2 nhánh conv 3×3 + 5×5 → concat → Linear).
- **Co-attention Mechanism** — bắt phụ thuộc liên-patch (DIỄN GIẢI: MHSA trên sequence patches + Add&Norm).
- **Cross-Attention** — 2 nhánh song song (CS→Z1, CA→Z2) mỗi nhánh MHSA→Add&Norm→FFN→Add&Norm, rồi MHCA(Z1, Z2)→Z.
- **Transformer Encoder × 6** — pre-norm MHSA + FFN với Add&Norm, có CLS token + learnable positional encoding.
- **Classification Head** — LN → MHCA(query=CLS token, key/value=Z) → residual → LN → MLP → logits.

### Bảng kiến trúc

| Layer / Block | Cấu hình | Output Shape | Ghi chú |
|--------------|---------|-------------|---------|
| Input | RGB | (B, 3, 224, 224) | — |
| PatchEmbed | Conv 16×16, stride 16 | (B, 196, 384) | Image Splitter |
| CoScale | Conv 3×3 + Conv 5×5 → concat → Linear | (B, 196, 384) | ⚠️ DIỄN GIẢI |
| CoAttention | MHSA(6 heads) + Add&Norm | (B, 196, 384) | ⚠️ DIỄN GIẢI |
| CrossAttention | branch_CS→Z1, branch_CA→Z2, MHCA→Z | (B, 196, 384) | Fig. 3 (Eq. 4,5) |
| + CLS token + Pos enc | learnable | (B, 197, 384) | ViT standard |
| Encoder × 6 | MHSA(6h) + FFN(384→1536→384) + Add&Norm | (B, 197, 384) | Fig. 2 |
| LayerNorm | — | (B, 197, 384) | — |
| ClassificationHead | LN→MHCA(CLS,Z)→res→LN→MLP | (B, NUM_CLASSES) | Fig. 4 (logits, no Sigmoid) |

**Total params (paper):** KHÔNG công bố — chỉ in từ `torchinfo.summary()` trong notebook.

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper (Table 2 — visual-confirmed page 10)
| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework gốc | MATLAB 2023a | Sect. 3.2 |
| Optimizer | Adam | Table 2 / Sect. 5.1 |
| Loss function (classification) | Categorical cross-entropy | Eq. (2), Sect. 4.1 |
| Initial learning rate | 0.001 | Table 2 / Sect. 5.1 |
| LR schedule | StepLR ×0.5 mỗi 15 epochs (step_size=15, gamma=0.5) | Table 2 / Sect. 5.1 |
| Batch size | 16 | Table 2 / Sect. 5.1 |
| Dropout | 0.2 (encoder + decoder) | Table 2 / Sect. 5.1 |
| L2 regularization (weight_decay) | 0.0005 | Table 2 / Sect. 5.1 |
| Early stopping | theo validation loss (threshold 0.001) | Table 2 / Sect. 5.1 |
| Data handling | random shuffle khi train/test | Table 2 |

### ⚠️ Giá trị giả định / do người dùng xác nhận (KHÔNG có trong paper)
| Thông số | Giá trị dùng | Lý do |
|---------|------------|-------|
| Input size | 224×224 | ViT standard; paper không nêu |
| Patch size | 16×16 (→196 patches) | ViT standard; paper không nêu |
| Embed dim | 384 | ViT-Small; paper không nêu |
| Depth (encoder layers) | 6 | ViT-Small; paper không nêu |
| Attention heads | 6 | ViT-Small; paper không nêu |
| MLP dim | 1536 (=4×384) | Tỉ lệ ViT chuẩn; paper không nêu |
| Normalization | ImageNet mean/std | Paper không nêu |
| Max epochs | 100 + early stopping (patience=7, monitor val_loss) | Paper không cố định số epoch |
| Split | 70/15/15 stratified (nếu dataset chưa pre-split) | Table 1 không cho tỉ lệ split rõ |
| Co-scale impl. | 2 conv branches (3×3 + 5×5) → concat → Linear | Paper không có layer spec |
| Co-attention impl. | inter-patch MHSA + Add&Norm | Paper không có layer spec |
| Output activation | Logits (CrossEntropyLoss softmax) | Paper dùng Sigmoid; thay đổi cho multiclass CE |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ để tham khảo)*

| Mục | Chi tiết từ Paper |
|-----|---------|
| Tên dataset gốc (classification) | PlantVillage |
| Số classes (paper) | 38 |
| Tổng số ảnh (paper) | 54,306 |
| Train/Val/Test (paper) | Table 1 cho per-class count, KHÔNG cho tỉ lệ split rõ ràng |
| Dataset localization (out-of-scope) | PlantDoc (2,940 ảnh, 13 species, có mask) |
| Input size dùng trong paper | KHÔNG nêu |
| Normalization (paper) | KHÔNG nêu |

### 4b. Dataset Thực Tế *(dataset của người dùng — auto-detect khi chạy notebook)*

| Mục | Chi tiết |
|-----|---------|
| Đường dẫn | `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` |
| Số classes | *(auto-detect từ ImageFolder.classes)* |
| Danh sách classes | *(in ra trong cell "Dataset Loading")* |
| Tổng số ảnh | *(in ra khi chạy)* |
| Train / Val / Test | tự detect: dùng subfolder nếu có, hoặc stratified 70/15/15 |
| Image size | 224×224 |
| Normalization | ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

---

## 5. AUGMENTATION (Scenario 2)

Paper chỉ nói chung "affine transformation" (Sect. 3.1), không liệt kê range. Scenario 2 dùng bộ 4 augmentation chuẩn của workspace (CORE RULE #3):

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | RandomRotation(15) | Workspace RULE #3 (paper chỉ nói "affine") |
| Zoom / Scale | RandomResizedCrop(224, scale=(0.8, 1.0)) | Workspace RULE #3 |
| Horizontal Flip | RandomHorizontalFlip(p=0.5) | Workspace RULE #3 |
| Brightness | ColorJitter(brightness=0.2) | Workspace RULE #3 |

> Val/Test KHÔNG dùng augmentation — luôn dùng transform baseline.

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ để tham khảo)*

> ⚠️ Kết quả dưới đây là của tác giả trên PlantVillage gốc (MATLAB) — KHÔNG thể so sánh trực tiếp với kết quả phục dựng trên dataset của người dùng.

| Cấu hình (paper) | Accuracy | F1 | Precision | AUC |
|------------------|---------|-----|-----------|-----|
| Type A (binary Healthy/Unhealthy) | 99.97% | 0.9995 | 0.9995 | 0.9993 |
| Multiclass (mean ablation baseline) | 99.37% | 0.9904 | 0.9898 | — |
| Localization (PlantDoc) | Mean Dice 0.9921, Mean IoU 0.9973 | — | — | — |

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Score (macro) | Recall (macro) | Parameters |
|----------|---------|------------------|----------------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization *(paper có Grad-CAM nhưng phục vụ localization — out-of-scope branch này)*
- [ ] Ablation Study *(paper có Table 6 nhưng cần segmentation head — out-of-scope)*
- [ ] Transfer Learning Comparison *(không áp dụng)*
- [ ] Inference Benchmark *(không yêu cầu)*

> Notebook chỉ gồm các phần bắt buộc của workspace (2 scenario + metrics + Excel export). Không thêm cell ngoài phạm vi paper/scope đã xác nhận.

---

## 8. OUTPUT FILES

```
papers/10 - PDLCViT_Experiment/
├── inputs/
│   ├── s44196-024-00597-3.pdf
│   ├── paper_content.md          ← extracted (text + figures)
│   ├── implement_plan.md         ← analyzed + user-confirmed
│   └── figures/                  ← rendered PDF figures (Fig.1-4, tables)
└── outputs/
    ├── PDLCViT_Experiment.ipynb        ← notebook (2 scenario + Excel export)
    └── PDLCViT_ReproduceSummary.md     ← tài liệu này
```

**Artifacts notebook sinh ra khi chạy (trong `/kaggle/working`):**
`best_model_s1.pth`, `best_model_s2.pth`, `class_distribution.png`,
`s1_learning_curve.png`, `s1_confusion_matrix.png`,
`s2_learning_curve.png`, `s2_confusion_matrix.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION

- **Rủi ro mismatch CAO:** Paper không công bố bất kỳ con số định lượng nào của ViT (patch/dim/depth/heads/MLP). Toàn bộ cấu hình kiến trúc là lựa chọn ViT-Small do người dùng xác nhận → KHÔNG thể tái tạo accuracy 99.97% của paper.
- **Co-scale / Co-attention là điểm mơ hồ nhất:** Paper không có pseudo-code hay layer spec. Hai module này được triển khai theo diễn giải hợp lý và đánh dấu `# [INTERPRETED - not in paper]` trong code.
- **MTL → single-task:** Bỏ nhánh segmentation → loss đơn giản hóa còn `L_cls` (CrossEntropyLoss). Classification Head theo Fig.4 nhận MHCA(query, Z); vì không còn segmentation/ground-truth-mask query, dùng CLS token làm query cho MHCA với Z từ cross-attention — đây là điều chỉnh do bỏ MTL.
- **Sigmoid → Logits:** Paper ghi MLP classification dùng sigmoid. Với multiclass + CrossEntropyLoss (softmax bên trong) thì output là logits thô, không dùng sigmoid — khác biệt đã ghi note trong notebook và summary.
- **Class imbalance:** Cell "Dataset Distribution" tự tính phân phối + bật `class_weights` nếu max/min ≥ 1.5 (CORE RULE #11).
- **Multi-GPU Kaggle T4×2:** Tuân thủ CORE RULE #8/#10/#13/#14 — DataParallel, `num_workers=4`, `pin_memory=True`, lưu/load `model.module.state_dict()`, `empty_cache()` giữa 2 scenario, `torchinfo.summary` dùng `model.module`.
- **Params:** Paper không công bố số params → chỉ in từ `torchinfo.summary`, không so sánh.

---

## 10. HƯỚNG DẪN CHẠY TRÊN KAGGLE

1. Tạo Kaggle Notebook mới, bật **GPU T4×2** (Settings → Accelerator).
2. Add dataset: `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` (hoặc chỉnh `DATASET_PATH` trong cell Config cho đúng đường dẫn mount).
3. Upload `PDLCViT_Experiment.ipynb`, chạy tuần tự từ trên xuống (Run All).
4. Cell đầu cài `torchinfo`, `openpyxl`, `Pillow`. Các thư viện còn lại có sẵn trên Kaggle.
5. Kết quả: 2 scenario huấn luyện xong, file `results_summary.xlsx` + các ảnh `.png` xuất ra `/kaggle/working`.
6. Nếu dataset không có sẵn subfolder `train/val/test` → notebook tự split stratified 70/15/15.
