# LDI-NET — Reproduce Summary

**Paper:** A novel plant type, leaf disease and severity identification framework using CNN and transformer with multilabel method
**Tác giả:** Bin Yang, Mingwei Li, Fei Li, Yongbo Wang, Qiaokang Liang, Ruiyuan Zhao, Caihong Li, Jianwu Wang
**Nguồn:** Scientific Reports (Nature), 2024 — DOI: 10.1038/s41598-024-62452-x
**Ngày phục dựng:** 2026-06-19
**Notebook xuất:** `papers/09- LDI-NET — Plant Disease Experiment/outputs/LDINET_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | A novel plant type, leaf disease and severity identification framework using CNN and transformer with multilabel method |
| Tác giả | Yang, Li, Li, Wang, Liang, Zhao, Li, Wang |
| Journal / Conference | Scientific Reports (Nature) |
| Năm | 2024 |
| DOI / Link | 10.1038/s41598-024-62452-x |
| Nhóm phân loại | Nhóm C (có model + nhiều hyperparameter rõ, thiếu một số kích thước nội bộ kiến trúc) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `LDI-NET`
**Loại:** Hybrid CNN + Transformer (Feature Tokenizer → Token Encoder → Classifier Head)
**Framework:** PyTorch
**Module đặc biệt:** Feature Tokenizer Module (Conv patch-embed + Transformer + local Conv), Token Encoder Module với fused-token residual addition.

> **Lưu ý adaptation single-label:** Paper gốc là **multi-label** (12 plant + 19 disease + 3 severity = 34 query embeddings) với một Multi-Label Decoder Module kiểu DETR (learnable query + self-attention + cross-attention). Dataset người dùng là **single-label**, nên Module 3 (decoder multi-label) đã được **thay bằng Global Average Pooling + LayerNorm + Linear classifier**. Module 1 và Module 2 giữ nguyên cấu trúc.

### Bảng kiến trúc (single-label adaptation)

| Layer / Block | Cấu hình | Output Shape | Ghi chú |
|--------------|----------|-------------|---------|
| Input | — | (B, 3, 224, 224) | |
| **FTM** Conv2d patch-embed | k=16, s=16, 3→384 | (B, 384, 14, 14) | patch embedding |
| Flatten + PosEmbed | learnable (1, 196, 384) | (B, 196, 384) | 196 = 14×14 |
| FTM Transformer × 2 | LN→MHSA(6h)→res; LN→MLP(384→1536→384, GELU, drop 0.1)→res | (B, 196, 384) | pre-norm |
| Reshape → Conv2d local | k=3, pad=1, 384→384 | (B, 384, 14, 14)→(B,196,384) | local info |
| **Encoder** Token Encoder × 2 | giống transformer block | (B, 196, 384) | context-rich tokens |
| Fused token | compact + context (residual add) | (B, 196, 384) | Fig 1 circle-plus |
| **Head** GAP | mean over 196 tokens | (B, 384) | thay decoder |
| LayerNorm → Linear | 384 → num_classes | (B, num_classes) | classifier |

**Total params (paper):** không report con số tham số. Param thực tế sẽ in qua `torchinfo.summary()` trong notebook (auto theo num_classes).

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper
| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework | PyTorch | Implementation details (dòng 197) |
| Optimizer | AdamW | dòng 200 |
| Learning rate | 6.25e-6 | dòng 201 |
| LR schedule | StepLR(step_size=4, gamma=0.5) | dòng 201 |
| Batch size | 16 | dòng 199 |
| Epochs | 60 | dòng 198 |
| Input size | 224×224×3 | dòng 185 |
| Patch size | 16×16 (best config) | dòng 86-87, Table 9 |
| Token Encoder blocks | 2 (best config) | Table 10 |

### ✅ Confirmed bởi người dùng (không có trong paper, 2026-06-19)
| Thông số | Giá trị | Lý do |
|---------|--------|-------|
| embed_dim (d_model) | 384 | xác nhận người dùng (chuẩn ViT) |
| n_heads | 6 | 384÷6=64 per head |
| N (FTM transformer blocks) | 2 | xác nhận người dùng |
| MLP hidden dim | 1536 (mlp_ratio=4) | chuẩn Transformer |
| Dropout | 0.1 | xác nhận người dùng |
| Conv2d out_channels | 384 (= embed_dim) | xác nhận người dùng |
| Loss function | CrossEntropyLoss (single-label) | dataset single-label |
| Normalization | ImageNet mean/std | xác nhận người dùng |
| Weight decay (AdamW) | 0.01 | xác nhận người dùng |
| Seed | 42 | convention workspace |
| Class weighting | auto-detect (max/min ≥ 1.5 → bật) | Rule 11 workspace |

### ⚠️ Giá trị giả định còn lại
| Thông số | Giá trị dùng | Lý do giả định |
|---------|------------|--------------|
| Khởi tạo PosEmbed | trunc_normal_(std=0.02) | chuẩn ViT, paper không nêu |
| Pre-norm vs post-norm | pre-norm (LN trước attention) | chuẩn ViT hiện đại; paper Fig 3 vẽ "Norm → MSA → add" gợi ý pre-norm |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ để tham khảo)*

| Mục | Chi tiết từ Paper |
|-----|---------|
| Tên dataset gốc | AI Challenger 2018 (plant disease) |
| Số classes (paper) | 56 categories sau lọc (multi-label: 12 plant + 19 disease + 3 severity) |
| Tổng số ảnh (paper) | Train 24,159 / Val 6,054 / Test 4,324 |
| Train/Val/Test (paper) | Train chia 80/20, validation gốc dùng làm test |
| Input size dùng trong paper | 224×224 |
| Normalization (paper) | Không nêu |

### 4b. Dataset Thực Tế *(dataset của người dùng)*

| Mục | Chi tiết |
|-----|---------|
| Tên dataset | dataset_smartcrop_plit_300 (single-label) |
| Đường dẫn | `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` |
| Số classes | auto-detect từ `train/` (`len(ImageFolder.classes)`) |
| Danh sách classes | auto-detect (in trong cell class detection) |
| Train / Val / Test | dùng trực tiếp 3 folder `train/`, `val/`, `test/` |
| Image size | 224×224 |
| Normalization | ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |

---

## 5. AUGMENTATION (Scenario 2)

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| RandomResizedCrop (Zoom/Scale) | size=224, scale=(0.8, 1.0) | Rule DUAL EXPERIMENT workspace |
| Random Horizontal Flip | mặc định p=0.5 | Paper dòng 186 + Rule workspace |
| Random Rotation | degrees=15 | Rule DUAL EXPERIMENT workspace |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 | Rule DUAL EXPERIMENT workspace |

> Paper gốc chỉ dùng RandomResizedCrop + HorizontalFlip. Theo rule DUAL EXPERIMENT của workspace, Scenario 2 dùng đủ 4 augmentation chuẩn. Val/Test KHÔNG augment.

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ để tham khảo)*

> ⚠️ Kết quả dưới đây là multi-label trên dataset AI Challenger 2018 — không so sánh trực tiếp với single-label dataset người dùng.

| Metric (paper, ps=16, enc=2, dec=2) | Giá trị |
|-------------------------------------|---------|
| Plant accuracy | 99.42% |
| Disease accuracy | 98.68% |
| Severity accuracy | 88.55% |
| PDSA (đồng thời đúng cả 3) | 87.40% |
| Precision / Recall / F1 | 95.28 / 95.29 / 95.28 |

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Score (macro) | Recall (macro) | Parameters |
|----------|---------|---------|--------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization (không có trong paper, không thêm)
- [ ] Ablation Study (paper có Table 9/10 ablation về patch size & block count, nhưng dùng config best — không tái hiện full ablation)
- [ ] Transfer Learning Comparison
- [ ] Inference Benchmark

---

## 8. OUTPUT FILES

```
papers/09- LDI-NET — Plant Disease Experiment/
├── inputs/
│   ├── paper_content.md          ← extracted by pdf-extractor
│   └── implement_plan.md         ← analyzed by paper-brainstorm + confirmed by user
└── outputs/
    ├── LDINET_Experiment.ipynb
    └── LDINET_ReproduceSummary.md
```

Notebook tạo thêm các artifact khi chạy trên Kaggle (`/kaggle/working/`):
`best_model_s1.pth`, `best_model_s2.pth`, `learning_curve_s1.png`, `learning_curve_s2.png`,
`confusion_matrix_s1.png`, `confusion_matrix_s2.png`, `class_distribution.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION

- **Thay đổi lớn nhất so với paper:** Module 3 Multi-Label Decoder (DETR-style query + cross-attention) → thay bằng **GAP + LayerNorm + Linear**, vì dataset người dùng là single-label. Loss đổi từ multi-label 3-branch → 1 `CrossEntropyLoss`.
- **Module 1 và Module 2 giữ nguyên** cấu trúc: Conv patch-embed → pos-embed → N transformer → local conv (compact tokens); token encoder → fused = compact + context (residual add).
- **Fused token shape** được giữ nhất quán: cả compact và context đều (B, 196, 384) nên residual add hợp lệ.
- **Pre-norm transformer block** dùng cho cả FTM và Encoder, `nn.MultiheadAttention(batch_first=True)`.
- **Class weighting** tự bật nếu max/min ≥ 1.5 (Rule 11). Có in rõ trong output.
- **Best checkpoint** lưu theo `val_accuracy`, reload trước khi evaluate trên test (Rule 10). DataParallel-aware save/load qua `model.module`.
- **Multi-GPU** (Kaggle T4×2) tự kích hoạt DataParallel; `torchinfo.summary` và checkpoint dùng `model.module`.
- **Rủi ro mismatch:** vì embed_dim/heads/N/dropout/conv-channels không có trong paper (đã được người dùng xác nhận), param count thực tế không thể so khớp với paper (paper không report param). Đây là phục dựng theo cấu hình chuẩn ViT, không phải khớp tuyệt đối số tham số.
- **Smoke test:** forward pass shape đã được trace thủ công (224/16=14 grid → 196 tokens → GAP → num_classes). Không chạy được local do torch DLL lỗi trong môi trường build, nhưng logic shape nhất quán.
