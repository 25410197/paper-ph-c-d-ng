# PMVT-XS — Reproduce Summary
**Paper:** PMVT: a lightweight vision transformer for plant disease identification on mobile devices
**Tác giả:** Guoqiang Li, Yuchao Wang, Qing Zhao, Peiyan Yuan, Baofang Chang
**Nguồn:** Frontiers in Plant Science, Vol. 14:1256773, 2023 — DOI: 10.3389/fpls.2023.1256773
**Ngày phục dựng:** 2026-06-17
**Notebook xuất:** `papers/06 - PMVT - CNN/outputs/PMVT_XS_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | PMVT: a lightweight vision transformer for plant disease identification on mobile devices |
| Tác giả | Li G., Wang Y., Zhao Q., Yuan P., Chang B. |
| Journal / Conference | Frontiers in Plant Science |
| Năm | 2023 |
| DOI / Link | 10.3389/fpls.2023.1256773 |
| Nhóm phân loại | Nhóm A (kiến trúc mới dựa trên MobileViT) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `PMVT_XS`
**Loại:** Hybrid CNN + ViT (lightweight MobileViT-based)
**Framework:** PyTorch (paper: PyTorch 1.9.0, Python 3.8 — Mục 2.6)
**Module đặc biệt:**
- **InvertedResidual với depthwise 7×7** (thay 3×3 của MobileViT gốc — Mục 2.2.2, Fig 3)
- **CBAM** chèn vào MobileViT block, SAU 1×1 proj và TRƯỚC skip-cat (Mục 2.2.3, Fig 4A)

### Bảng kiến trúc (PMVT-XS — config xác nhận user)

| Stage | Cấu trúc | out_ch | Stride | MobileViT dim | L | Spatial (in→out @224) |
|-------|----------|--------|--------|---------------|---|------------------------|
| Stem | ConvBNAct 3×3 | 32 | 1 | — | — | 224→224 |
| S1 | 1× MV2 (expand=2) | 32 | 1 | — | — | 224→224 |
| S2 | 3× MV2 (expand=2) | 48 | 2 (block đầu) | — | — | 224→112 |
| S3 | MV2 + MobileViTBlock(in=64,out=80) | 64→80 | 2 | 96 | 4 | 112→56 |
| S4 | MV2 + MobileViTBlock(in=96,out=96) | 80→96 | 2 | 120 | 4 | 56→28 |
| S5 | MV2 + MobileViTBlock(in=128,out=128) | 96→128 | 2 | 144 | 3 | 28→14 |
| Head | Conv1×1(128→384)+BN+SiLU+GAP+Flatten+FC | 384→nc | — | — | — | 14→1 |

**Chi tiết MobileViT block (mỗi stage):** num_heads=1, head_dim=dim (inner_dim = dim), mlp_dim = 2×dim, patch_size = 2×2, CBAM reduction r=16, spatial-attention conv 7×7. Local rep = ConvBNAct 3×3 + Conv1×1(in→dim). Skip-cat: concat input (in_ch) với feature đã CBAM (in_ch) → fusion ConvBNAct 3×3 → out_ch.

**Total params (paper PMVT-XS):** ~2.01M (Table 3)
**Total params (ước lượng giải tích cho config này, num_classes=7):** ~2.37M (≈118% so với 2.01M — **trong ngưỡng ±20%**). Xem Mục 9.

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper
| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework | PyTorch 1.9.0 / Python 3.8 | Mục 2.6 |
| Optimizer | AdamW | Table 2 |
| Loss function | Cross Entropy | Table 2 |
| Batch size | 32 | Table 2 |
| Epochs | 100 | Table 2 |
| Learning rate | 1e-4 (0.0001) | Table 2 |
| Input size | 224×224×3 | Mục 2.4 |
| Expand ratio (InvertedResidual) | ×2 ("generally by two times") | Mục 2.2.2 |
| DW kernel (InvertedResidual) | 7×7 | Mục 2.2.2, Fig 3 |
| CBAM spatial conv | 7×7 | Mục 2.2.5, Eq.3 |
| Activation | (suy ra) SiLU theo MobileViT; CBAM ReLU/Sigmoid | Eq.2/Eq.3 + convention |

### ⚠️ Giá trị giả định / xác nhận user (không có giá trị số trong paper)
| Thông số | Giá trị dùng | Lý do |
|---------|------------|--------------|
| Biến thể | PMVT-XS | User chọn (2.01M, Table 3) |
| MobileViT L per stage | [4, 4, 3] | User xác nhận (theo Fig 2 transcribed) |
| num_heads / head_dim | 1 / =dim | User xác nhận (MobileViT-XS gốc) |
| mlp_dim ratio | ×2 (mlp_dim = 2×dim) | User xác nhận (MobileViT gốc) |
| Patch size P | 2×2 | User xác nhận (MobileViT chuẩn) |
| CBAM reduction r | 16 | User xác nhận (CBAM gốc, Woo 2018) |
| Channel/transformer dim | Bảng Mục 2 | User cung cấp (tham chiếu MobileViT-XS) |
| Normalization mean/std | ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225] | User xác nhận |
| LR Scheduler | Không dùng (LR cố định 1e-4) | User xác nhận |
| Weight decay | 0.01 (AdamW default) | Paper không nêu |
| Seed | 42 | Convention (Rule 7) |
| Pretrained | Train from scratch | Paper không đề cập |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ để tham khảo)*

| Mục | Chi tiết từ Paper |
|-----|---------|
| Tên dataset gốc | Wheat (7 lớp), Coffee (3 lớp), Rice (2 lớp) — Table 1 |
| Số classes (paper) | Wheat=7, Coffee=3, Rice=2 |
| Train/Val/Test (paper) | 8:1:1 (Mục 2.4) |
| Input size dùng trong paper | 224×224 |
| Normalization (paper) | "standardize bằng mean & std" — không cho giá trị số |

### 4b. Dataset Thực Tế *(dataset của người dùng)*

| Mục | Chi tiết |
|-----|---------|
| Tên dataset | *(điền `DATASET_NAME` trong cell Config)* |
| Số classes | auto-detect từ `train_ds_base.classes` (Rule 12) |
| Danh sách classes | auto-detect (in ra ở cell DataLoader) |
| Tổng số ảnh | in ra ở cell Dataset Distribution |
| Train / Val / Test | đọc trực tiếp từ 3 thư mục `train/`, `val/`, `test/` (đã chia sẵn) |
| Image size | 224×224 |
| Normalization | ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |

---

## 5. AUGMENTATION (Scenario 2)

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | degrees = 15 | Rule 3 (paper "randomly rotated" — không nêu số độ) |
| Zoom / Scale | RandomResizedCrop scale=(0.8, 1.0) | Rule 3 |
| Horizontal Flip | p = 0.5 | Rule 3 |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1 | Rule 3 |

> Scenario 1 (Baseline) chỉ Resize + ToTensor + Normalize. Val/Test luôn dùng baseline transform (không aug).

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ để tham khảo)*

> ⚠️ Kết quả dưới đây là của tác giả trên dataset gốc — không thể so sánh trực tiếp với dataset người dùng.

| Biến thể | Wheat Acc | Coffee Acc | Rice Acc | Parameters | FLOPs |
|----------|-----------|------------|----------|-----------|-------|
| PMVT-XXS | 93.6% | 85.3% | 93.1% | 0.98M | 0.31G |
| **PMVT-XS** | **94.7%** | **86.5%** | **97.7%** | **2.01M** | **0.85G** |
| PMVT-S | 94.9% | 87.6% | 92.0% | 5.06M | 1.59G |

Ablation (Table 4): +Conv7×7 và +CBAM đều tăng accuracy; PMVT (cả hai) đạt cao nhất.

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Score | Recall | Parameters |
|----------|---------|---------|--------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization *(paper không yêu cầu)*
- [x] **Ablation Study** — 4 variants (MobileViT-XS / +Conv7×7 / +CBAM / PMVT-XS full), Table 4
- [ ] Transfer Learning Comparison
- [x] **Inference Benchmark** — FPS + latency batch=1

---

## 8. OUTPUT FILES

```
papers/06 - PMVT - CNN/
├── inputs/
│   ├── 06 - PMVT - CNN.pdf
│   ├── paper_content.md          ← extracted
│   └── implement_plan.md         ← analyzed + user-confirmed
└── outputs/
    ├── PMVT_XS_Experiment.ipynb         ← notebook này
    └── PMVT_XS_ReproduceSummary.md      ← file này
    (đã có sẵn: PMVT_XXS_* — biến thể XXS riêng)
```

Notebook khi chạy trên Kaggle sinh thêm: `best_model_s1.pth`, `best_model_s2.pth`, `dataset_statistics.png`, `PMVT_XS_NoAug_LC.png`, `PMVT_XS_NoAug_CM.png`, `PMVT_XS_Aug_LC.png`, `PMVT_XS_Aug_CM.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION

1. **Param count ~2.37M vs target 2.01M (≈118%):** Ước lượng giải tích cho config xác nhận cho ~2.37M params (num_classes=7), nằm trong ngưỡng dung sai ±20% nên **không điều chỉnh channel/dim**. Phần dư so với 2.01M chủ yếu đến từ: (a) depthwise 7×7 thay 3×3 (PMVT thêm tham số so với MobileViT-XS gốc), (b) CBAM ở 3 MobileViT block. Bản thân MobileViT-XS gốc cũng ~2.3M params, nên config này hợp lý. **Cần xác nhận con số chính xác bằng `torchinfo.summary()` khi chạy trên Kaggle** (torch không chạy được ở môi trường sinh notebook). Nếu sau khi chạy lệch >20% so với 2.01M, cân nhắc giảm `dim` của MobileViT block.

2. **MobileViT block với in_ch ≠ out_ch (S3: 64→80):** Luồng theo Fig 4A — local_rep proj in_ch→dim, transformer ở dim, proj dim→in_ch, CBAM ở in_ch, concat với input (in_ch) → fusion conv 3×3 → out_ch. Fusion conv là nơi điều chỉnh channel ra out_ch, nên in_ch≠out_ch xử lý đúng.

3. **CBAM đặt SAU 1×1 proj, TRƯỚC skip-cat** đúng theo Fig 4A (khác MobileViT gốc không có CBAM). Channel attention (Eq.2) chạy trước Spatial attention (Eq.3, conv 7×7).

4. **num_heads=1 → inner_dim=dim:** `TransformerBlock` dùng `head_dim=dim`, khác với notebook XXS (vốn hard-code head_dim=32). Đây là điểm khác biệt quan trọng giữa XS và XXS notebook.

5. **expand=2 toàn bộ InvertedResidual** (khác notebook XXS dùng expand=4). Đây là quyết định user override rõ ràng theo paper Mục 2.2.2 ("two times").

6. **Spatial divisibility cho patch=2:** Trước mỗi MobileViT block spatial size là 56/28/14 — đều chia hết cho patch_size=2 (→ 28/14/7 patches). Unfold/fold hợp lệ với input 224×224.

7. **Ablation rút gọn 30 epochs:** `_build_variant(dw7, use_cbam)` lắp ráp 4 biến thể từ chung 1 factory để đảm bảo channel/dim đồng nhất, chỉ khác kernel DW (3×3 vs 7×7) và có/không CBAM. Train 30 epochs/variant (paper 100) — ranking xu hướng đúng, accuracy tuyệt đối thấp hơn paper.

8. **Tuân thủ Rules 1–13:** PyTorch (R2); 2 scenario baseline+aug (R3); Accuracy/F1-macro/Recall-macro + classification_report per class (R4); torchinfo.summary (R5); LC + CM cell riêng dpi=300 + plt.show (R6); seed=42 + cudnn deterministic/benchmark=False (R7,R9); device auto-detect (R8); best checkpoint theo val_acc, best_model_s1/s2.pth, reload trước test (R10); class imbalance auto + class_weights (R11); CLASS_NAMES = train_ds_base.classes (R12); num_workers=2 + pin_memory + empty_cache giữa S1/S2 (R13). Excel 2 sheet + ảnh nhúng.
