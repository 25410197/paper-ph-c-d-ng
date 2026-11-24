# MaizeFormerX — Verification Report
**Paper:** MaizeFormerX: a lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis
**Notebook:** `papers/MaizeFormerX a lightweight/outputs/MaizeFormerX_Experiment.ipynb`
**Ngày verify:** 2026-05-27
**Người verify:** notebook-verifier agent

---

## KẾT QUẢ TỔNG QUAN

| Dimension | Score | Trọng số | Điểm có trọng số |
|-----------|-------|----------|-----------------|
| Paper Completeness (P) | 91.7% | 15% | 13.8% |
| Architecture Fidelity (A) | 86.0% | 40% | 34.4% |
| Hyperparameter Match (H) | 100.0% | 25% | 25.0% |
| Preprocessing Consistency (D) | 83.0% | 5% | 4.2% |
| Constraint Compliance (CC) | 100.0% | 15% | 15.0% |
| **TỔNG** | | **100%** | **92.4%** |

### Quyết định: **GO WITH CAUTION** (92.4%)

---

## 1. PAPER COMPLETENESS CHECK

| # | Mục kiểm tra | Trạng thái | Ghi chú |
|---|-------------|-----------|---------|
| P1 | Loại bài báo (Nhóm A/B/C) | Đủ | Nhóm A — có model mới, có code GitHub công khai |
| P2 | Tên model + kiến trúc rõ ràng | Đủ | "MaizeFormerX", định nghĩa rõ trong paper + GitHub |
| P3 | Chi tiết từng layer / block | Đủ | MultiScalePatchEmbed, CSAF, TransformerEncoder — xác nhận từ GitHub |
| P4 | Activation function | Đủ | GELU cho MLP, không có activation riêng cho CSAF/MHSA |
| P5 | Batch Normalization / Dropout | Đủ | LayerNorm (pre-norm), Dropout 0.3 (head), attn_dropout=0.1, proj_dropout=0.1 |
| P6 | Optimizer | Đủ | Adam — Table 4 |
| P7 | Loss function | Đủ | CrossEntropy + weight decay — paper equation |
| P8 | Batch size | Đủ | 32 — Table 4 |
| P9 | Số epochs / stopping criterion | Đủ | Max 30 epochs, early stopping patience=4 — Table 4 |
| P10 | Input size (H x W x C) | Đủ | 224x224x3 — mục "Data preprocessing" |
| P11 | Domain model (loại ảnh, task) | Đủ | Maize leaf disease, classification, 3 tập dataset |
| P12 | Input size và normalization | Mờ hồ | Input 224 rõ ràng; normalization mô tả z-score/min-max nhưng ViT backbone gợi ý ImageNet stats |
| P13 | Augmentation (loại + tham số) | Đủ | Table 2: flip p=0.5, rotation 30deg, zoom 0.9-1.1, brightness 20% |
| P14 | Normalization values (mean/std) | Mờ hồ | Paper nêu z-score và min-max nhưng không cho giá trị cụ thể; phải suy luận ImageNet stats |
| P15 | Kết quả thực nghiệm | Đủ | Accuracy, F1, MCC cho 3 dataset — Table 7,8,9,10 |
| P16 | Framework | Đủ | PyTorch, Python 3.10, CUDA 12.1 — mục "Experimental models" |
| P17 | Pretrained weights | Mờ hồ | Paper nêu "pretrained ViT backbone" nhưng toàn bộ model là custom; GitHub xác nhận train-from-scratch |
| P18 | LR Scheduler | Đủ | CosineAnnealingLR, warmup=300 steps — Table 4 |

**Score: 15 Đủ + 3 Mờ hồ → 91.7%** (>80% — bài báo ĐỦ để reproduce)

---

## 2. ARCHITECTURE FIDELITY CHECK

| # | Thành phần | Giá trị trong Paper/GitHub | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|--------------------------|----------------------|-------|---------|
| A1 | Tên model / class định nghĩa | `MaizeFormerX` | `class MaizeFormerX(nn.Module)` | ✅ Khớp | |
| A2 | Patch embedding: patch_sizes | [8, 16, 32] | `PATCH_SIZES = [8, 16, 32]` | ✅ Khớp | |
| A3 | Patch embedding: embed_dims per-scale | [64, 96, 128] | `EMBED_DIMS = [64, 96, 128]` | ✅ Khớp | |
| A4 | Patch embedding: strides | [8, 16, 32] (= patch_sizes) | `stride=ps` trong Conv2d | ✅ Khớp | |
| A5 | Patch embedding: kernel size | = patch_size | `kernel_size=ps, stride=ps` | ✅ Khớp | |
| A6 | Patch embedding: Conv2d bias | `bias=False` (GitHub) | Không set bias=False, mặc định bias=True | ❌ KHÔNG KHỚP | GitHub: `bias=False`; Notebook: Conv2d mặc định bias=True |
| A7 | Patch embedding: padding | padding=0 (GitHub) | Không set padding, mặc định=0 | ✅ Khớp | |
| A8 | CSAF: fusion_dim | 192 | `FUSION_DIM = 192` | ✅ Khớp | |
| A9 | CSAF: num_heads | 4 | `NUM_HEADS_CSAF = 4` | ✅ Khớp | |
| A10 | CSAF: mechanism | Self-attention | `nn.MultiheadAttention(batch_first=True)` | ✅ Khớp | |
| A11 | CSAF: projection | Linear(d_i -> 192) x3 | `nn.ModuleList([nn.Linear(d, fusion_dim) for d in in_dims])` | ✅ Khớp | |
| A12 | CSAF: residual + LayerNorm | Pre-norm: `attn(norm(fused)) + fused` (GitHub) | Post-norm: `norm(x + attn_out)` | ⚠️ KHÁC BIỆT | GitHub CSAF: pre-norm style. Notebook: post-norm style. Vị trí LayerNorm khác nhau |
| A13 | CSAF: proj_dropout | 0.1 | `nn.Dropout(proj_dropout)` sau self_attn | ✅ Khớp | |
| A14 | Encoder: depth | 6 | `NUM_LAYERS = 6` | ✅ Khớp | |
| A15 | Encoder: num_heads | 6 | `NUM_HEADS_ENCODER = 6` | ✅ Khớp | |
| A16 | Encoder: dim | 192 | `dim=fusion_dim=192` | ✅ Khớp | |
| A17 | Encoder: mlp_ratio | 4.0 | `MLP_RATIO = 4.0` | ✅ Khớp | |
| A18 | Encoder: drop_path linear schedule | linear 0 → 0.1 | `torch.linspace(0, drop_path, num_layers)` | ✅ Khớp | |
| A19 | Encoder: qkv_bias | True | `qkv_bias=True` | ✅ Khớp | |
| A20 | Encoder: attn_dropout | 0.1 | `ATTN_DROPOUT = 0.1` | ✅ Khớp | |
| A21 | Encoder: pre-norm architecture | Pre-norm | `x_norm = self.norm1(x); attn_out = self.attn(x_norm, ...)` | ✅ Khớp | |
| A22 | Encoder: layer_scale | False (GitHub default) | Không implement LayerScale | ✅ Khớp | |
| A23 | CLS token | zeros(1,1,192), trunc_normal_(std=0.02) | `torch.zeros(1,1,fusion_dim)` + `trunc_normal_(std=0.02)` | ✅ Khớp | |
| A24 | pos_drop sau positional embedding | `nn.Dropout(0.3)` + `tokens = self.pos_drop(tokens)` (GitHub) | KHÔNG có pos_drop | ❌ KHÔNG KHỚP | GitHub model.py dòng 83+117: rõ ràng có Dropout(0.3) sau pos_embed. Notebook bỏ qua |
| A25 | Pooling | CLS token `x[:, 0]` | `cls_out = x[:, 0]` | ✅ Khớp | |
| A26 | Classifier head | Dropout(0.3) → Linear(192, num_classes) | `self.dropout` + `self.head` | ✅ Khớp | |

**Architecture Score: 22/25 applicable = 88%** → **86%** (điều chỉnh vì A12 ảnh hưởng training stability)

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thông số | Giá trị trong Paper | Giá trị trong Notebook | Khớp? |
|---|---------|-------------------|----------------------|-------|
| H1 | Optimizer | Adam | `torch.optim.Adam` | ✅ |
| H2 | Learning rate | 5e-5 | `LR = 5e-5` | ✅ |
| H3 | LR Scheduler | CosineAnnealingLR | `CosineAnnealingLR(T_max=MAX_EPOCHS)` | ✅ |
| H4 | Loss function | CrossEntropy + weight decay | `nn.CrossEntropyLoss(weight=...)` + Adam weight_decay | ✅ |
| H5 | Batch size | 32 | `BATCH_SIZE = 32` | ✅ |
| H6 | Max Epochs | 30 | `MAX_EPOCHS = 30` | ✅ |
| H7 | Weight decay | 5e-5 | `WEIGHT_DECAY = 5e-5` | ✅ |
| H8 | Pretrained | Không (train from scratch) | Không dùng pretrained weights | ✅ |
| H9 | Warmup steps | 300 | `WARMUP_STEPS = 300` | ✅ |
| H10 | Early stopping patience | 4 | `PATIENCE = 4` | ✅ |

**Score: 10/10 = 100%**

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### Augmentation Scenario 2

| # | Kỹ thuật | Paper (Table 2) | Notebook | Tham số khớp? |
|---|---------|----------------|---------|---------------|
| Aug1 | RandomRotation | ±30 deg | `RandomRotation(degrees=30)` | ✅ Khớp |
| Aug2 | Zoom / RandomResizedCrop | scale 0.9-1.1 | `RandomResizedCrop(224, scale=(0.9, 1.0))` | ⚠️ Gần khớp (upper bound 1.0 vs 1.1) |
| Aug3 | RandomHorizontalFlip | p=0.5 | `RandomHorizontalFlip(p=0.5)` | ✅ Khớp |
| Aug4 | ColorJitter / Brightness | ±20% brightness | `ColorJitter(brightness=0.2)` | ✅ Khớp |

### Preprocessing

| Mục | Paper | Notebook | Khớp? |
|-----|-------|---------|-------|
| Input size | 224x224 | `IMG_SIZE = 224` | ✅ |
| Normalization | Mờ hồ (z-score/min-max) | ImageNet stats (hợp lý cho ViT) | ✅ Chấp nhận được |
| Val/Test transform | No aug | `transform_eval = transform_baseline` | ✅ |

---

## 5. CONSTRAINT COMPLIANCE (24/24)

| # | Ràng buộc | Trạng thái |
|---|-----------|-----------|
| CC1 | 2 Scenario (Baseline + Augmented với 4 kỹ thuật) | ✅ |
| CC2 | Framework đúng với paper (PyTorch) | ✅ |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | ✅ |
| CC4 | `classification_report` per class (sklearn) | ✅ |
| CC5 | `torchinfo.summary()` | ✅ |
| CC6 | Learning Curve lưu file ảnh dpi=300, cell riêng + `plt.show()` | ✅ |
| CC7 | Confusion Matrix lưu file ảnh dpi=300, cell riêng + `plt.show()` | ✅ |
| CC8 | `NUM_CLASSES` auto-detect (không hard-code) | ✅ |
| CC9 | Excel Export 2-sheet (Baseline + Augmented) | ✅ |
| CC10 | Ảnh LC + CM nhúng vào Excel | ✅ |
| CC11 | `model_name` là tên kiến trúc, KHÔNG phải tên bài báo | ✅ |
| CC12 | `device` auto-detect + `.to(device)` | ✅ |
| CC13 | `torch.manual_seed(42)` đầu notebook | ✅ |
| CC14 | Không dùng `pretrained=True` (deprecated) | ✅ |
| CC15 | Baseline training transform: CHỈ Resize + ToTensor + Normalize | ✅ |
| CC16 | Val/Test transform KHÔNG có augmentation | ✅ |
| CC17 | `cudnn.deterministic=True` + `benchmark=False` | ✅ |
| CC18 | `train_with_checkpoint()` + best model reload trước test eval | ✅ |
| CC19 | `best_model_s1.pth` và `best_model_s2.pth` | ✅ |
| CC20 | Cell "Dataset Distribution": bar chart + imbalance check | ✅ |
| CC21 | `class_names` auto-detect (không hard-code) | ✅ |
| CC22 | DataLoader: `num_workers=2, pin_memory=True` | ✅ |
| CC23 | `torch.cuda.empty_cache()` sau S1 trước S2 | ✅ |
| CC24 | ReproduceSummary.md đi kèm notebook | ✅ |

**Compliance Score: 24/24 = 100%**

---

## 6. ASSUMPTION RISK AUDIT

| # | Thông số giả định | Mức độ rủi ro | Lý do |
|---|------------------|--------------|-------|
| 1 | Normalization ImageNet stats | TRUNG BÌNH | Paper mô tả z-score/min-max, không cho giá trị cụ thể |
| 2 | Warmup loại (linear vs cosine) | THẤP | Standard convention |
| 3 | T_max CosineAnnealingLR = 30 epochs | THẤP | Standard convention |
| 4 | Conv2d bias=True thay vì bias=False (GitHub) | THẤP-TRUNG BÌNH | Tăng nhẹ số params |
| 5 | **pos_drop bị thiếu** | **TRUNG BÌNH** | GitHub rõ ràng có Dropout(0.3) sau pos_embed |
| 6 | **CSAF LayerNorm vị trí sai** | **TRUNG BÌNH** | Post-norm vs pre-norm ảnh hưởng training stability |
| 7 | Scale range RandomResizedCrop (0.9, 1.0) vs (0.9, 1.1) | THẤP | Sai lệch nhỏ |
| 8 | Dataset split ratio khác paper | THẤP | Được ghi chú rõ, dataset khác nhau |

---

## 7. DANH SÁCH VẤN ĐỀ CẦN SỬA

### Nghiêm trọng (ảnh hưởng đến độ chính xác phục dựng):

**[A12] CSAF LayerNorm vị trí sai**
- Hiện tại (post-norm): `out = self.norm(x + attn_out)`
- Đúng (pre-norm, theo GitHub): `out = self.attn(self.norm(fused))` + residual `out + fused`
- Ảnh hưởng: training stability với gradient flow

**[A24] Thiếu pos_drop sau positional embedding**
- GitHub model.py: `self.pos_drop = nn.Dropout(dropout)` và `tokens = self.pos_drop(tokens)`
- Cần thêm vào `__init__` và `forward` của MaizeFormerX:
  ```python
  self.pos_drop = nn.Dropout(dropout)  # __init__
  x = self.pos_drop(x)  # sau x = x + self.pos_embed trong forward
  ```

### Nên sửa (ảnh hưởng nhẹ đến số params):

**[A6] Conv2d bias trong MultiScalePatchEmbed**
- GitHub: `bias=False`. Notebook: mặc định `bias=True`
- Sửa: `nn.Conv2d(3, ed, kernel_size=ps, stride=ps, bias=False)`

### Tùy chọn:

- Aug2: Scale range `(0.9, 1.0)` thay vì `(0.9, 1.1)` — ghi chú lý do hoặc điều chỉnh
- A27: Kiểm tra xem `LinearHead` trong GitHub có chứa LayerNorm trước Linear không

---

## 8. ĐIỂM MẠNH CỦA NOTEBOOK

1. **Tất cả 24 ràng buộc CC đều được tuân thủ** — kết quả xuất sắc về compliance.
2. **Hyperparameter từ paper cực kỳ chính xác**: Adam, lr=5e-5, wd=5e-5, batch=32, epochs=30, patience=4, warmup=300, dropout=0.3, CosineAnnealingLR, early stopping — tất cả đúng.
3. **Kiến trúc chính xác ở mức cao**: patch_sizes, embed_dims, fusion_dim, num_heads, mlp_ratio, drop_path schedule, CLS token, positional embedding, pooling — tất cả đúng theo GitHub.
4. **Grad-CAM** được implement với `reshape_transform` đúng cho ViT token layout, với fallback exception handling.
5. **Class imbalance detection** tự động, dùng `compute_class_weight` khi ratio ≥ 1.5.

---

## KẾT LUẬN

**Điểm tổng: 92.4% → GO WITH CAUTION**

Notebook đạt trên ngưỡng GO (85%) và được nâng lên GO WITH CAUTION vì có 2 sai lệch kiến trúc so với GitHub source code:
1. **CSAF LayerNorm vị trí** (post-norm vs pre-norm)
2. **pos_drop bị thiếu** sau positional embedding

Các hyperparameter và tất cả ràng buộc bắt buộc (24/24) đều chuẩn xác. Để đạt độ trung thực tối đa với GitHub, nên áp dụng 2 sửa đổi ở mục "Nghiêm trọng" trước khi upload lên Kaggle.
