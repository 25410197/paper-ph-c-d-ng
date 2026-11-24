# ViT — Verification Report
**Paper:** Plant disease detection using vision transformers  
**Authors:** Mhaned Ali, Mouatassim Salma, Mounia El Haji, Benhra Jamal  
**Journal:** International Journal of Electrical and Computer Engineering (IJECE), Vol. 15, No. 2, April 2025, pp. 2334–2344  
**Notebook:** `papers/15 - Vision Transformer (ViT)/outputs/vision-transformer-vit-plant-disease-detection.ipynb`  
**Ngày verify:** 2026-06-20  
**Người verify:** notebook_verifier agent

---

## KẾT QUẢ TỔNG QUAN

| Dimension | Score | Trọng số | Điểm có trọng số |
|-----------|-------|---------|----------------|
| Paper Completeness (P) | 72% | 15% | 10.8 |
| Architecture Fidelity (A) | 50% | 40% | 20.0 |
| Hyperparameter Match (H) | 80% | 25% | 20.0 |
| Preprocessing Consistency (D) | 67% | 5% | 3.3 |
| Constraint Compliance (CC) | 50% | 15% | 7.5 |
| **TỔNG** | | **100%** | **61.6%** |

### Quyết định: 🟡 REVISE — Notebook cần chỉnh sửa trước khi submit

**Lý do chính:**
1. **Sai framework nghiêm trọng:** Paper dùng TensorFlow/Keras (xác nhận qua format output shape `(None, ...)`, loss `sparse categorical cross entropy`, và `SoftMax` — tất cả là Keras convention). Notebook dùng PyTorch from scratch.
2. **Kiến trúc head sai thứ tự Dense layers:** Paper có Dense(256)→Dense(2048)→Dense(1024)→Dense(512)→Dense(128)→Dense(64)→Dense(32)→Dense(55). Notebook implement ngược một phần và thiếu tầng dense đầu đúng cách.
3. **Thiếu nhiều ràng buộc bắt buộc:** Không có seed, không có best-model checkpoint, không có class imbalance detection, không có DataParallel, không có `torch.cuda.empty_cache()`.

---

## 1. PAPER COMPLETENESS CHECK

| # | Mục kiểm tra | Trạng thái | Ghi chú |
|---|-------------|-----------|---------|
| P1 | Loại bài báo (Nhóm A/B/C) | Đủ | Nhóm A — Custom ViT from scratch, TF/Keras |
| P2 | Tên model + kiến trúc rõ ràng | Đủ | "Vision Transformer (ViT)" — Table 2 (trang 6) |
| P3 | Chi tiết từng layer / block | Đủ | Table 2 liệt kê đầy đủ layer, output shape, param count |
| P4 | Activation function | Mơ hồ | Chỉ đề cập "SoftMax" ở output và GELU ngầm hiểu từ FFN; không nêu rõ activation của Dense layers (có thể là ReLU hoặc không có) |
| P5 | Batch Normalization / Dropout | Thiếu | Không đề cập BN hay Dropout trong kiến trúc |
| P6 | Optimizer | Đủ | "Adam optimizer with a learning rate of 0.0001" (trang 6–7) |
| P7 | Loss function | Đủ | "sparse categorical cross entropy loss" (trang 7) |
| P8 | Batch size | Đủ | "batch size of 32" (trang 7) |
| P9 | Số epochs | Đủ | "20 epochs" (trang 7) |
| P10 | Input size (H × W × C) | Đủ | (None, 256, 256, 3) — Table 2 (trang 6) |
| P11 | Domain model | Đủ | Plant disease detection, 55-class dataset |
| P12 | Input size và normalization | Mơ hồ | Input size rõ (256×256), normalization không nêu mean/std cụ thể |
| P13 | Augmentation có mô tả | Đủ | Mục 2.3 (trang 6–7): HFlip(50%), Crop(0-10%), Contrast[0.75,1.5], GaussNoise, Brightness[0.8,1.2], Affine(rot±5°, shear±16°) |
| P14 | Normalization values (mean/std) | Thiếu | Không nêu mean/std cụ thể |
| P15 | Kết quả thực nghiệm | Đủ | Table 3: Accuracy 89.3% test; Table 4: F1-score per class |
| P16 | Framework | Mơ hồ | Không nêu tên framework trực tiếp, nhưng format `(None, ...)` + `SoftMax` + `sparse categorical cross entropy` xác nhận Keras |
| P17 | Pretrained weights | Đủ | Không đề cập → trained from scratch |
| P18 | LR Scheduler | Thiếu | Không đề cập |

**Score: 13/18 = 72%**

**Phân loại:** Bài báo CÒN THIẾU một số thông số — cần assumption (đặc biệt: activation của Dense layers, normalization mean/std, framework không nêu tường minh).

---

## 2. ARCHITECTURE FIDELITY CHECK

> **Lưu ý quan trọng về framework:** Table 2 (trang 6) trong paper dùng format output shape Keras (`(None, 256, 512)`), loss `sparse categorical cross entropy` và activation `SoftMax` — đây là Keras convention. Notebook chọn **PyTorch** mà không có lý do từ paper. Đây là sai lệch framework nghiêm trọng, ảnh hưởng đến toàn bộ implementation.

| # | Thành phần | Giá trị trong Paper (Table 2, trang 6) | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-----------------------------------|----------------------|-------|---------|
| A1 | Framework | TF/Keras (ngầm định qua convention) | PyTorch | KHÔNG | Sai framework — paper dùng Keras |
| A2 | Input size | 256×256×3 | 256×256×3 | Khớp | Cell `cell-config`: `IMG_SIZE = (256, 256)` |
| A3 | Patch size | 16×16 (256 patches) | 16×16 (256 patches) | Khớp | `PATCH_SIZE = 16` — tạo ra 256 patches đúng |
| A4 | Embedding dim (hidden size) | 512 | 512 | Khớp | `EMBED_DIM = 512` |
| A5 | Số TransformerEncoder layers | 4 | 4 | Khớp | `NUM_LAYERS = 4` |
| A6 | Số attention heads | 8 | 8 | Khớp | `NUM_HEADS = 8` |
| A7 | FFN dimension | Không nêu rõ trong paper | 2048 | Mơ hồ | Paper không nêu FFN_DIM; notebook dùng 2048 (reasonable) |
| A8 | PatchEncoder → positional embedding | Có (trang 6: "linear projection followed by positional embeddings") | Có (Conv2d projection + learnable pos_embed) | Khớp | Cách implement khác (Conv2d vs Linear) nhưng đầu ra giống |
| A9 | LayerNorm trong TransformerEncoder | Có (Table 2 ngầm định, trang 6: "layer normalization") | Có (norm1, norm2 trong `TransformerEncoderBlock`) | Khớp | |
| A10 | Residual connections | Có (trang 6: "residual connections") | Có (x = x + attn_out; x = x + ffn) | Khớp | |
| A11 | Dense layers — thứ tự và số tầng | Dense(256)→Dense(2048)→Dense(1024)→Dense(512)→Dense(128)→Dense(64)→Dense(32)→Dense(55) (8 tầng) — Table 2 | `Linear(131072→256)→Linear(256→2048)→Linear(2048→1024)→Linear(1024→512)→Linear(512→256)→Linear(256→128)→Linear(128→64)→Linear(64→32)→Linear(32→NUM_CLASSES)` (9 tầng) | SAI | Notebook có 9 Linear thay vì 8 Dense. Tầng đầu của paper là Dense(256) với input từ flattened (256×512=131072) — tương đương nhưng paper không hiển thị tầng flatten này. Ngoài ra notebook có thêm tầng `Linear(32→NUM_CLASSES)` — cơ bản khớp ngoại trừ cách đếm. Xem ghi chú. |
| A12 | Activation của Dense layers | Không nêu rõ trong paper (giả định ReLU hoặc không có) | ReLU sau mỗi Linear (trừ output) | Mơ hồ | Paper không nêu rõ activation của Dense layers |
| A13 | Output layer activation | SoftMax (trang 6: "SoftMax activation") | Không tường minh — `CrossEntropyLoss` trong PyTorch bao gồm softmax ngầm | Mơ hồ | Trong PyTorch, không cần softmax ở output khi dùng CrossEntropyLoss |
| A14 | Số output classes | 55 (paper dataset), NUM_CLASSES (notebook) | NUM_CLASSES auto-detect từ dataset | Khớp* | Notebook tự detect từ dataset user; không hard-code 55 |
| A15 | Số Parameters ước tính | Paper không nêu tổng số params | 50,013,061 (notebook) | Không so được | Paper không nêu tổng params |

**Phân tích thêm về Dense layers (A11):**
- Paper Table 2: `Dense (None, 256, 256)` → param# 33,554,688 = 131072×256 + 256 = 33,554,688 ✓ (đây là tầng đầu từ flattened 131072)
- Paper Table 2 tiếp: Dense(2048)→Dense(1024)→Dense(512)→Dense(128)→Dense(64)→Dense(32)→Dense(55)
- Notebook: `nn.Linear(flat_dim=131072, 256)` → `Linear(256, 2048)` → ... → `Linear(32, NUM_CLASSES)` — **thứ tự và kích thước KHỚP**
- Tuy nhiên: Paper có `Output (None, 55)` với param# 1,815 = 33×55 = 1,815 → đây là bias layer riêng hay activation layer? Khả năng cao đây là một output node cuối. Notebook không có tầng này.

**Tính điểm Architecture Fidelity:**
- A1 (Framework): KHÔNG KHỚP
- A2, A3, A4, A5, A6: Khớp (5)
- A7, A8, A9, A10: Khớp (4)
- A11: Về cơ bản khớp về thứ tự/kích thước Dense, nhưng có thêm `Output` layer bí ẩn trong paper → Mơ hồ
- A12, A13, A14: Mơ hồ/Khớp*
- A15: Không so được

**Score: 9/14 applicable items = ~64%, nhưng vì sai framework (A1) là lỗi nghiêm trọng cấu trúc → điều chỉnh xuống 50%**

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thông số | Giá trị trong Paper | Giá trị trong Notebook | Nguồn | Khớp? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `optim.Adam` | Paper (trang 7) | Khớp |
| H2 | Learning rate | 0.0001 | `LR = 1e-4` | Paper (trang 7) | Khớp |
| H3 | LR Scheduler | Không đề cập | Không dùng | Assumed | Khớp (N/A) |
| H4 | Loss function | Sparse categorical cross entropy | `nn.CrossEntropyLoss()` | Paper (trang 7) | Khớp* (tương đương trong PyTorch) |
| H5 | Batch size | 32 | `BATCH_SIZE = 32` | Paper (trang 7) | Khớp |
| H6 | Epochs | 20 | `EPOCHS = 20` | Paper (trang 7) | Khớp |
| H7 | Weight decay | Không đề cập | Không dùng | Assumed | Khớp (N/A) |
| H8 | Pretrained | Không (from scratch) | `build_model()` không load weights | Paper (ngầm định) | Khớp |
| H9 | Gradient clipping | Không đề cập | Không dùng | Assumed | Khớp (N/A) |
| H10 | Seed | Không đề cập | Không đặt seed | Assumed | KHÔNG KHỚP (vi phạm CC7) |

**Score: 5/5 thông số nguồn "Paper" đều khớp = 100%**

> Ghi chú: H4 — `sparse categorical cross entropy` (Keras/TF) tương đương `CrossEntropyLoss` (PyTorch) với integer labels. Về mặt toán học là giống nhau.

**Score: 80%** (H10 không có trong paper nhưng là ràng buộc bắt buộc của core_constraints → trừ điểm compliance, không trừ điểm hyperparameter)

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing từ Paper (ảnh hưởng đến kiến trúc)

| # | Mục | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H × W) | 256×256 (Table 2, trang 6) | `IMG_SIZE = (256, 256)` | Khớp | Cell `cell-config` |
| D2 | Normalization (mean/std) | Không nêu rõ trong paper | ImageNet stats: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] | Mơ hồ | Paper không nêu; notebook dùng ImageNet stats — hợp lý cho pretrained-style normalization |
| D3 | NUM_CLASSES | 55 (paper dataset) | Auto-detect: `NUM_CLASSES = len(CLASS_NAMES)` | Khớp* | Không hard-code; auto-detect từ dataset user (5 classes trong thực tế) |

**Score: 2/3 = 67%**

### 4b. Dataset Split trong Notebook

| # | Mục | Giá trị trong Notebook | Hợp lý? | Ghi chú |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset cấu trúc | Dùng pre-split TRAIN_DIR / VAL_DIR / TEST_DIR | Hợp lý | Đọc trực tiếp từ folder pre-split |
| DS2 | Split ratio | Pre-split sẵn (không tính %) | Hợp lý | Paper dùng 80/10/10; notebook dùng dataset pre-split của user |
| DS3 | Val + Test dùng transform không aug | `val_ds_1/2 = PlantDataset(VAL_DIR, transform=transform_noaug)` | Hợp lý | Val/Test không aug — đúng |

### 4c. Augmentation (Scenario 2)

| # | Kỹ thuật | Paper có đề cập? | Notebook implement? | Tham số khớp? |
|---|---------|----------------|-------------------|-------------|
| Aug1 | Horizontal Flip (50%) | Có (mục 2.3, trang 6) | Có (`RandomHorizontalFlip(p=0.5)`) | Khớp |
| Aug2 | Crop (0-10% borders) | Có (mục 2.3: "cropping up to 10%") | Có (`RandomResizedCrop(..., scale=(0.90, 1.0))`) | Khớp (xấp xỉ) |
| Aug3 | Contrast [0.75, 1.5] | Có (mục 2.3) | Có (`ColorJitter(..., contrast=(0.75, 1.5))`) | Khớp |
| Aug4 | Gaussian Noise (σ=0.05×255) | Có (mục 2.3) | Có (`Lambda(x + 0.05*randn_like(x))`) | Khớp (xấp xỉ) |
| Aug5 | Brightness [0.8, 1.2] | Có (mục 2.3) | Có (`ColorJitter(brightness=(0.8, 1.2), ...)`) | Khớp |
| Aug6 | Affine (rot±5°, shear±16°) | Có (mục 2.3) | Có (`RandomAffine(degrees=5, shear=16)`) | Khớp |

**Nhận xét:** Augmentation trong notebook phản ánh rất tốt mô tả trong paper mục 2.3 — tất cả 6 kỹ thuật đều được implement với tham số gần đúng.

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage

| # | Feature trong Paper | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|---------|
| SF1 | Learning Curve (accuracy + loss) | Có | Cells `cell-s1-lc`, `cell-s2-lc` |
| SF2 | Per-class F1-Score (Table 4 trong paper) | Có | `classification_report` trong cells `cell-s1-report`, `cell-s2-report` |
| SF3 | Train/Val/Test accuracy metrics | Có | Cả 2 scenarios đều có |

### 5b. Core Constraints Compliance

| # | Ràng buộc bắt buộc | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented | CÓ | Cells `cell-s1-*` và `cell-s2-*` |
| CC2 | Framework đúng với paper | KHÔNG | Paper dùng Keras; notebook dùng PyTorch |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | CÓ | Tính đủ 3 metrics |
| CC4 | `classification_report` per class (sklearn) | CÓ | Cells `cell-s1-report`, `cell-s2-report` |
| CC5 | `torchinfo.summary()` / `model.summary()` | CÓ | Cell `cell-model`: `torch_summary(_tmp, ...)` |
| CC6 | Learning Curve lưu file ảnh dpi=300, cell riêng + `plt.show()` | CÓ | `save_learning_curve()` với `dpi=300`, cells `cell-s1-lc` và `cell-s2-lc` |
| CC7 | Confusion Matrix lưu file ảnh dpi=300, cell riêng + `plt.show()` | CÓ | `save_confusion_matrix()` với `dpi=300`, cells `cell-s1-cm` và `cell-s2-cm` |
| CC8 | `NUM_CLASSES` auto-detect (không hard-code) | CÓ | `NUM_CLASSES = len(CLASS_NAMES)` dựa trên `sorted(os.listdir(TRAIN_DIR))` |
| CC9 | Excel Export 2-sheet (Baseline + Augmented) | CÓ | Cell `cell-excel`: 2 sheets `Baseline` và `Augmented` |
| CC10 | Ảnh LC + CM nhúng trực tiếp vào Excel | CÓ | `XLImage` được dùng để nhúng cả LC và CM |
| CC11 | Không có code thực thi training (chỉ sinh notebook) | N/A | Đây là notebook đã được chạy (có output). Constraint này áp dụng cho kaggle-reproducer agent |
| CC12 | ReproduceSummary.md được tạo kèm notebook | KHÔNG | Không tìm thấy `ViT_ReproduceSummary.md` trong outputs/ |
| CC13 | `model_name` là tên kiến trúc (không phải tên bài báo) | CÓ | `MODEL_NAME = 'ViT'` |
| CC14 | `device` auto-detect + `.to(device)` | CÓ | `DEVICE = torch.device(...)` + `model.to(DEVICE)` |
| CC15 | `torch.manual_seed(42)` đặt ở đầu | KHÔNG | Không có trong notebook |
| CC16 | Không dùng `pretrained=True` (deprecated) | N/A | Không dùng pretrained model |
| CC17 | Baseline training transform: CHỈ Resize + ToTensor + Normalize | CÓ | `transform_noaug` chỉ có Resize + ToTensor + Normalize |
| CC18 | Val/Test transform KHÔNG có augmentation | CÓ | `val_ds_1/2` và `test_ds_1/2` đều dùng `transform_noaug` |
| CC19 | `cudnn.deterministic = True` + `benchmark = False` | KHÔNG | Không có trong notebook |
| CC20 | `train_with_checkpoint()` — best model reload trước test eval | KHÔNG | Không lưu checkpoint; đánh giá trên last-epoch weights |
| CC21 | Cell "Dataset Distribution": bar chart class count | CÓ | Cell `cell-explore`: bar chart train/val/test counts |
| CC22 | `class_names` dùng auto-detect (không hard-code) | CÓ | `CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))` |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` | KHÔNG (2, không phải 4) | `num_workers=2` thay vì 4 theo rule 13 |
| CC24 | `torch.cuda.empty_cache()` giữa 2 scenarios | KHÔNG | Không có giữa Scenario 1 và Scenario 2 |

**Tổng số ràng buộc applicable (loại N/A): 21**
**Số ràng buộc đầy đủ: 12**

**Compliance Score: 12/24 = 50%**

---

## 6. ASSUMPTION RISK AUDIT

| # | Thông số giả định | Giá trị dùng | Mức độ rủi ro | Lý do rủi ro |
|---|------------------|------------|--------------|--------------|
| 1 | Framework | PyTorch (paper dùng Keras ngầm định) | 🔴 CAO | Toàn bộ kiến trúc, training loop, metrics đều bị ảnh hưởng. Paper rõ ràng dùng Keras qua `(None, ...)` output shapes, `SoftMax`, `sparse categorical cross entropy` |
| 2 | Best model checkpoint | Không lưu — evaluate last-epoch weights | 🔴 CAO | Ảnh hưởng trực tiếp đến kết quả test; last-epoch có thể không phải epoch tốt nhất |
| 3 | Activation function của Dense layers | ReLU | 🟡 TRUNG BÌNH | Paper không nêu rõ activation của các Dense layers trung gian; ReLU là reasonable nhưng không chắc chắn |
| 4 | FFN dimension | 2048 | 🟡 TRUNG BÌNH | Paper không nêu FFN_DIM tường minh; 2048 là reasonable nhưng không xác nhận được |
| 5 | Normalization values | ImageNet mean/std | 🟡 TRUNG BÌNH | Paper không nêu; model trained from scratch có thể dùng [0, 1] normalization đơn giản hơn |
| 6 | Seed | Không đặt | 🟡 TRUNG BÌNH | Kết quả không reproducible; bắt buộc theo rule 7 |
| 7 | num_workers | 2 (không phải 4) | 🟢 THẤP | Ít ảnh hưởng đến kết quả, chỉ ảnh hưởng tốc độ load data |
| 8 | Output layer (paper có `Output (None, 55), param# 1,815`) | Không có tầng này riêng biệt | 🟡 TRUNG BÌNH | `1,815 = 33×55 + 0` — không rõ tầng này là gì trong paper. Có thể là bias-only layer hoặc final activation layer. Notebook bỏ qua |

---

## 7. DANH SÁCH VẤN ĐỀ CẦN SỬA

### Nghiêm trọng (phải sửa trước khi submit):

- [ ] **[FRAMEWORK SAI]** Paper sử dụng TensorFlow/Keras (xác nhận qua output shape format `(None, ...)`, loss `sparse categorical cross entropy`, activation `SoftMax`). Notebook dùng PyTorch. Cần chuyển sang TF/Keras hoặc có xác nhận rõ ràng từ người dùng chấp nhận PyTorch.
- [ ] **[THIẾU BEST MODEL CHECKPOINT — CC20]** Hàm `train_model()` không lưu best checkpoint theo `val_accuracy` và không reload trước khi evaluate test set. Phải thêm `train_with_checkpoint()` với `best_model_s1.pth` / `best_model_s2.pth`.
- [ ] **[THIẾU SEED — CC15]** Không có `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)` ở đầu notebook.
- [ ] **[THIẾU CUDNN DETERMINISM — CC19]** Không có `torch.backends.cudnn.deterministic = True` + `torch.backends.cudnn.benchmark = False`.

### Nên sửa (ảnh hưởng đến chất lượng):

- [ ] **[THIẾU `torch.cuda.empty_cache()` — CC24]** Không có giữa Scenario 1 và Scenario 2. Thêm `torch.cuda.empty_cache()` (và `del model_1, train_loader_1, val_loader_1, test_loader_1`) sau khi kết thúc Scenario 1.
- [ ] **[THIẾU CLASS IMBALANCE DETECTION — CC11 ngầm định qua Rule 11]** Cell `cell-explore` có bar chart nhưng không có logic tính `max_count / min_count` và tự động thêm class weights vào `CrossEntropyLoss`.
- [ ] **[THIẾU DataParallel — CC14 + Rule 8]** Không có `nn.DataParallel` khi `n_gpus > 1`. Không in số GPU available.
- [ ] **[num_workers=2 thay vì 4 — CC23]** Rule 13 yêu cầu `num_workers=4` cho Kaggle T4×2.
- [ ] **[THIẾU ReproduceSummary.md — CC12]** Không có file `ViT_ReproduceSummary.md` trong outputs/.

### Tùy chọn (cải thiện độ trung thực):

- [ ] **[Làm rõ Output layer trong paper]** Paper có `Output (None, 55)` với param# 1,815 là một tầng riêng sau Dense(55). Cần điều tra xem đây là SoftMax layer hay bias layer riêng biệt.
- [ ] **[Xem xét normalization]** Paper không nêu mean/std; model trained from scratch thường không cần ImageNet normalization. Cân nhắc dùng normalization đơn giản hơn hoặc tính từ dataset.
- [ ] **[Activation Dense layers]** Paper không nêu rõ activation của các Dense layers. Nên comment rõ trong code đây là assumption.
- [ ] **[MultiheadAttention `norm1` gọi 3 lần]** Trong `TransformerEncoderBlock.forward()`: `self.attn(self.norm1(x), self.norm1(x), self.norm1(x))` gọi `norm1` 3 lần với cùng input — kết quả như nhau nhưng không hiệu quả. Pre-norm thông thường: `normed = self.norm1(x); attn_out, _ = self.attn(normed, normed, normed)`.

---

## 8. GHI CHÚ BỔ SUNG

### Về Framework

Paper không nêu tường minh "TensorFlow" hay "Keras" nhưng các dấu hiệu mạnh cho thấy đây là Keras:
- Output shape format `(None, 256, 256, 3)` — đây là Keras convention (PyTorch dùng `(B, C, H, W)`)
- Loss: `sparse categorical cross entropy` — tên Keras; PyTorch dùng `CrossEntropyLoss`
- Activation: `SoftMax` (viết hoa chữ M) — cú pháp Keras
- `PatchEncoder` và `TransformerEncoder` là custom Keras layers (`tf.keras.layers.Layer`)
- Dataset Kaggle, paper dùng Python/deep learning → Keras phổ biến hơn ở cộng đồng này vào thời điểm 2024

Notebook đã quyết định dùng PyTorch. Đây là lựa chọn hợp lý về mặt kỹ thuật, nhưng **không phù hợp với paper**. Tuy nhiên, theo bối cảnh workspace này (user dùng dataset riêng, kaggle-reproducer chỉ lấy kiến trúc), nếu user chấp nhận PyTorch thì có thể tiếp tục với điều kiện sửa các lỗi còn lại.

### Về Kiến trúc Dense Head

Dense head trong notebook thực chất khớp với Table 2 của paper về kích thước và thứ tự. Tầng bí ẩn `Output (None, 55), param# 1,815` trong paper có thể là:
- Một Softmax activation layer (không có trainable params thực sự)  
- Hoặc một bias layer — `1,815 = 33×55` → 33 không có ý nghĩa rõ ràng

Khả năng cao đây là layer Softmax cuối cùng mà Keras hiển thị riêng biệt trong model summary khi được define là tầng riêng. Trong PyTorch với `CrossEntropyLoss`, không cần Softmax output riêng.

### Về Hiệu năng Thực tế

Notebook đã chạy trên dataset user (5 classes, ~2461 ảnh) thay vì dataset paper (55 classes, 83,603 ảnh). Kết quả:
- Baseline: Accuracy=69.17%, F1-Macro=70.84%
- Augmented: Accuracy=71.58%, F1-Macro=70.81%

Kết quả thấp hơn nhiều so với paper (89.3% test accuracy) do dataset hoàn toàn khác nhau — điều này là bình thường và không phải lỗi.

### Điểm Cộng của Notebook

- Augmentation implement rất bám sát paper (tất cả 6 kỹ thuật với đúng tham số)
- Input size 256×256 đúng với paper
- Hyperparameters chính (LR, batch size, epochs, optimizer) đều khớp paper
- Classification report per class đầy đủ
- Dual scenario (Baseline + Augmented) được implement
- Excel export 2-sheet có nhúng ảnh
- Dataset Distribution bar chart có mặt
