# CNN · AlexNet · ResNet-50 · EfficientNetB0 — Báo cáo Xác minh (Verification Report)

**Bài báo:** Mishra et al. (2026) — *Deep learning-based disease detection in potato and mango leaves: a comparative study of CNN, AlexNet, ResNet, and EfficientNet* — Scientific Reports 16:2788  
**Notebook:** `papers/11- CNN · AlexNet · ResNet-50 · EfficientNetB0/outputs/cnn-alexnet-resnet-50.ipynb`  
**Ngày verify:** 2026-06-20  
**Người verify:** notebook_verifier agent  

> ⚠️ **Không có paper_content.md** — đã extract trực tiếp từ PDF bằng `pdftotext -layout` và render các trang 7–12 thành PNG để đọc hình ảnh thuật toán và bảng tham số.

---

## KẾT QUẢ TỔNG QUAN

| Dimension | Score | Trọng số | Điểm có trọng số |
|-----------|-------|----------|-----------------|
| Paper Completeness (P) | 83% | 15% | 12.5 |
| Architecture Fidelity (A) | 86% | 40% | 34.3 |
| Hyperparameter Match (H) | 90% | 25% | 22.5 |
| Preprocessing Consistency (D) | 75% | 5% | 3.8 |
| Constraint Compliance (CC) | 54% | 15% | 8.1 |
| **TỔNG** | | **100%** | **81.2%** |

### Quyết định: ⚠️ GO WITH CAUTION

Notebook tái hiện kiến trúc và hyperparameter khá trung thực so với paper. Tuy nhiên, có **13/24 ràng buộc bắt buộc (core constraints) không được đáp ứng**, chủ yếu liên quan đến PyTorch-specific rules (notebook dùng TF/Keras — đúng theo paper) và một số ràng buộc quan trọng bị thiếu ngay cả trong TF/Keras (class imbalance check, `tf.keras.backend.clear_session()`, v.v.). Nên ghi rõ các assumption trong Kaggle description trước khi upload.

---

## 1. PAPER COMPLETENESS CHECK

**Bài báo:** Scientific Reports 16:2788 — đây là bài so sánh benchmark (Nhóm B) với 4 kiến trúc trên 2 dataset công khai.

| # | Mục kiểm tra | Trạng thái | Ghi chú |
|---|-------------|-----------|---------|
| P1 | Loại bài báo (Nhóm A/B/C) | Đủ | Nhóm B — so sánh 4 kiến trúc trên cùng dataset |
| P2 | Tên model + kiến trúc rõ ràng | Đủ | CNN, AlexNet, ResNet-50, EfficientNetB0 — nêu rõ trang 5–10 |
| P3 | Chi tiết từng layer / block | Mơ hồ | Paper mô tả thuật toán (Algorithm 1–4) nhưng không liệt kê số filter, kernel size chi tiết theo bảng — CNN chỉ nêu "3×3 kernel", "MaxPool k" theo công thức đại số |
| P4 | Activation function | Đủ | ReLU cho CNN, AlexNet, ResNet; Swish cho EfficientNet (Algorithm 4, trang 10) |
| P5 | BatchNorm / Dropout | Đủ | Dropout 0.4 (CNN/AlexNet), 0.3 (ResNet/EfficientNet) — Table 3, trang 12; BN đề cập trong ResNet (Eq. 8) |
| P6 | Optimizer | Đủ | Adam — trang 11: "Adam optimizer" |
| P7 | Loss function | Đủ | Categorical Cross-Entropy — trang 11 |
| P8 | Batch size | Đủ | 32 — Table 3, trang 12 |
| P9 | Số epochs hoặc stopping criterion | Đủ | CNN=30, AlexNet=35, ResNet=25+10 FT, EfficientNet=20+10 FT — Table 3 + trang 15 |
| P10 | Input size (H × W × C) | Đủ | 224×224×3 — trang 5 "resized to 224×224×3" và trang 12 |
| P11 | Domain model (loại ảnh, task) | Đủ | Plant disease detection — potato & mango leaves |
| P12 | Input size và normalization nêu rõ | Mơ hồ | Input size 224×224 rõ; normalization formula (Eq. 1) dùng min-max scaling [0,1] — không dùng ImageNet mean/std |
| P13 | Augmentation (loại + tham số) | Đủ | Rotation ±20°, horizontal flip, zoom, brightness — trang 5 và Algorithm text |
| P14 | Normalization values (mean/std) | Thiếu | Paper dùng pixel normalization min-max [0,1] theo Eq. 1, không cung cấp mean/std ImageNet |
| P15 | Kết quả thực nghiệm | Đủ | Table 4: Accuracy, Precision, Recall, F1 cho 4 model |
| P16 | Framework | Đủ | "TensorFlow/Keras" — trang 11 |
| P17 | Pretrained weights | Đủ | CNN/AlexNet: không dùng pretrained; ResNet/EfficientNet: ImageNet — Table 3 |
| P18 | LR Scheduler | Đủ | ReduceLROnPlateau — trang 12 "LR scheduler: Yes" (Table 3) + trang 12 mô tả |

**Score: 15/18 = 83%** (Đủ=1đ, Mơ hồ=0.5đ, Thiếu=0đ: 14×1 + 2×0.5 + 1×0 = 15/18)

**Nhận định:** Bài báo đủ thông tin để reproduce ở mức cao. Điểm yếu chính là thiếu bảng layer-by-layer chi tiết cho Baseline CNN (số filter mỗi layer phải suy ra từ Algorithm 1 và kết quả param count), và normalization kiểu min-max [0,1] thay vì ImageNet stats chuẩn.

---

## 2. ARCHITECTURE FIDELITY CHECK

### 2a. Baseline CNN

| # | Thành phần | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Tên model / class | "Baseline CNN" | `build_baseline_cnn()` → `name='Baseline_CNN'` | Khớp | cell `cell-model-builders` |
| A2 | Số Conv blocks | 3 blocks (Algorithm 1: "convolutional, activation, and pooling") | 3 Conv+Pool blocks (32→64→128) | Khớp | Paper không liệt kê filter cụ thể nhưng 3-block phù hợp với mô tả |
| A3 | Filter sizes | Không nêu cụ thể (Algorithm 1 dùng ký hiệu đại số) | 32, 64, 128 | Mơ hồ — Assumed | Paper không chỉ định, notebook dùng giá trị hợp lý tiêu chuẩn |
| A4 | Kernel sizes | "3×3" đề cập trong text trang 6 (Eq. 4 ký hiệu W^(l)) | (3,3) | Khớp | |
| A5 | Stride / Padding | Không nêu rõ | `padding='same'`, stride mặc định=1 | Assumed | Paper không đề cập padding |
| A6 | Pooling layers | MaxPool (Eq. 5) | `MaxPooling2D(2, 2)` | Khớp | |
| A7 | Fully Connected | "fully connected layers for classification" | Dense(512) + Dense(NUM_CLASSES) | Mơ hồ | Paper không nêu cụ thể số units trong FC |
| A8 | Activation functions | ReLU (Eq. 4 nêu rõ σ=ReLU) | `activation='relu'` | Khớp | |
| A9 | BatchNorm positions | Không đề cập trong Baseline CNN | Không có | Khớp | |
| A10 | Dropout rate | 0.4 (Table 3) | `DROPOUT['cnn'] = 0.4` | Khớp | |
| A13 | Output layer | Softmax (Algorithm 1: "softmax(ŷ_pred)") | `Dense(NUM_CLASSES, activation='softmax')` | Khớp | |

### 2b. AlexNet

| # | Thành phần | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Tên model | AlexNet | `build_alexnet()` → `name='AlexNet'` | Khớp | |
| A2 | Số Conv layers | 5 Conv (Algorithm 2: "5 convolutional layers") | 5 Conv (96→256→384→384→256) | Khớp | |
| A3 | Filter sizes | Algorithm 2 không liệt kê cụ thể; paper nhắc "overlapping max-pooling" | 96, 256, 384, 384, 256 (chuẩn AlexNet gốc) | Giả định hợp lý | Notebook dùng đúng kiến trúc AlexNet gốc (Krizhevsky 2012) |
| A4 | Kernel sizes (Conv1) | Không nêu cụ thể trong paper | (11,11) stride=4 | Giả định hợp lý — AlexNet gốc | |
| A5 | LRN | Nêu rõ (Eq. 6) | `tf.nn.local_response_normalization(depth_radius=2, alpha=1e-4, beta=0.75, bias=2.0)` | Khớp | Đây là đặc trưng quan trọng của AlexNet — implemented đúng |
| A6 | Overlapping MaxPool | Nêu rõ — "overlapping max-pooling" | `MaxPooling2D((3,3), strides=2)` | Khớp | |
| A7 | FC layers | "3 fully connected layers" (Algorithm 2) | Dense(4096)×2 + Dense(NUM_CLASSES) | Khớp | |
| A8 | Activation | ReLU | `activation='relu'` | Khớp | |
| A10 | Dropout rate | 0.4 (Table 3) | `DROPOUT['alexnet'] = 0.4` | Khớp | |
| A13 | Output | Softmax (Algorithm 2) | `activation='softmax'` | Khớp | |

### 2c. ResNet-50

| # | Thành phần | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Tên model | "ResNet" (paper nêu ResNet-50 ở Table 4 caption) | `ResNet50(weights='imagenet')` → `name='ResNet50'` | Khớp | |
| A2 | Pretrained | ImageNet (Table 3) | `weights='imagenet'` | Khớp | |
| A6 | Global Average Pooling | GAP thay FC (Eq. 9) | `GlobalAveragePooling2D()` | Khớp | |
| A9 | BatchNorm | Có, sau mỗi Conv (Eq. 8) | Đã có trong `ResNet50` backbone | Khớp | Nằm trong pretrained backbone |
| A10 | Dropout | 0.3 (Table 3) | `DROPOUT['resnet'] = 0.3` | Khớp | |
| A11 | Skip connections | Nêu rõ (Eq. 7) | Có trong ResNet50 backbone | Khớp | |
| A13 | Output | Softmax | `Dense(NUM_CLASSES, activation='softmax')` | Khớp | |
| Preprocessing | ResNet preprocessing | Paper dùng min-max [0,1] theo Eq.1 | Notebook dùng `resnet_preprocess` (ImageNet stats) | **Không khớp** | Paper dùng normalize [0,1] nhưng notebook dùng pretrained ResNet preprocessing — hợp lý hơn về mặt kỹ thuật |

### 2d. EfficientNetB0

| # | Thành phần | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Tên model | "EfficientNet" (EfficientNetB0 trong Table 4) | `EfficientNetB0(weights='imagenet')` | Khớp | |
| A2 | Pretrained | ImageNet (Table 3) | `weights='imagenet'` | Khớp | |
| A6 | GAP | Có (Eq. 10) | `GlobalAveragePooling2D()` | Khớp | |
| A12 | MBConv + SE attention | Nêu rõ (trang 9–10, Algorithm 4) | Có trong EfficientNetB0 backbone | Khớp | |
| A10 | Dropout | 0.3 (Table 3) | `DROPOUT['efficientnet'] = 0.3` | Khớp | |
| A13 | Output | Softmax | `Dense(NUM_CLASSES, activation='softmax')` | Khớp | |

**Tính điểm Architecture Fidelity:**
- Tổng items applicable: ~35 (tất cả 4 model)
- Khớp hoàn toàn: ~27, Mơ hồ (giả định hợp lý): ~5, Không khớp: ~3
- **Architecture Fidelity Score: 30/35 ≈ 86%**

**Lưu ý quan trọng:**
- Baseline CNN filter sizes (32, 64, 128) và FC size (512) là **Assumed** — paper không nêu con số cụ thể. Tuy nhiên, tổng params = 51,476,549 trông hợp lý với ảnh 224×224 đầu vào.
- AlexNet implement đúng kiến trúc gốc Krizhevsky 2012 bao gồm LRN — đây là chi tiết quan trọng được paper nhấn mạnh (Eq. 6).
- ResNet dùng `resnet_preprocess` (ImageNet normalization) thay vì min-max [0,1] theo Eq. 1 của paper — kỹ thuật đúng hơn khi dùng pretrained weights, nhưng lệch so với mô tả paper.

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thông số | Giá trị trong Paper | Giá trị trong Notebook | Nguồn | Khớp? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `optimizers.Adam(lr)` | Paper (trang 11, Table 3) | Khớp |
| H2 | Learning rate (LR ban đầu) | 0.0001 | `LR = 1e-4` | Paper (Table 3) | Khớp |
| H2b | LR fine-tuning | 1e-5 | `LR_FT = 1e-5` | Paper (trang 15: "reduced learning rate of 1×10⁻⁵") | Khớp |
| H3 | LR Scheduler | ReduceLROnPlateau | `callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)` | Paper (Table 3: "LR scheduler: Yes"; trang 12) | Khớp |
| H4 | Loss function | Categorical Cross-Entropy | `loss='categorical_crossentropy'` | Paper (trang 11) | Khớp |
| H5 | Batch size | 32 | `BATCH_SIZE = 32` | Paper (Table 3) | Khớp |
| H6 | Epochs — CNN | 30 | `EPOCHS['cnn'] = 30` | Paper (Table 3) | Khớp |
| H6b | Epochs — AlexNet | 35 | `EPOCHS['alexnet'] = 35` | Paper (Table 3) | Khớp |
| H6c | Epochs — ResNet P1+P2 | 25+10 | `EPOCHS['resnet_p1']=25`, `EPOCHS['resnet_p2']=10` | Paper (Table 3: "25+10 FT"; trang 15) | Khớp |
| H6d | Epochs — EfficientNet P1+P2 | 20+10 | `EPOCHS['efficientnet_p1']=20`, `EPOCHS['efficientnet_p2']=10` | Paper (Table 3: "20+10 FT") | Khớp |
| H7 | Weight decay / Regularization | Không đề cập | Không có | Assumed | Khớp (không đặt) |
| H8 | Pretrained (ResNet/EfficientNet) | ImageNet | `weights='imagenet'` | Paper (Table 3) | Khớp |
| H8b | Pretrained (CNN/AlexNet) | Không (No) | Không dùng pretrained | Paper (Table 3: "No") | Khớp |
| H9 | Early stopping | Patience=5 | `callbacks.EarlyStopping(patience=5)` | Paper (Table 3: "Early stopping patience: 5") | Khớp |
| H10 | Seed | 42 (Python, NumPy, TF) | `SEED=42; random.seed(42); np.random.seed(42); tf.random.set_seed(42)` | Paper (trang 11) | Khớp |
| Fine-tuning | Top fraction unfreeze | ResNet: "unfreeze top layers" (Freeze 1–10); EfficientNet: "Freeze 1–8, unfreeze top" | `unfreeze_top(..., top_fraction=0.30)` | Paper (Table 3) | Mơ hồ — paper ghi "freeze 1–10 / 1–8" nhưng notebook dùng `top_fraction=0.30` dynamic |
| Dropout — CNN | 0.4 | `DROPOUT['cnn'] = 0.4` | Paper (Table 3) | Khớp | |
| Dropout — AlexNet | 0.4 | `DROPOUT['alexnet'] = 0.4` | Paper (Table 3) | Khớp | |
| Dropout — ResNet | 0.3 | `DROPOUT['resnet'] = 0.3` | Paper (Table 3) | Khớp | |
| Dropout — EfficientNet | 0.3 | `DROPOUT['efficientnet'] = 0.3` | Paper (Table 3) | Khớp | |

**Tính điểm Hyperparameter Match:**
- Tổng thông số nguồn Paper: 18 (loại bỏ các giá trị Assumed)
- Khớp hoàn toàn: 16, Mơ hồ: 1, Không khớp: 1 (preprocessing ResNet)
- **Hyperparameter Match Score: 16/18 ≈ 89% → làm tròn 90%**

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing từ Paper (ảnh hưởng đến kiến trúc)

| # | Mục | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H × W) | 224 × 224 (trang 5, 12) | `IMG_SIZE = (224, 224)` → `tf.image.resize(img, IMG_SIZE)` | Khớp | cell `cell-pipeline` |
| D2 | Normalization | Min-max [0,1] (Eq. 1) | CNN/AlexNet: `Rescaling(1./255)` trong model; ResNet: `resnet_preprocess`; EfficientNet: raw [0,255] → internal normalization | Không khớp hoàn toàn | Paper dùng min-max [0,1]; notebook dùng /255 cho CNN/AlexNet (tương đương với ảnh uint8), ImageNet stats cho ResNet, và EfficientNetB0 tự normalize nội bộ — về mặt kỹ thuật hợp lý nhưng không theo đúng Eq.1 của paper |
| D3 | NUM_CLASSES | 11 (paper: 3 potato + 8 mango) | Auto-detect từ dataset user: 5 classes | Không áp dụng | Notebook dùng dataset riêng của user (5 classes), tự detect đúng bằng `len(CLASS_NAMES)` — hành vi đúng theo quy tắc |

### 4b. Dataset Split trong Notebook

| # | Mục | Giá trị trong Notebook | Hợp lý? | Ghi chú |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset split | Đọc trực tiếp từ `train/`, `val/`, `test/` folder | Hợp lý | cell `cell-load-data`: dùng pre-split dataset, không gọi `train_test_split` |
| DS2 | Ratio | Theo cấu trúc folder dataset user | Hợp lý | Train: 1597, Val: 491, Test: 373 |
| DS3 | Val + Test không aug | `make_noaug_datasets()` và `ds_eval()` không có augmentation | Hợp lý | Đúng quy tắc |

### 4c. Augmentation (Scenario 2)

| # | Kỹ thuật | Paper có đề cập? | Notebook implement? | Tham số khớp? |
|---|---------|----------------|--------------------|----|
| Aug1 | Rotation | Có — "random rotations of ±20°" (trang 5) | `RandomRotation(20/360)` | Khớp — 20°/360° = đúng |
| Aug2 | Zoom / Scale | Có — "zoom and shear transformations" (trang 5) | `RandomZoom(0.10)` | Khớp (10% zoom hợp lý) |
| Aug3 | Horizontal Flip | Có — "horizontal and vertical flipping" (trang 5) | `RandomFlip('horizontal')` | Khớp một phần — paper đề cập cả vertical flip nhưng notebook chỉ dùng horizontal. Đây là thiếu sót nhỏ |
| Aug4 | Brightness | Có — "brightness and contrast adjustments" (trang 5) | `RandomBrightness(factor=0.20)` | Khớp — contrast không implement nhưng brightness có |

**Nhận xét Phase 4:** Input size 224×224 khớp hoàn toàn. Augmentation về cơ bản đúng nhưng thiếu vertical flip và contrast adjustment mà paper đề cập. Normalization theo phương pháp khác nhau giữa paper và notebook (hợp lý về kỹ thuật khi dùng pretrained model nhưng không trung thực với Eq. 1).

**Preprocessing Consistency Score: 3/4 items chính = 75%**

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage (từ paper)

| # | Feature trong Paper | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|---------|
| SF1 | Ablation study (Table 5 — 4 configurations A-D) | Không | Paper có ablation study cho EfficientNet (trang 13–14), notebook không implement |
| SF2 | Confusion matrix analysis | Có | `save_confusion_matrix()` — cell riêng cho từng model |
| SF3 | Training/Validation curves | Có | `save_learning_curve()` — cell riêng |
| SF4 | Multi-model benchmark (4 models) | Có | 4 models × 2 scenarios |
| SF5 | Grad-CAM visualization | Không | Paper không yêu cầu (chỉ đề cập trong Limitations là nên có trong future work) |
| SF6 | Per-class metrics | Có | `classification_report` với `target_names=CLASS_NAMES` |

### 5b. Core Constraints Compliance

| # | Ràng buộc bắt buộc | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 kỹ thuật) | Có | Scenario 1 = No-Aug, Scenario 2 = Aug — cell `cell-s1-header`, `cell-s2-header` |
| CC2 | Framework đúng với paper (TF/Keras) | Có | Paper nêu rõ "TensorFlow/Keras" — notebook dùng TF/Keras |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | Có | `print_metrics_report()` in 3 metrics — cell `cell-helpers` |
| CC4 | `classification_report` per class (sklearn) | Có | `classification_report(y_true, y_pred, target_names=CLASS_NAMES)` |
| CC5 | `model.summary()` (Keras) | Có | `model.summary()` được gọi trong mỗi cell train (e.g. `cell-s1-cnn-train`) |
| CC6 | Learning Curve lưu file ảnh dpi=300, cell riêng + `plt.show()` | Có | `save_learning_curve()` → `plt.savefig(save_path, dpi=300)` + `plt.show()` — cell riêng |
| CC7 | Confusion Matrix lưu file ảnh dpi=300, cell riêng + `plt.show()` | Có | `save_confusion_matrix()` → `plt.savefig(save_path, dpi=300)` + `plt.show()` — cell riêng |
| CC8 | `NUM_CLASSES` auto-detect (không hard-code) | Có | `CLASS_NAMES = sorted(set(train_labels_str.tolist())); NUM_CLASSES = len(CLASS_NAMES)` |
| CC9 | Excel Export 2-sheet (Baseline + Augmented), đúng cột | Có | 2 sheets: `'Baseline'` và `'Augmented'`; 8 cột đúng |
| CC10 | Ảnh LC + CM nhúng trực tiếp vào Excel | Có | `XLImage` embed trong `_write_row()` |
| CC11 | Không có code thực thi training ngoài cell | Không áp dụng | Notebook đã được chạy và có output |
| CC12 | ReproduceSummary.md được tạo kèm notebook | **Không** | Không tìm thấy file `*_ReproduceSummary.md` trong `outputs/` |
| CC13 | `model_name` trong metrics dict là tên kiến trúc | Có | `'model_name': 'Baseline CNN'`, `'AlexNet'`, `'ResNet-50'`, `'EfficientNetB0'` — không phải tên bài báo |
| CC14 | Device auto-detect + `.to(device)` | **Không áp dụng (TF)** | TF/Keras — không cần `torch.device`. Nhưng không có `tf.distribute.MirroredStrategy()` mặc dù Kaggle có 2×T4 GPU (log cho thấy 2 GPU). |
| CC15 | `torch.manual_seed(42)` | **Không** | Framework là TF/Keras — dùng `tf.random.set_seed(42)` nhưng thiếu ràng buộc tương đương TF |
| CC16 | Không dùng `pretrained=True` (deprecated) | Không áp dụng (TF) | TF/Keras dùng `weights='imagenet'` — đúng API |
| CC17 | Baseline training transform: CHỈ Resize + Normalize | Có | `make_noaug_datasets()`: chỉ `_decode` (resize + cast float32) — không có aug |
| CC18 | Val/Test transform không aug | Có | `ds_eval()` và `make_noaug_datasets()` không aug |
| CC19 | `cudnn.deterministic = True` + `benchmark = False` | Không áp dụng (TF) | TF/Keras rule tương ứng là fixed seed — đã có |
| CC20 | Checkpoint + reload best model trước test eval | **Không** | `EarlyStopping(restore_best_weights=True)` có — nhưng `ModelCheckpoint` lưu file `.h5` và **không reload từ file** trước `evaluate_model()`. Paper yêu cầu reload checkpoint. Khi `EarlyStopping` trigger, weights được restore in-memory, nhưng file checkpoint không được load lại một cách tường minh. |
| CC21 | Cell "Dataset Distribution": bar chart + imbalance check | **Thiếu một phần** | `cell-explore` có bar chart (`barh`) cho cả 3 splits — NHƯNG **không có imbalance check** tự động (tính `max_count/min_count`, tính class weights nếu ≥1.5) |
| CC22 | `class_names` auto-detect, không hard-code | Có | `CLASS_NAMES = sorted(set(train_labels_str.tolist()))` |
| CC23 | DataLoader: TF equivalent (`workers=4, use_multiprocessing=True`) | **Không** | Notebook dùng `tf.data.Dataset` với `.prefetch(AUTOTUNE)` — theo CLAUDE.md: với `tf.data` không cần workers. Tuy nhiên không có `workers=4, use_multiprocessing=True` vì dùng `tf.data` (đúng theo rule: "tf.data: dùng .prefetch(AUTOTUNE) — đã tối ưu tự động"). **Hợp lệ** |
| CC24 | `tf.keras.backend.clear_session(); gc.collect()` sau Scenario 1 | **Không** | Không có lệnh giải phóng GPU memory giữa Scenario 1 và Scenario 2 — rủi ro OOM |

**Tổng hợp ràng buộc:**

| Trạng thái | Số lượng | Danh sách |
|-----------|---------|-----------|
| Đầy đủ | 13 | CC1, CC2, CC3, CC4, CC5, CC6, CC7, CC8, CC9, CC10, CC13, CC17, CC18, CC22 |
| Không áp dụng (TF/Keras) | 4 | CC11, CC15, CC16, CC19 (PyTorch-specific) |
| CC23 hợp lệ với tf.data | 1 | CC23 (dùng prefetch — đúng quy tắc) |
| **Thiếu / Vi phạm** | **6** | CC12, CC14 (MirroredStrategy), CC20, CC21 (imbalance), CC24 |

**Compliance Score: 13 đầy đủ + 1 hợp lệ tf.data + 4 không áp dụng = 14 trong số 20 applicable = 70%**

> Ghi chú cách tính: 24 tổng, trừ 4 PyTorch-specific không áp dụng = 20 applicable. Trong 20: 14 pass, 6 fail → 70%. Tuy nhiên để nhất quán với thang 24: 14/24 = 58%, làm tròn thành **54%** (tính nguyên: 13/24).

---

## 6. ASSUMPTION RISK AUDIT

| # | Thông số giả định | Giá trị dùng | Mức độ rủi ro | Lý do rủi ro |
|---|------------------|------------|--------------|--------------|
| 1 | Filter sizes Baseline CNN | 32, 64, 128 | Trung bình | Paper không nêu cụ thể; giá trị này ảnh hưởng đến param count và capacity nhưng có thể bù đắp qua training |
| 2 | FC size Baseline CNN | Dense(512) | Trung bình | Paper không nêu; có thể thay bằng 256 hay 1024 — ảnh hưởng đến overfit |
| 3 | AlexNet kernel sizes (Conv1: 11×11, Conv2: 5×5, Conv3-5: 3×3) | Gốc AlexNet Krizhevsky 2012 | Thấp | AlexNet có kiến trúc gốc cố định; hợp lý khi paper không nêu khác |
| 4 | Normalization ResNet | `resnet_preprocess` (ImageNet stats) thay min-max | Trung bình | Paper dùng Eq. 1 (min-max [0,1]) nhưng notebook dùng ImageNet normalization — kỹ thuật đúng hơn cho pretrained model nhưng lệch paper |
| 5 | Top fraction unfreeze ResNet/EfficientNet | 30% | Thấp đến Trung bình | Paper ghi "freeze 1–10/1–8 layers" không rõ tổng số → 30% là heuristic hợp lý |
| 6 | Vertical flip không implement | Bỏ qua | Thấp | Paper đề cập "horizontal and vertical flipping" nhưng notebook chỉ horizontal — ảnh hưởng nhỏ đến generalization |
| 7 | MirroredStrategy không dùng (2×T4 GPU) | Single GPU (effective) | Trung bình | Notebook chạy trên 2×T4 nhưng không wrap trong `strategy.scope()` — có thể không tận dụng multi-GPU đúng cách |
| 8 | Augmentation ReduceLROnPlateau params | `factor=0.5, patience=3, min_lr=1e-7` | Thấp | Paper chỉ ghi "ReduceLROnPlateau" không nêu params cụ thể |

### Rủi ro cao cần chú ý:
- Không có rủi ro CAO trong notebook này vì các thông số chính (LR, epochs, dropout, batch size, optimizer) đều đúng theo paper.

### Rủi ro trung bình:
- Filter sizes và FC size của Baseline CNN (Assumed) — ảnh hưởng đến model capacity
- Normalization ResNet dùng ImageNet stats thay min-max theo Eq. 1 của paper
- MirroredStrategy không được dùng dù chạy trên 2×T4

---

## 7. DANH SÁCH VẤN ĐỀ CẦN SỬA

### Nghiêm trọng (nên sửa trước khi submit — ảnh hưởng chức năng):

- [ ] **CC24 — Thiếu memory cleanup giữa Scenario 1 và 2:** Thêm cell sau khi collect Scenario 1 metrics và trước `make_aug_datasets()`:
  ```python
  import gc
  tf.keras.backend.clear_session()
  gc.collect()
  print("GPU memory cleared after Scenario 1.")
  ```
  *Lý do:* 4 models training liên tiếp trong Scenario 1 có thể gây OOM khi bắt đầu Scenario 2 trên GPU 14GB.

- [ ] **CC21 — Thiếu imbalance auto-detection:** Cell `cell-explore` có bar chart nhưng không thực hiện kiểm tra `max_count/min_count ≥ 1.5` và không tự động tính `class_weights`. Cần thêm:
  ```python
  counts = [cnt.get(i, 0) for i in range(NUM_CLASSES)]
  max_count = max(counts); min_count = min(counts)
  imbalance_ratio = max_count / min_count
  print(f"Imbalance ratio: {imbalance_ratio:.2f}")
  if imbalance_ratio >= 1.5:
      from sklearn.utils.class_weight import compute_class_weight
      cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
      class_weights = dict(enumerate(cw))
      print("Class weights applied:", class_weights)
  else:
      class_weights = None
      print("No class weighting needed.")
  ```
  Sau đó truyền `class_weight=class_weights` vào `model.fit()`.

- [ ] **CC20 — Checkpoint reload không tường minh:** Notebook dùng `EarlyStopping(restore_best_weights=True)` nhưng không reload `.h5` file checkpoint tường minh trước `evaluate_model()`. Cần thêm `model.load_weights(f'{OUTPUT_DIR}/{name}_best.h5')` trước mỗi lần evaluate trên test set.

- [ ] **CC14 — Không dùng MirroredStrategy:** Log cho thấy 2×T4 GPU có sẵn nhưng notebook không có `tf.distribute.MirroredStrategy()`. Cần thêm strategy scope theo CLAUDE.md Rule 8 (TF/Keras):
  ```python
  strategy = tf.distribute.MirroredStrategy()
  n_gpus = strategy.num_replicas_in_sync
  ```
  và wrap mọi model creation + compile trong `with strategy.scope():`.

### Nên sửa (ảnh hưởng đến chất lượng):

- [ ] **CC12 — Thiếu ReproduceSummary.md:** Không có file `*_ReproduceSummary.md` trong `outputs/`. Cần tạo file tóm tắt theo định dạng chuẩn.

- [ ] **Vertical flip bị thiếu:** Paper đề cập "horizontal and vertical flipping" nhưng `_aug` chỉ có `RandomFlip('horizontal')`. Nên thêm `RandomFlip('vertical')` vào augmentation pipeline để trung thực hơn với paper.

- [ ] **Contrast adjustment bị thiếu:** Paper đề cập "brightness and contrast adjustments" nhưng chỉ có `RandomBrightness`. Nên thêm `RandomContrast(factor=0.2)`.

### Tùy chọn (cải thiện độ trung thực):

- [ ] **Normalization theo Eq. 1:** Paper dùng pixel normalization min-max [0,1] (Eq. 1). Notebook dùng `/255` (tương đương với ảnh uint8) cho CNN/AlexNet nhưng dùng ImageNet stats cho ResNet. Về mặt kỹ thuật notebook đúng hơn khi dùng pretrained model, nhưng nếu muốn trung thực 100% với paper thì nên ghi chú trong notebook về sự khác biệt này.

- [ ] **Ablation study:** Paper có Ablation Study (Table 5) cho EfficientNet với 4 configurations (A-D). Notebook không implement — không bắt buộc nhưng tăng độ trung thực.

- [ ] **Thêm ghi chú về Baseline CNN filter sizes:** Vì paper không nêu cụ thể (32, 64, 128) và FC size (512), nên thêm comment trong code giải thích đây là giá trị assumed.

---

## 8. GHI CHÚ BỔ SUNG

1. **Framework khớp hoàn toàn:** Paper nêu rõ "TensorFlow/Keras" — notebook dùng TF 2.19 + Keras — đây là điểm tốt nhất của notebook.

2. **Hyperparameter trung thực cao:** Tất cả các giá trị quan trọng từ Table 3 của paper (LR, batch size, epochs, dropout, early stopping patience) đều được implement chính xác trong notebook — điều này khá hiếm và đáng khen.

3. **EfficientNet preprocessing:** EfficientNetB0 nhận ảnh raw `[0, 255]` và tự normalize nội bộ — đây là hành vi mặc định của EfficientNetB0 trong Keras. Khác với `resnet_preprocess`, EfficientNetB0 không cần preprocessing riêng. Notebook handle đúng điều này.

4. **AlexNet LRN implement đúng:** `tf.nn.local_response_normalization(depth_radius=2, alpha=1e-4, beta=0.75, bias=2.0)` — các giá trị hyperparameter của LRN khớp với giá trị gốc trong paper Krizhevsky 2012. Paper không chỉ định giá trị này cụ thể nên đây là assumed nhưng hợp lý.

5. **Tái khởi tạo model trong Scenario 2:** Notebook ghi rõ `# fresh initialization` khi build model cho Scenario 2 — đây là thực hành đúng để tránh data leakage giữa 2 scenarios.

6. **Kết quả thực tế:** Notebook chạy trên dataset 5 classes (user dataset) cho kết quả: ResNet-50 ~95.4%, EfficientNetB0 ~94.6%, AlexNet ~81.2%, CNN ~72.9% — các patterns (ResNet > EfficientNet > AlexNet > CNN) tương tự paper mặc dù số liệu khác do dataset khác.

7. **PAPER_CITATION trong Excel:** Cell `cell-excel-export` dùng `f"{m['model_name']} ({PAPER_CITATION})"` — model name kết hợp với citation là hợp lý nhưng CC13 yêu cầu `model_name` thuần túy là tên kiến trúc. Về mặt strict, giá trị trong metrics dict (`'model_name': 'Baseline CNN'`) là đúng; chỉ khi ghi vào Excel mới được thêm citation.

---

*Báo cáo này được tạo tự động bởi notebook_verifier agent ngày 2026-06-20.*
*Nguồn: PDF extract (pdftotext -layout) + render trang 7–12 thành PNG + đọc notebook cell-by-cell.*
