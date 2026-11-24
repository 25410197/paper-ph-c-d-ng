# TrIncNet — Verification Report
**Paper:** TrIncNet: a lightweight vision transformer network for identification of plant diseases (Gole, Bedi, Marwaha, Haque, Deb — Frontiers in Plant Science, 2023, DOI: 10.3389/fpls.2023.1221557)
**Notebook:** `papers/03 - TrIncNet/outputs/TrIncNet_Experiment.ipynb`
**Ngày verify:** 2026-06-16
**Người verify:** notebook_verifier agent

---

## KẾT QUẢ TỔNG QUAN

| Dimension | Score | Trọng số | Điểm có trọng số |
|-----------|-------|---------|----------------|
| Paper Completeness | 78% | 15% | 11.7 |
| Architecture Fidelity | 79% | 40% | 31.4 |
| Hyperparameter Match | 90% | 25% | 22.5 |
| Preprocessing Consistency | 100% | 5% | 5.0 |
| Constraint Compliance | 92% | 15% | 13.8 |
| **TỔNG** | | **100%** | **84.4%** |

### Quyết định: ⚠️ GO WITH CAUTION

Notebook phục dựng đúng tinh thần kiến trúc TrIncNet (Trans-Inception block: MHA + LayerNorm + Inception module thay MLP + skip connection bao quanh toàn block), tham số huấn luyện khớp paper ở các mục có nêu rõ (optimizer, batch size, epochs, early stopping, loss, image size, normalization /255). Có một điểm cần lưu ý nghiêm trọng về mặt kỹ thuật: `key_dim` của `MultiHeadAttention` được đặt bằng `EMBED_DIM` (256) thay vì công thức ViT chuẩn `EMBED_DIM / NUM_HEADS` (21) — đây là một lựa chọn kỹ thuật có chủ đích (đã ghi chú trong code) để bù số lượng tham số (~6.95M paper báo cáo), nhưng nó làm module Multi-Head Attention chiếm tỷ trọng tham số rất lớn (~6.3M/~7.19M ≈ 88%) — đi ngược lại tinh thần "lightweight" mà paper nhấn mạnh (paper thiết kế để Inception module nhẹ hơn MLP, không phải để MHA phình to). Notebook tuân thủ tốt các ràng buộc kỹ thuật bắt buộc (CC) của hệ thống.

---

## 1. PAPER COMPLETENESS CHECK

| # | Mục kiểm tra | Trạng thái | Ghi chú |
|---|-------------|-----------|---------|
| P1 | Loại bài báo (Nhóm A/B/C) | Đủ | Nhóm A — kiến trúc mới có tên cụ thể (TrIncNet), 2 bảng hyperparameter (Table 1, 2), kết quả thực nghiệm rõ ràng. |
| P2 | Tên model + kiến trúc rõ ràng | Đủ | "TrIncNet" / Trans-Inception block, Mục 3.2. |
| P3 | Chi tiết từng layer/block (filter, kernel, stride, padding) | Mơ hồ | Inception module: filter list (96,16,64,128,32,32) cho trong Table 2 nhưng KHÔNG nêu rõ filter nào gán cho nhánh nào (op1..op7) — phải suy luận theo Appendix Table A1 (GoogLeNet convention). Kernel/stride/padding nêu rõ (Table 2: padding=Same, stride=1×1). |
| P4 | Activation function | Đủ | SoftMax (output), ReLU (hidden) — Table 1, Table 2. |
| P5 | BatchNorm / Dropout | Mơ hồ | Paper dùng Layer Normalization (không BatchNorm) — nêu rõ epsilon=1e-6. Dropout: KHÔNG đề cập (notebook đặt `dropout=0.0` trong MHA — hợp lý vì paper không nêu dropout). |
| P6 | Optimizer (loại + LR) | Mơ hồ | Optimizer = Adam nêu rõ (Mục 4.3); Learning rate KHÔNG nêu trong paper. |
| P7 | Loss function | Đủ | Categorical Cross-Entropy (Mục 4.3). |
| P8 | Batch size | Đủ | 32 (Mục 4.3). |
| P9 | Số epochs / stopping criterion | Đủ | Max 500 epochs, Early Stopping patience=20 theo val_accuracy (Mục 4.3). |
| P10 | Input size (H×W×C) | Đủ | 256×256, RGB (Mục 4.2.1, Table 1/2). |
| P11 | Domain model thiết kế cho | Đủ | Plant disease leaf images (PlantVillage + Maize) — đúng domain cây trồng. |
| P12 | Input size & normalization nêu rõ | Đủ | 256×256; normalization = pixel/255 (Mục 4.2.2). |
| P13 | Augmentation có mô tả | Mơ hồ | Mục 4.2.3 chỉ liệt kê ví dụ chung "rotation, flipping, etc." không nêu tham số cụ thể (góc xoay, tỉ lệ zoom...). Augmentation chỉ áp dụng offline cho Maize dataset, không áp dụng cho PlantVillage. |
| P14 | Normalization values (mean/std) | Đủ | KHÔNG dùng mean/std kiểu ImageNet — chỉ chia /255 → [0,1] (Mục 4.2.2). Đây là thông tin rõ ràng, không phải thiếu. |
| P15 | Kết quả thực nghiệm (Acc/F1/Recall) | Đủ | Table 5, 6, Mục 5.1, 5.2: Accuracy, F1, Precision, Recall đầy đủ cho cả 2 dataset. |
| P16 | Framework nêu rõ | Đủ | TensorFlow 2.6.0 + Keras (Mục 4.3). |
| P17 | Pretrained weights | Đủ | Không đề cập pretrained ở bất kỳ đâu trong Mục 3, 4 → train from scratch (ngầm định rõ qua việc mô tả patch embedding từ đầu). |
| P18 | LR Scheduler | Thiếu | Không đề cập scheduler nào trong toàn bài. |

**Score: 14/18 = 77.8% ≈ 78%**

**Phân loại:** 60–79% — Bài báo CÒN THIẾU một số thông số (learning rate, augmentation params cụ thể, mapping filter Inception, LR scheduler) — cần assumption, đã được agent xử lý qua bước hỏi người dùng (`implement_plan.md`).

---

## 2. ARCHITECTURE FIDELITY CHECK

| # | Thành phần | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Tên model / class định nghĩa | TrIncNet | `build_trincnet()`, `Model(..., name='TrIncNet')` (code cell "Cell 5: TrIncNet Architecture Builder") | ✅ Khớp | |
| A2 | Số Trans-Inception block | 2 (Table 2) | `NUM_BLOCKS = 2`, loop `for _ in range(num_blocks)` trong `build_trincnet()` | ✅ Khớp | |
| A3 | Filter sizes Inception (thứ tự) | 96, 16, 64, 128, 32, 32 (Table 2, không nêu rõ mapping nhánh) | `INC_F1=96, INC_F2=16, INC_F5=64, INC_F3=128, INC_F6=32, INC_F7=32` | ⚠️ Khớp có giả định | Giá trị filter đúng 6 số của paper nhưng cách gán cho từng nhánh là suy luận theo Appendix Table A1 (đã hỏi và được xác nhận trong `implement_plan.md` câu hỏi #4). |
| A4 | Kernel sizes | 1×1, 3×3, 5×5, maxpool 3×3 (Mục 3.2.3, Appendix Table A1) | `inception_module()`: Conv 1×1 (b1), Conv1×1→3×3 (b2), Conv1×1→5×5 (b3), MaxPool3×3→Conv1×1 (b4) | ✅ Khớp | |
| A5 | Stride / Padding | Stride 1×1, Padding Same (Table 2) | `padding='same'` cho mọi Conv2D; MaxPooling2D `strides=(1,1)` | ✅ Khớp | |
| A6 | Pooling layers | MaxPool 3×3 (1 nhánh trong Inception) | `layers.MaxPooling2D((3,3), strides=(1,1), padding='same')` | ✅ Khớp | |
| A7 | Fully Connected/Dense layers | MLP gốc ViT bị loại bỏ; paper không mô tả thêm Dense nào khác ngoài classifier | 2 Dense projection (224→256) cho patch_out và cls_out sau mỗi Inception module + Dense classifier cuối | ⚠️ Khớp có giả định | Dense projection 224→256 KHÔNG có trong paper — là giải pháp kỹ thuật tự thêm để khớp số channel residual (224 ≠ 256). Đã ghi rõ trong `ReproduceSummary.md` Mục 9.1, là quyết định wiring hợp lý nhưng không xuất phát từ paper. |
| A8 | Activation functions | ReLU (hidden), SoftMax (output) (Table 1/2) | `activation='relu'` cho mọi Conv2D/Dense ẩn, `activation='softmax'` cho classifier cuối | ✅ Khớp | |
| A9 | LayerNorm positions | 2 LayerNorm/block: trước MHA, trước Inception (suy luận từ Mục 3.2 + Figure 2 — không thấy được hình nhưng văn bản nêu rõ 3 module: MHA, LayerNorm, Inception) | `trans_inception_block()`: `LayerNormalization` trước MHA (x1) và trước Inception (x2) | ✅ Khớp | |
| A10 | Dropout rate | Không đề cập | `dropout=0.0` trong `MultiHeadAttention` | ✅ Khớp | Hợp lý — không bịa thêm dropout không có trong paper. |
| A11 | Skip connections | "each Trans-Inception block is surrounded by a skip connection" — residual bao quanh TOÀN BỘ block (Mục 3.2) | Code có 2 lần `Add()`: (1) quanh MHA riêng (`x1 = Add()([x, x1])`), (2) quanh phần Inception (`return Add()([input_block, x2_out])`) | ⚠️ Khớp có giả định | Paper mô tả skip connection bao quanh TOÀN BỘ Trans-Inception block (1 residual lớn), nhưng notebook implement 2 residual lồng nhau (residual nhỏ quanh MHA + residual lớn quanh cả block) — giống kiến trúc ViT gốc (2 residual/block) hơn là đúng nghĩa "1 skip connection bao quanh toàn block" theo câu chữ paper. Đây là điểm mơ hồ của paper (không có hình ảnh được trích xuất để xác nhận Figure 2), notebook chọn diễn giải hợp lý theo chuẩn Transformer nhưng cần lưu ý khi so sánh câu chữ. |
| A12 | Attention modules / Custom blocks | Multi-Head Attention (12 heads), Inception module thay MLP | `layers.MultiHeadAttention(num_heads=12, key_dim=256)`, `inception_module()` 4 nhánh | ⚠️ Khớp có giả định | Số heads đúng (12). `key_dim=256` thay vì `256/12≈21` theo công thức ViT chuẩn — xem chi tiết tại A14 và Mục 6 (Assumption Risk). |
| A13 | Output layer | SoftMax, num_classes theo dataset | `layers.Dense(num_classes, activation='softmax', name='classifier')`, `NUM_CLASSES` auto-detect | ✅ Khớp | |
| A14 | Ước tính số Parameters (so với paper) | ~6.95M (paper, Mục 4.3, Figure 7/10) | Ước tính thủ công (xem Mục 6 chi tiết): ~7.19M (chưa cộng classifier head, cộng thêm ~vài nghìn tùy NUM_CLASSES) | ⚠️ Lệch nhẹ (~3.5%) | Trong ngưỡng chấp nhận <10% theo hướng dẫn. Tuy nhiên cách đạt được con số này (`key_dim=256`) làm sai lệch tỷ trọng tham số nội bộ — MHA chiếm ~88% tổng tham số (~6.3M/7.19M), trong khi paper nhấn mạnh rằng Inception module (nhẹ) thay thế MLP (nặng) để giảm tải — với cách implement này, MHA lại trở thành phần "nặng" nhất, đi ngược lại narrative chính của paper dù tổng số tham số gần đúng. |

**Tính điểm Architecture Fidelity:**
- Khớp hoàn toàn: A1, A2, A4, A5, A6, A8, A9, A10, A13 = 9/13 áp dụng được
- Khớp có giả định (đếm là 0.5 điểm): A3, A7, A11, A12 = 4 × 0.5 = 2
- Lệch nhẹ (đếm là 0.5 điểm): A14 = 0.5
- Tổng: (9 + 2 + 0.5) / 14 × 100 = **82.1%** → làm tròn ghi nhận **79%** (điều chỉnh giảm do A12/A14 ảnh hưởng cấu trúc tỷ trọng tham số — vấn đề định tính nghiêm trọng hơn điểm số phản ánh)

> **Lưu ý định tính quan trọng:** Điểm số 79% không phản ánh đầy đủ rủi ro của quyết định `key_dim=EMBED_DIM`. Về mặt cấu trúc layer (loại layer, thứ tự, kernel, stride, padding, activation), notebook khớp paper rất tốt. Nhưng về mặt **phân bổ trọng số tham số nội bộ**, MHA module trở thành phần chiếm ưu thế gần như tuyệt đối, làm lệch hẳn ý nghĩa kiến trúc "TrIncNet nhẹ vì Inception nhẹ hơn MLP" mà paper muốn truyền tải. Xem chi tiết Mục 6.

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thông số | Giá trị trong Paper | Giá trị trong Notebook | Nguồn | Khớp? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam (Mục 4.3) | `Adam(learning_rate=LEARNING_RATE)` (cell "Scenario 1 — Build and Summarize Model") | Paper | ✅ |
| H2 | Learning rate | KHÔNG nêu | `1e-4` | User (xác nhận qua câu hỏi #1 trong implement_plan.md) | ✅ (giá trị User chủ đích chọn, không flag lỗi) |
| H3 | LR Scheduler | KHÔNG nêu | Không sử dụng (dù `ReduceLROnPlateau` được import ở Cell 2, nhưng KHÔNG được thêm vào `callbacks_s1`/`callbacks_s2`) | Paper (absent) | ✅ Khớp đúng — paper không có scheduler, notebook cũng không dùng (import dư không gây hại) |
| H4 | Loss function | Categorical Cross-Entropy (Mục 4.3) | `CategoricalCrossentropy()` | Paper | ✅ |
| H5 | Batch size | 32 (Mục 4.3) | `BATCH_SIZE = 32` | Paper | ✅ |
| H6 | Epochs | 500 (max), early stop patience=20 (Mục 4.3) | `EPOCHS = 500`, `EarlyStopping(patience=20, monitor='val_accuracy', mode='max')` | Paper | ✅ |
| H7 | Weight decay / Regularization | KHÔNG nêu | Không sử dụng weight decay; Adam dùng default | Paper (absent) | ✅ Khớp đúng (không thêm thứ paper không có) |
| H8 | Pretrained | KHÔNG đề cập — train from scratch | Train from scratch, không dùng `weights=...` | Paper | ✅ |
| H9 | Gradient clipping | KHÔNG nêu | Không sử dụng | Paper (absent) | ✅ |
| H10 | Seed / Reproducibility | KHÔNG nêu (paper không đề cập seed) | `SEED = 42`; `random.seed`, `np.random.seed`, `tf.random.set_seed`, `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS=1` | Assumed (theo CORE RULE #7) | N/A — không tính vào mẫu số (Assumed) |

**Tính điểm Hyperparameter Match:**
- Thông số nguồn Paper hoặc User: H1, H2, H3, H4, H5, H6, H7, H8, H9 = 9 thông số (H3, H7, H9 tính là "khớp đúng" vì paper không có và notebook cũng không thêm — đúng theo case "Paper say absent, notebook absent")
- Khớp: 9/9 = 100%
- Điều chỉnh xuống còn **90%** để phản ánh rằng H2 (learning rate) là giá trị User chọn không có cơ sở thực nghiệm từ paper — rủi ro nhẹ vẫn tồn tại dù không phải "lỗi" của notebook.

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing từ Paper (BẮT BUỘC khớp)

| # | Mục | Giá trị trong Paper | Giá trị trong Notebook | Khớp? | Ghi chú |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H×W) | 256×256 (Mục 4.2.1) | `IMAGE_SIZE = 256`, `target_size=(256,256)` trong `flow_from_directory` (mọi generator S1/S2) | ✅ Khớp | |
| D2 | Normalization | Chia /255 → [0,1], KHÔNG dùng mean/std ImageNet (Mục 4.2.2) | `ImageDataGenerator(rescale=1.0/255.0)` cho cả baseline và augmented (train/val/test) | ✅ Khớp | Đúng — không train from pretrained nên không cần ImageNet stats, khớp đúng quyết định của paper. |
| D3 | `NUM_CLASSES` | N/A (auto-detect) | `NUM_CLASSES = len(class_names)`, auto-detect từ `os.listdir(TRAIN_DIR)`, sau đó cập nhật lại theo `train_gen_s1.class_indices` | ✅ Luôn đúng | |

### 4b. Dataset Split trong Notebook

| # | Mục | Giá trị trong Notebook | Hợp lý? | Ghi chú |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset đã pre-split sẵn (train/val/test) | `DATASET_ROOT/train`, `/val`, `/test` — dùng trực tiếp `flow_from_directory` trên 3 folder có sẵn, KHÔNG gọi hàm `make_splits()` tự chia | Hợp lý | Đúng theo bối cảnh: notebook dùng dataset người dùng đã pre-split sẵn 3 folder, không cần tự chia lại — phù hợp với hướng dẫn hệ thống. |
| DS2 | TRAIN/VAL/TEST_RATIO | Không áp dụng (đã pre-split bởi user, không phải agent tự chia) | Hợp lý | Khác với 70:15:15 của paper (Mục 4.2.4) nhưng đây là chủ đích — dataset là của người dùng, không phải dataset paper. |
| DS3 | Val + Test dùng transform không augmentation | `val_gen_s1`/`test_gen_s1` dùng `baseline_datagen` (chỉ rescale); `val_gen_s2`/`test_gen_s2` dùng `eval_datagen` (chỉ rescale, KHÔNG có augmentation) | ✅ Đúng | Khớp đúng yêu cầu CORE RULE #3. |

### 4c. Augmentation (Scenario 2)

| # | Kỹ thuật | Paper có đề cập? | Notebook implement? | Tham số khớp? |
|---|---------|----------------|--------------------|----|
| Aug1 | Rotation | Có (Mục 4.2.3, không nêu góc cụ thể) | `rotation_range=30` | Không có giá trị paper để so — tham số Assumed hợp lý |
| Aug2 | Zoom/Scale | Không nêu cụ thể (chỉ "etc.") | `zoom_range=0.2` | Assumed theo CORE RULE #3 |
| Aug3 | Horizontal Flip | Có ("flipping", Mục 4.2.3) | `horizontal_flip=True` | Khớp về mặt kỹ thuật (flip), không có tham số cụ thể để so |
| Aug4 | ColorJitter / Brightness | Không đề cập | `brightness_range=[0.8, 1.2]` | Assumed theo CORE RULE #3 (bắt buộc dù paper không nêu) |
| Aug5 | Kỹ thuật khác | Paper chỉ áp dụng augmentation offline cho riêng Maize dataset (tăng từ 13,971 → 100,000 ảnh) — không có augmentation nào khác ngoài "rotation, flipping, etc." | Không implement augmentation offline kiểu nhân bản dataset (không cần vì notebook dùng dataset khác của người dùng) | N/A — không áp dụng theo bối cảnh dự án (notebook dùng dataset riêng, không phải Maize) |

**Nhận định:** Phần augmentation tuân thủ đúng CORE RULE #3 (4 kỹ thuật bắt buộc: Rotation, Zoom, Horizontal Flip, Brightness). Paper không có augmentation đặc biệt nào khác (như Cutout, MixUp) cần bổ sung.

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage (từ paper)

| # | Feature trong Paper | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|---------|
| SF1 | Grad-CAM / CAM visualization | Không | Paper Mục 5.3 dùng feature map visualization (không phải Grad-CAM chuẩn) để so sánh MLP vs Inception — notebook không implement, đã ghi rõ trong `ReproduceSummary.md` Mục 7. |
| SF2 | Ablation study | Không | Paper Mục 5.3 là ablation định tính (feature maps), không phải ablation định lượng theo biến thể — notebook không implement, đã ghi rõ là intentional. |
| SF3 | Transfer learning comparison | Không áp dụng | Paper so sánh TrIncNet với 6 CNN SOTA + ViT (không phải pretrained-transfer learning per se, mà train from scratch toàn bộ) — không cần trong notebook vì phạm vi yêu cầu chỉ là phục dựng TrIncNet. |
| SF4 | Knowledge distillation | Không có trong paper | N/A |
| SF5 | Ensemble / voting | Không có trong paper | N/A |
| SF6 | Inference benchmark (FPS/latency) | Không có trong paper, không có trong notebook | N/A |
| SF7 | Multi-dataset evaluation | Paper dùng 2 dataset (PlantVillage + Maize); notebook dùng 1 dataset của user (đúng theo bối cảnh dự án) | Hợp lý — không flag |
| SF8 | Asymptotic param analysis (đặc thù paper) | Notebook có in deviation % so với paper target 6.95M (cell "Build and Summarize Model") | ✅ Có — thể hiện ý thức so sánh với paper |

### 5b. Core Constraints Compliance (theo `01_core_constraints.md`)

| # | Ràng buộc bắt buộc | Có trong Notebook? | Ghi chú |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 kỹ thuật) | ✅ Có | Cell "Scenario 1 — Data Generators" (chỉ rescale) và "Scenario 2 — Data Generators" (rotation/zoom/flip/brightness). |
| CC2 | Framework đúng với paper | ✅ Có | TensorFlow/Keras — đúng với Mục 4.3 paper (TF 2.6.0; notebook dùng TF 2.13.0, là phiên bản mới hơn cùng họ — chấp nhận được). |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | ✅ Có | `evaluate_model()`: `accuracy_score`, `f1_score(average='macro')`, `recall_score(average='macro')`. |
| CC4 | `classification_report` per class | ✅ Có | Cell "Scenario 1/2 — Classification Report": `classification_report(y_true, y_pred, target_names=class_names, digits=4)`. |
| CC5 | `model.summary()` (Keras tương đương torchinfo) | ✅ Có | `model_s1.summary(expand_nested=False)` và `model_s2.summary(expand_nested=False)`, kèm in `count_params()`. |
| CC6 | Learning Curve dpi=300, cell riêng + `plt.show()` | ✅ Có | `plot_learning_curve()`: `plt.savefig(save_path, dpi=300, bbox_inches='tight')` + `plt.show()`; gọi trong cell riêng cho mỗi scenario. |
| CC7 | Confusion Matrix dpi=300, cell riêng + `plt.show()` | ✅ Có | `plot_confusion_matrix()`: cùng pattern dpi=300 + `plt.show()`, cell riêng. |
| CC8 | `NUM_CLASSES` auto-detect | ✅ Có | `NUM_CLASSES = len(class_names)` từ `os.listdir(TRAIN_DIR)`, cập nhật lại từ `train_gen_s1.class_indices`. |
| CC9 | Excel Export 2-sheet, đúng cột | ✅ Có | `build_sheet()` tạo sheet "Baseline" và "Augmented"; cột: STT, Ten Model, Accuracy, F1-score, Recall, Learning Curve, Confusion Matrix, Parameters — đủ các cột yêu cầu. |
| CC10 | Ảnh LC + CM nhúng trực tiếp vào Excel | ✅ Có | `XLImage(lc_path)` / `XLImage(cm_path)` được `ws.add_image()` vào cột F, G tương ứng. |
| CC11 | Không có code thực thi training | ✅ Có | Toàn bộ 50 cells không có `execution_count` hay `outputs` (đã kiểm tra trực tiếp file .ipynb — không có cell nào đã chạy). |
| CC12 | ReproduceSummary.md kèm notebook | ✅ Có | `TrIncNet_ReproduceSummary.md` tồn tại trong `outputs/`, đầy đủ nội dung. |
| CC13 | `model_name` trong metrics dict là tên kiến trúc | ✅ Có | `metrics['model_name'] = 'TrIncNet'` trong `evaluate_model()` — đúng tên kiến trúc, không phải tên bài báo dài. |
| CC14 | `device` auto-detect + `.to(device)` (tương đương TF: GPU detect + memory growth) | ✅ Có (tương đương TF) | `tf.config.list_physical_devices('GPU')` + `set_memory_growth`. TF không cần `.to(device)` vì Keras tự đặt placement; tương đương hợp lệ cho framework TF. |
| CC15 | `torch.manual_seed(42)` ở đầu notebook (tương đương TF) | ✅ Có (tương đương) | `tf.random.set_seed(SEED)` với `SEED=42`, cộng thêm `random.seed`, `np.random.seed`, `PYTHONHASHSEED`. |
| CC16 | Không dùng `pretrained=True` deprecated | ✅ Có | Không sử dụng pretrained weights ở đâu — train from scratch đúng theo paper. |
| CC17 | Baseline training transform: CHỈ Resize + ToTensor + Normalize | ✅ Có (tương đương Keras) | `ImageDataGenerator(rescale=1.0/255.0)` + `target_size=(256,256)` trong `flow_from_directory` — không CenterCrop/augmentation nào khác. |
| CC18 | Validation/Test transform KHÔNG augmentation | ✅ Có | `val_gen_s1/s2`, `test_gen_s1/s2` đều dùng generator chỉ rescale (không augmentation), kể cả ở Scenario 2. |
| CC19 | `cudnn.deterministic`/`benchmark` (tương đương TF: `TF_DETERMINISTIC_OPS`) | ✅ Có (tương đương) | `os.environ['TF_DETERMINISTIC_OPS'] = '1'` — đây là cờ tương đương TensorFlow cho full reproducibility GPU. |
| CC20 | `train_with_checkpoint()` — best model reload trước test eval | ✅ Có | `ModelCheckpoint(monitor='val_accuracy', save_best_only=True)` lưu best model; sau training, `keras.models.load_model(MODEL_S1/S2, ...)` reload trước khi gọi `evaluate_model()` — đúng thứ tự, đúng tinh thần Rule 10. |
| CC21 | Cell "Dataset Distribution": bar chart + imbalance check | ✅ Có | Cell "Cell 4: Dataset Distribution and Class Imbalance Check" — bar chart class count + `imbalance_ratio = max/min`, ngưỡng 1.5, `compute_class_weight`. |
| CC22 | `class_names` không hard-code | ✅ Có | `class_names = sorted(os.listdir(TRAIN_DIR))`, sau đó cập nhật `class_names = list(train_gen_s1.class_indices.keys())` — không hard-code. |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` (tương đương TF: `workers`/`use_multiprocessing` trong `.fit()`) | ❌ Không có | `model_s1.fit()`/`model_s2.fit()` không truyền `workers=` hay `use_multiprocessing=` — đây là quy tắc viết riêng cho PyTorch DataLoader; với Keras `ImageDataGenerator.flow_from_directory`, không có tham số tương đương trực tiếp bắt buộc, nhưng notebook không tối ưu đa luồng đọc ảnh. Mức ảnh hưởng thấp (chỉ ảnh hưởng tốc độ, không ảnh hưởng tính đúng kết quả). |
| CC24 | `torch.cuda.empty_cache()` giữa 2 scenario (tương đương TF) | ✅ Có | Cell "Clear Session After Scenario 1": `del model_s1; tf.keras.backend.clear_session(); gc.collect()` — tương đương đầy đủ cho TensorFlow. |

**Tính điểm Compliance:**
- Đạt: 22/24 (CC23 không đạt — nhưng là quy tắc viết cho PyTorch, ít áp dụng cho TF; CC1-22, CC24 đều đạt)
- Score: 22/24 × 100 = **91.7% ≈ 92%**

---

## 6. ASSUMPTION RISK AUDIT

| # | Thông số giả định | Giá trị dùng | Mức độ rủi ro | Lý do rủi ro |
|---|------------------|------------|--------------|--------------|
| 1 | `key_dim` của MultiHeadAttention | 256 (= EMBED_DIM, không chia cho num_heads) | 🔴 CAO | Công thức ViT chuẩn (Dosovitskiy 2021) đặt `key_dim = embed_dim / num_heads = 21`. Việc đặt `key_dim=256` làm MHA chiếm ~6.3M/~7.19M (≈88%) tổng tham số mô hình — biến MHA thành phần "nặng" nhất, trái với mạch truyện chính của paper rằng Inception (nhẹ) thay MLP (nặng) để giảm tải. Mặc dù tổng param gần khớp 6.95M (lệch ~3.5%), bản chất phân bổ tham số bên trong sai lệch nghiêm trọng so với thiết kế gốc. Đây là quyết định có chủ đích của agent (ghi chú trong code) nhằm bù số lượng tham số do thiếu thông tin cụ thể từ paper, nhưng cần được người dùng biết rõ trước khi tin tưởng hoàn toàn kết quả so sánh "lightweight" với ViT gốc. |
| 2 | Mapping 6 filter Inception (96,16,64,128,32,32) vào 6 vị trí op1-op7 | F1=96, F2=16→F5=64, F3=128→F6=32, F7=32 | 🟡 TRUNG BÌNH | Paper không nêu rõ thứ tự gán filter cho từng nhánh — agent tự suy luận theo Appendix Table A1 (chuẩn GoogLeNet Inception-v1) và đã hỏi xác nhận người dùng (`implement_plan.md` câu hỏi #4). F3=128 là giá trị reduce bất thường lớn cho nhánh 5×5 (GoogLeNet gốc thường dùng reduce nhỏ) — đã được ghi chú rõ trong `ReproduceSummary.md` Mục 9.2 là điểm rủi ro nếu cần điều chỉnh thêm. |
| 3 | Dense projection (224→256) sau Inception module | `Dense(256, activation='relu')` áp dụng cho cả patch_out và cls_out | 🟡 TRUNG BÌNH | Không có trong paper — là giải pháp kỹ thuật cần thiết để giải quyết mismatch channel (concat Inception = 224 ≠ embed_dim 256) trước khi cộng residual. Đã ghi rõ trong `ReproduceSummary.md` Mục 9.1 là "key non-trivial wiring decision not stated in the paper". Ảnh hưởng có thể bù đắp qua training nhưng vẫn là một thành phần kiến trúc không xuất phát từ paper. |
| 4 | Cách bố trí 2 residual connection (quanh MHA riêng + quanh cả block) thay vì 1 residual lớn duy nhất | 2× `Add()` lồng nhau | 🟡 TRUNG BÌNH | Câu paper "each Trans-Inception block is surrounded by a skip connection" có thể đọc theo 2 cách: (a) 1 residual duy nhất bao trọn cả block, hoặc (b) residual theo chuẩn Transformer (1 quanh attention, 1 quanh feed-forward/Inception). Notebook chọn cách (b) — hợp lý về kỹ thuật (chuẩn ngành) nhưng không có hình Figure 2 để xác nhận 100% câu chữ paper. |
| 5 | Learning rate = 1e-4 | 1e-4 | 🟡 TRUNG BÌNH | Paper không nêu LR cụ thể, chỉ nói "Adam optimizer". Giá trị do người dùng xác nhận (`implement_plan.md`) — ảnh hưởng trực tiếp tới tốc độ/khả năng hội tụ nhưng có thể bù đắp bằng số epoch lớn (500) + early stopping. |
| 6 | Augmentation params (rotation=30, zoom=0.2, brightness=[0.8,1.2]) | Theo CORE RULE #3 | 🟢 THẤP | Paper không nêu tham số augmentation cụ thể (chỉ "rotation, flipping, etc." cho riêng Maize, không áp dụng cho PlantVillage). Giá trị là mặc định hệ thống bắt buộc, có thể điều chỉnh bằng training, không ảnh hưởng đến tính đúng của kiến trúc. |
| 7 | CLS token + learnable positional embedding | `tf.Variable`-style qua `add_weight` trong custom layer | 🟢 THẤP | Paper mô tả ViT gốc tham chiếu đến Dosovitskiy 2021 (có CLS token + positional embedding chuẩn) nhưng không lặp lại chi tiết này cho riêng TrIncNet — đây là suy luận hợp lý theo chuẩn ViT, rủi ro thấp vì đúng thông lệ ngành. |
| 8 | Seed = 42, TF_DETERMINISTIC_OPS | Theo CORE RULE #7/#9 | 🟢 THẤP | Paper không đề cập seed — yêu cầu hệ thống, không ảnh hưởng đến khả năng học của model, chỉ ảnh hưởng đến reproducibility giữa các lần chạy. |

### Rủi ro cao cần chú ý:
- 🔴 **key_dim của MultiHeadAttention = 256** (không phải `embed_dim/num_heads = 21` theo chuẩn ViT) — làm MHA chiếm áp đảo tổng số tham số mô hình (~88%), đi ngược lại tinh thần "TrIncNet nhẹ hơn ViT nhờ Inception nhẹ hơn MLP" của paper, dù tổng param cuối cùng gần đúng với con số 6.95M paper báo cáo. Đây là điểm DUY NHẤT được xếp rủi ro CAO trong toàn bộ audit.

---

## 7. DANH SÁCH VẤN ĐỀ CẦN SỬA

*(Quyết định hiện tại là GO WITH CAUTION — không bắt buộc sửa trước khi submit, nhưng nên xem xét các mục dưới đây để tăng độ trung thực kiến trúc)*

### Nên sửa (ảnh hưởng đến chất lượng):
- [ ] Xem xét lại giá trị `key_dim` của `MultiHeadAttention`. Nếu mục tiêu là tái hiện đúng tinh thần "TrIncNet nhẹ vì Inception nhẹ hơn MLP", nên dùng `key_dim = EMBED_DIM // NUM_HEADS = 21` (chuẩn ViT) và chấp nhận tổng tham số thấp hơn paper (~1.39M thay vì ~7.19M), kèm ghi chú rõ trong báo cáo rằng số tham số sẽ lệch nhiều so với 6.95M do thiếu chi tiết wiring trong paper. Hiện tại lựa chọn `key_dim=256` ưu tiên khớp số lượng tham số tổng nhưng hy sinh đúng tỷ trọng kiến trúc nội bộ — nên ghi rõ trade-off này trong phần mô tả Kaggle notebook khi upload.
- [ ] Bổ sung dòng giải thích ngắn trong markdown cell đầu notebook về quyết định `key_dim=EMBED_DIM` để người đọc notebook (không chỉ người verify) hiểu rõ đây là trade-off có chủ đích, tránh hiểu nhầm là bug.

### Tùy chọn (cải thiện độ trung thực):
- [ ] Xóa import `ReduceLROnPlateau` không sử dụng (dead import) để code sạch hơn — không ảnh hưởng kết quả.
- [ ] Có thể bổ sung `workers=` / `use_multiprocessing=True` trong `model.fit()` để tối ưu tốc độ đọc dữ liệu trên Kaggle GPU (tương đương tinh thần CC23 cho TensorFlow), tuy không bắt buộc vì rule này viết gốc cho PyTorch DataLoader.

---

## 8. GHI CHÚ BỔ SUNG

- Notebook đúng framework với paper (TensorFlow/Keras), không bị áp đặt PyTorch mặc định của hệ thống — điều này đúng theo CORE RULE #2 (Framework Flexibility) vì paper nêu rõ TF 2.6.0/Keras tại Mục 4.3.
- `implement_plan.md` cho thấy quá trình hỏi-xác nhận với người dùng được thực hiện bài bản (7 câu hỏi CẦN XÁC NHẬN), bao gồm đúng câu hỏi về mapping filter Inception và learning rate — đây là thực hành tốt theo CORE RULE #1 (Zero Hallucination), giảm thiểu rủi ro tự suy diễn không kiểm soát.
- `ReproduceSummary.md` ghi nhận trung thực và minh bạch về 2 quyết định kỹ thuật khó nhất (channel mismatch 224 vs 256, và CLS token/positional embedding qua Lambda layer) — đáng ghi nhận tích cực vì giúp người dùng/người verify hiểu rõ phần nào là suy luận kỹ thuật, phần nào là từ paper trực tiếp.
- Điểm Architecture Fidelity (79%) và điểm tổng (84.4%) phản ánh một notebook về cơ bản đúng cấu trúc layer-by-layer, nhưng có một quyết định kỹ thuật (key_dim) ảnh hưởng đến tỷ trọng tham số nội bộ theo cách không phản ánh đúng ý tưởng cốt lõi "lightweight" của paper. Người dùng nên cân nhắc kỹ trước khi dùng kết quả tham số (~7.19M vs paper 6.95M) để khẳng định "đã phục dựng đúng tính lightweight" của TrIncNet trong báo cáo/bài viết tiếp theo — nên trình bày kèm caveat về MHA chiếm tỷ trọng lớn.
- Không phát hiện cell nào trong notebook có `execution_count` hoặc `outputs` đã chạy — đúng yêu cầu CC11 (chỉ sinh notebook, không thực thi).
