# MRW-CNN — Reproduce Summary
**Paper:** Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning
**Tác giả:** Diana Susan Joseph, Pranav M. Pawar, Kaustubh Chakradeo
**Nguồn:** IEEE Access, Vol. 12, 2024 — DOI: 10.1109/ACCESS.2024.3358333
**Ngày phục dựng:** 2026-06-16
**Notebook xuất:** `papers/04- Real-Time Plant Disease Dataset Development/outputs/MRW-CNN_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning |
| Tác giả | Diana Susan Joseph, Pranav M. Pawar, Kaustubh Chakradeo |
| Journal / Conference | IEEE Access, Vol. 12 |
| Năm | 2024 |
| DOI / Link | 10.1109/ACCESS.2024.3358333 |
| Nhóm phân loại | Nhóm A (model đề xuất có tên cụ thể MRW-CNN; kiến trúc nay đã rõ ràng nhờ đọc trực tiếp Figure 9 bằng vision — chỉ còn stride/padding cần giả định) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** `MRW-CNN` (Maize, Rice, Wheat CNN)
**Loại:** CNN train from scratch (không pretrained backbone)
**Framework:** TensorFlow / Keras (lựa chọn người dùng, giữ theo paper gốc — ghi đè rule mặc định PyTorch của dự án)
**Module đặc biệt:** Không có module mới đặc biệt; là CNN tuần tự gọn nhẹ (5 conv block + GAP + Dense softmax).

### Bảng kiến trúc

Spatial output shape dưới đây tính theo giả định stride=2, padding='same' (downsampling thay cho pooling vì network KHÔNG có pooling layer).

| Layer / Block | Kernel / Filters | Stride | Output Shape (224 input) | Ghi chú |
|--------------|-----------------|--------|--------------------------|---------|
| Input | — | — | 224×224×3 | — |
| Conv Block 1 | 3×3 / 32 | 2 | 112×112×32 | Conv→BN→ReLU (✅ filters/kernel CONFIRMED Fig. 9; ⚠️ stride ASSUMED) |
| Conv Block 2 | 3×3 / 64 | 2 | 56×56×64 | Conv→BN→ReLU (✅ CONFIRMED Fig. 9; ⚠️ stride ASSUMED) |
| Conv Block 3 | 3×3 / 64 | 2 | 28×28×64 | Conv→BN→ReLU (✅ CONFIRMED Fig. 9 — filters=64 giống block 2; ⚠️ stride ASSUMED) |
| Conv Block 4 | 3×3 / 256 | 2 | 14×14×256 | Conv→BN→ReLU (✅ CONFIRMED Fig. 9; ⚠️ stride ASSUMED) |
| Conv Block 5 | 3×3 / 256 | 2 | 7×7×256 | Conv→BN→ReLU (✅ CONFIRMED Fig. 9 — filters=256 giống block 4; ⚠️ stride ASSUMED) |
| MaxPooling | — | — | — | ✅ CONFIRMED KHÔNG CÓ pooling nào (Fig. 9 + Section V text) |
| Global Average Pooling | — | — | 256 | ✅ Confirmed (Fig. 9 + paper) |
| Dense (output) | 5 (num_classes) | — | 5 | Softmax — ✅ Confirmed (Fig. 9 + paper) |

**Total params (paper):** KHÔNG nêu con số cụ thể (paper chỉ mô tả định tính "less number of layers"). Số param thực tế in ra bằng `model.summary()` khi chạy notebook.

> ✅ **Confirmed (paper Figure 9, đọc trực tiếp bằng vision — xem `inputs/paper_content.md` mục "EXTRACTED FIGURES"):** 5 conv block; thứ tự mỗi block = Conv2D → BatchNorm → ReLU; số filters chính xác **32 → 64 → 64 → 256 → 256**; kernel **3×3** mọi block; **KHÔNG có MaxPooling/pooling** nào giữa hay sau các conv block (chỉ 1 GAP ở cuối); head = Global Average Pooling → Dense → Softmax; train from scratch; input 224×224.
> ⚠️ **Assumed (điểm giả định DUY NHẤT còn lại — paper không nêu ở text lẫn figure; người dùng duyệt 2026-06-16):** stride = 2 và padding = 'same' cho cả 5 conv layer. Vì không có pooling, stride=2 đóng vai trò downsampling. Đây là yếu tố ảnh hưởng trực tiếp output shape mỗi block và tổng số param.

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper
| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework | TensorFlow / Keras (CUDNN) | Section VI.A "Experiment Setup" |
| Optimizer | SGD | Section VI.B + Section V |
| Learning rate | 0.001 | Section VI.B |
| Batch size | 32 | Section VI.B |
| Epochs | 100 | Section VIII.A "trained for 100 epochs" |
| Input size | 224 × 224 | Section III.B |
| Train/Val/Test split | 80% / 10% / 10% (trên dataset gốc paper) | Section IV |
| Activation | ReLU (conv), Softmax (output) | Section V + Fig. 9 |
| Conv filters | 32 → 64 → 64 → 256 → 256 | **Figure 9 (đọc bằng vision)** |
| Kernel size | 3×3 (mọi block) | **Figure 9** |
| Pooling | KHÔNG có (chỉ GAP ở cuối) | **Figure 9 + Section V** |
| Pretrained | Không (train from scratch) | Section V, Abstract |

### ⚠️ Giá trị giả định (không có trong paper)
| Thông số | Giá trị dùng | Lý do giả định |
|---------|------------|--------------|
| Stride / Padding | **2 / 'same'** | Giả định DUY NHẤT còn lại. Paper không nêu stride/padding ở text lẫn Fig. 9. Vì network không có pooling, stride=2 đóng vai trò downsampling. Người dùng duyệt 2026-06-16. |
| Loss function | CategoricalCrossentropy | Paper không nêu rõ; chuẩn cho multi-class + softmax. Người dùng xác nhận. |
| Normalization | Scale /255 (KHÔNG ImageNet mean/std) | Paper không nêu. Phù hợp train-from-scratch. Người dùng xác nhận. |
| LR Scheduler / Warmup | Không dùng | Paper không nêu. |
| Best-model checkpoint theo val_accuracy | Có (rule dự án #10) | Paper dùng early stopping cho 8 model fine-tuned; với MRW-CNN không nêu rõ → dùng checkpoint thay thế. |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ để tham khảo)*

| Mục | Chi tiết từ Paper |
|-----|---------|
| Tên dataset gốc | Dataset tự phát triển của tác giả: Maize / Rice / Wheat (3 dataset riêng, mỗi cái 5 class) — KHÔNG công khai |
| Số classes (paper) | 5 mỗi dataset (4 bệnh + 1 healthy) |
| Tổng số ảnh (paper) | ~5000 ảnh/class sau augmentation (4000 train / 500 val / 500 test mỗi class) |
| Train/Val/Test (paper) | 80% / 10% / 10% |
| Input size dùng trong paper | 224 × 224 |
| Normalization (paper) | Không nêu rõ |

### 4b. Dataset Thực Tế *(dataset của người dùng — substitute)*

| Mục | Chi tiết |
|-----|---------|
| Tên dataset | Chili pepper (ớt) — `dataset_smartcrop_plit_300` |
| Đường dẫn | `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` |
| Số classes | 5 (auto-detect) |
| Danh sách classes | anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf |
| Cấu trúc | Pre-split sẵn: `train/`, `val/`, `test/` (mỗi folder có 5 class subfolder) |
| Train / Val / Test | Trỏ trực tiếp vào folder pre-split (không tự chia trong code) |
| Image size | 224 × 224 |
| Normalization | Scale /255 |

> **Lý do thay thế dataset:** Dataset maize/rice/wheat gốc của paper không công khai. Notebook giữ nguyên kiến trúc và hyperparameter theo paper, chỉ thay nguồn ảnh sang dataset ớt của người dùng (cùng số class = 5).

---

## 5. AUGMENTATION (Scenario 2)

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | rotation_range = 90 | Paper Section III.C (90 độ) |
| Zoom / Scale | zoom_range = 0.2 | Paper Section III.C |
| Horizontal Flip | bật | Paper Section III.C |
| Brightness | brightness_range = [0.2, 0.8] | Paper Section III.C |

> Paper dùng OFFLINE augmentation (sinh ảnh mới lưu ra folder). Notebook dùng ONLINE augmentation trong pipeline `ImageDataGenerator` theo rule dự án — khác biệt phương pháp, không ảnh hưởng giá trị tham số. Paper còn dùng thêm vertical flip / height_shift / width_shift / shear (0.2) nhưng rule dự án chỉ yêu cầu 4 kỹ thuật chuẩn nên các kỹ thuật còn lại không đưa vào Scenario 2.

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ để tham khảo)*

> ⚠️ Kết quả dưới đây là của tác giả trên dataset gốc (maize/rice/wheat) — KHÔNG thể so sánh trực tiếp với run trên dataset ớt. Lưu ý kiến trúc nay đã khớp paper (Fig. 9); chỉ stride/padding là giả định.

| Dataset (của paper) | MRW-CNN Accuracy | Best fine-tuned | Acc fine-tuned |
|---------------------|------------------|-----------------|----------------|
| Maize | 97.04% | Xception | 95.80% |
| Rice | 97.06% | Xception | 97.28% |
| Wheat | 98.08% | MobileNetV2 | 96.32% |

Parameters MRW-CNN: paper KHÔNG nêu con số cụ thể.

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Score | Recall | Parameters |
|----------|---------|---------|--------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization (không có trong paper)
- [ ] Ablation Study (không có trong paper cho MRW-CNN)
- [ ] Transfer Learning Comparison (8 model fine-tuned là baseline so sánh — optional, không build vì deliverable bắt buộc là MRW-CNN)
- [ ] Inference Benchmark (không có trong paper)

---

## 8. OUTPUT FILES

```
papers/04- Real-Time Plant Disease Dataset Development/
├── inputs/
│   ├── 04- Real-Time Plant Disease Dataset Development.pdf
│   ├── paper_content.md        ← extracted by pdf-extractor
│   └── implement_plan.md       ← analyzed by paper-brainstorm (6 quyết định đã xác nhận)
└── outputs/
    ├── MRW-CNN_Experiment.ipynb
    └── MRW-CNN_ReproduceSummary.md
```

Artifacts sinh khi chạy notebook trên Kaggle (`/kaggle/working/`):
`best_model_s1.keras`, `best_model_s2.keras`, `learning_curve_s1.png`, `confusion_matrix_s1.png`, `learning_curve_s2.png`, `confusion_matrix_s2.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION

- **Deliverable bắt buộc** là MRW-CNN (model đề xuất, train from scratch). 8 model fine-tuned (VGG16, ResNet50/101, InceptionV3, InceptionResNetV2, MobileNet/V2, Xception) là baseline so sánh — không build trong notebook này theo yêu cầu.
- **SỬA LỖI quan trọng (so với bản notebook trước):** Bản đầu giả định filters tăng dần 32→64→128→256→512 kèm MaxPool2D(2×2) sau mỗi block — ĐÃ SAI. Sau khi đọc trực tiếp Figure 9 bằng vision, filters thực tế là **32→64→64→256→256** và **KHÔNG có MaxPooling nào**. Đã viết lại builder (Cell 5) bỏ toàn bộ MaxPooling và sửa danh sách filters.
- **Provenance hiện tại:** filters / kernel 3×3 / thứ tự Conv→BN→ReLU / không-pooling / GAP / Dense-softmax đều **CONFIRMED từ Figure 9** (không còn là giả định). Điểm giả định DUY NHẤT còn lại là **stride=2 + padding='same'** (downsampling thay pooling) — đã đánh dấu rõ "ASSUMED" trong notebook (Cell 1 markdown, Cell 3 config, Cell 5 builder) và summary này.
- **Rủi ro còn lại:** param count và output shape mỗi block phụ thuộc vào stride giả định. Nếu paper thực sự dùng stride=1 thì spatial map giữ 224×224 đến cuối (GAP vẫn cho ra vector 256-dim, param Conv không đổi nhưng chi phí compute lớn hơn nhiều). Param của Conv layer KHÔNG phụ thuộc stride; chỉ Dense head cố định (256→5) nên tổng param ổn định bất kể stride.
- **Optimizer = SGD** (paper-confirmed) — khác với reference TrIncNet (Adam). Đã dùng `SGD(learning_rate=0.001)` theo đúng paper.
- **Loss = CategoricalCrossentropy** (`class_mode='categorical'` cho generator) — phù hợp output softmax.
- **Normalization /255** thay vì ImageNet mean/std — đúng cho train-from-scratch (người dùng xác nhận).
- **Dataset pre-split:** dùng trực tiếp `flow_from_directory` trên TRAIN_DIR/VAL_DIR/TEST_DIR; KHÔNG có logic random_split.
- **Early stopping:** Paper áp dụng early stopping cho 8 model fine-tuned; với MRW-CNN không nêu rõ. Notebook dùng best-model-checkpoint theo `val_accuracy` (rule dự án #10), reload checkpoint tốt nhất trước khi eval test. KHÔNG dùng EarlyStopping callback để chạy đủ 100 epoch như paper.
- **Class imbalance (rule #11):** auto-detect; paper nói dataset cân bằng nên nhiều khả năng ratio < 1.5 → không dùng class_weights. Code vẫn tự xử lý nếu dataset ớt mất cân bằng.
- **Augmentation:** Scenario 2 dùng 4 kỹ thuật chuẩn tham số hóa theo paper (rotation 90, zoom 0.2, h-flip, brightness 0.2–0.8). Online thay vì offline như paper.
- **Reproducibility (TF):** seed 42 cho random/numpy/tf, `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS=1`. Đây là TF-equivalent cho rule #7/#9 (PyTorch cudnn deterministic không áp dụng cho TF).
