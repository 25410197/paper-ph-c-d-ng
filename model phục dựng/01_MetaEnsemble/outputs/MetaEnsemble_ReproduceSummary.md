# Meta-Ensemble (MLP-Mixer + LSTM + SVM) — Reproduce Summary
**Paper:** A Lightweight Meta-Ensemble Approach for Plant Disease Detection Suitable for IoT-Based Environments
**Tác giả:** Ritesh Maurya, Satyajit Mahapatra, Lucky Rajput
**Nguồn:** IEEE Access, 2024 — DOI: 10.1109/ACCESS.2024.3367443
**Ngày phục dựng:** 2026-06-19
**Notebook xuất:** `papers/01_A_Lightweight_Meta-Ensemble_Approach_for_Plant_Disease_Detection_Suitable_for_IoT-Based_Environments/outputs/MetaEnsemble_Experiment.ipynb`

---

## 1. THÔNG TIN BÀI BÁO

| Mục | Chi tiết |
|-----|---------|
| Tiêu đề | A Lightweight Meta-Ensemble Approach for Plant Disease Detection Suitable for IoT-Based Environments |
| Tác giả | Ritesh Maurya, Satyajit Mahapatra, Lucky Rajput |
| Journal | IEEE Access |
| Năm | 2024 |
| DOI | 10.1109/ACCESS.2024.3367443 |
| Nhóm phân loại | Nhóm A (có model phục dựng được) |

---

## 2. KIẾN TRÚC MODEL

**Tên model:** Meta-Ensemble (MLP-Mixer + LSTM + SVM)
**Loại:** 2-level stacking ensemble (KHÔNG end-to-end)
**Framework:** TensorFlow / Keras (confirmed — khớp paper gốc)
**Module đặc biệt:** MLP-Mixer all-MLP block (token-mixing + channel-mixing), LSTM trên feature từ 4 CNN frozen, SVM làm meta-learner ở Level 2.

### Tổng quan luồng

```
LEVEL 1
  Branch A : MLP-Mixer  (64x64x3 -> softmax probs)
  Branch B : 4 frozen CNN -> concat features (5632-d) -> LSTM(30) -> softmax probs
LEVEL 2
  SVM (RBF) : meta-features = concat[P_Mixer, P_LSTM] (2*num_classes) -> final class
```

### 2a. MLP-Mixer (Branch A) — chi tiết layer

| Layer / Block | Cấu hình | Output Shape | Nguồn |
|--------------|----------|--------------|-------|
| Input | 64×64×3 | (B,64,64,3) | Table 5 |
| Resize (compression) | 64 → 72 (`tf.image.resize`) | (B,72,72,3) | Table 7 (`_compression`) |
| Patchify (reshape) | 9 patches × (24×24×3=1728) | (B,9,1728) | Table 7 (`reshape_1`) |
| Patch projection | Dense(1728 → 128) | (B,9,128) | Table 7 (confirmed by user) |
| Dropout | rate=0.25 | (B,9,128) | Mục IV.D.1.a |
| **×7 Mixer blocks** | token-mix + channel-mix + 2×LN + 2×skip | (B,9,128) | Table 5/7 |
| — Token mixing | Dense(128) → transpose → Dense(9) → transpose → Add → LN | (B,9,128) | Fig.3 |
| — Channel mixing | Dense(128, gelu) → Dense(128) → Add → LN | (B,9,128) | Fig.3 |
| Dropout | rate=0.25 | (B,9,128) | Mục IV.D.1.a |
| GlobalAveragePooling1D | — | (B,128) | Fig.3 |
| Output | Dense(num_classes, softmax) | (B,num_classes) | Fig.3 |

**Total params (paper):** ~810K (Maize) / ~815K (TPP) / ~915K (Cotton) — Table 9.

### 2b. LSTM (Branch B)

| Thành phần | Cấu hình | Nguồn |
|-----------|----------|-------|
| Feature extractors (frozen, ImageNet) | MobileNet, DenseNet121, DenseNet169, DenseNet201 | Mục IV.D.1.b |
| Cách trích | `include_top=False, pooling='avg'`, freeze | Mục IV.D.1.b |
| Concat feature dim | 5632 (auto-detect trong notebook) | Mục IV.D.1.b |
| Input shape vào LSTM | (timestep=1, 5632) | Table 6 |
| LSTM units | 30 | Table 6 |
| Output | Dense(num_classes, softmax) | Fig.2 |

**Total params (paper):** ~225K (< 0.3M) — Table 9.

### 2c. SVM (Level 2)

| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Input | concat[P_Mixer, P_LSTM] = (N, 2·num_classes) | Eq.6 |
| Kernel (best) | RBF | Table 11 |
| GridSearchCV C | [0.1, 1, 2, 10, 100] | confirmed |
| GridSearchCV gamma | [1, 0.1, 0.01, 0.001, 0.0001] | Mục V |
| GridSearchCV kernel | [linear, rbf] | Mục V |
| class_weight | 'balanced' nếu dataset imbalance (Rule 11) | workspace |

---

## 3. HYPERPARAMETERS

### ✅ Trực tiếp từ paper

| Thông số | Giá trị | Vị trí trong paper |
|---------|--------|-------------------|
| Framework | TensorFlow/Keras | Table 5/6/7 |
| Optimizer | Adam | Table 5/6 |
| Loss | sparse categorical crossentropy | Table 5 |
| Batch size | 512 | Table 5/6 |
| Input size | 64×64×3 | Mục IV.B |
| Validation split (LSTM) | 0.1 | Table 6 |
| Train/Test split | 80/20 (stratified) | Table 4 |
| Normalization | pixel/255 → [0,1] | Mục IV.B |
| Mixer embedding dim D | 128 | Table 5 |
| Mixer layers | 7 | Table 5/7 |
| Dropout | 0.25 | Mục IV.D.1.a |
| LSTM units | 30 | Table 6 |
| SVM gamma/kernel grid | xem §2c | Mục V |

### ⚠️ Giá trị giả định / chốt bởi người dùng (không cụ thể trong paper)

| Thông số | Giá trị dùng | Lý do |
|---------|------------|-------|
| Learning rate | 1e-3 | Paper chỉ ghi "Adam, LR < 0.001" — user chốt 1e-3 |
| Epochs | 100 (+ EarlyStopping) | Paper ghi range 50–100 — user chốt 100 với EarlyStopping |
| EarlyStopping | patience=10, monitor=val_accuracy, restore_best_weights=True | User chốt |
| Patch embedding mechanism | resize 64→72 → 9 patch (24×24×3) → Dense(1728→128) | Paper `_compression` không mô tả text — user xác nhận cơ chế |
| Augmentation params (S2) | rotation=30°, zoom=0.2, hflip, brightness=[0.7,1.3] | Paper nêu loại aug nhưng không nêu range |
| ColorJitter/Brightness | brightness_range=[0.7,1.3] | Yêu cầu cứng Rule 3 (paper không nêu) |
| SVM C grid | [0.1, 1, 2, 10, 100] | User chốt (paper text khác chút: best C=2) |

---

## 4. DATASET

### 4a. Dataset trong Paper *(chỉ để tham khảo)*

| Mục | Cotton | Maize | TPP |
|-----|--------|-------|-----|
| #Classes | 4 | 4 | 15 (Table 12 ghi 10 — mâu thuẫn nội bộ) |
| Total | 1518 | 2529 | 20637 |
| Train (80%) | 1214 | 2023 | 16573 |
| Test (20%) | 304 | 506 | 4064 |
| Input | 64×64×3 | 64×64×3 | 64×64×3 |
| Normalize | [0,1] | [0,1] | [0,1] |

> Maize-Healthy chỉ 21 ảnh → imbalance ~57× → Rule 11 sẽ auto bật class weighting.

### 4b. Dataset Thực Tế *(dataset của người dùng)*

| Mục | Chi tiết |
|-----|---------|
| Đường dẫn | `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` |
| Số classes | **auto-detect** từ `os.listdir` (Rule 12) |
| Danh sách classes | auto-detect (in ra ở Cell 7) |
| Tổng số ảnh | in ra ở Cell 7 |
| Train / Test | 80 / 20 stratified (validation = 0.1 internal split của `model.fit`) |
| Image size | 64×64×3 |
| Normalization | pixel/255 → [0,1] |

> Notebook chạy trên dataset của người dùng, KHÔNG phải dataset gốc của paper. `num_classes`, `class_names` đều auto-detect.

---

## 5. AUGMENTATION (Scenario 2)

| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | rotation_range=30 | Paper Mục IV.C (góc không nêu → giả định) |
| Zoom / Scale | zoom_range=0.2 | Paper Mục IV.C (range không nêu → giả định) |
| Horizontal Flip | horizontal_flip=True | Paper Mục IV.C |
| Brightness | brightness_range=[0.7, 1.3] | Rule 3 workspace (paper không nêu) |

Augmentation áp dụng OFFLINE: mỗi ảnh train sinh 1 bản augment → concat với bản gốc → shuffle. CNN feature extraction + LSTM train trên tập đã mở rộng. Test set KHÔNG augment.

---

## 6. KẾT QUẢ GỐC TRONG PAPER *(chỉ để tham khảo)*

> ⚠️ Kết quả dưới đây của tác giả trên dataset gốc — không so sánh trực tiếp với dataset người dùng.

| Dataset (paper) | Accuracy | Notes |
|-----------------|---------|-------|
| Cotton | 98.43% | AUC 0.98 |
| Maize | 94.27% | AUC 0.995 |
| TPP | 97.45% | AUC 0.9993 |
| Meta-ensemble params | ~0.92M–1.04M | Table 9 |

### Kết quả thực tế *(điền sau khi chạy notebook)*

| Scenario | Accuracy | F1-Score | Recall | Parameters (Mixer+LSTM) |
|----------|---------|---------|--------|------------------------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [ ] Grad-CAM Visualization (paper không có → không thêm)
- [ ] Ablation Study (không thêm)
- [ ] Transfer Learning Comparison (CNN feature extractor đã là transfer learning nội bộ)
- [ ] Inference Benchmark (không thêm — paper báo cáo memory 18KB nhưng không trong scope)
- [x] **2-level stacking ensemble** (Mixer + LSTM + SVM) — core của paper
- [x] **GridSearchCV cho SVM** (bám sát Table 11)
- [x] **Excel export** (2 sheet Baseline/Augmented với ảnh nhúng)

---

## 8. OUTPUT FILES

```
papers/01_A_Lightweight_Meta-Ensemble_Approach_for_Plant_Disease_Detection_Suitable_for_IoT-Based_Environments/
├── inputs/
│   └── implement_plan.md
└── outputs/
    ├── MetaEnsemble_Experiment.ipynb
    └── MetaEnsemble_ReproduceSummary.md
```

Artifacts sinh ra khi chạy notebook (trong `/kaggle/working/`):
`best_mixer_s1.keras`, `best_lstm_s1.keras`, `best_svm_s1.pkl`,
`best_mixer_s2.keras`, `best_lstm_s2.keras`, `best_svm_s2.pkl`,
`mixer_s1_learning_curve.png`, `lstm_s1_learning_curve.png`, `confusion_matrix_s1.png`,
`mixer_s2_learning_curve.png`, `lstm_s2_learning_curve.png`, `confusion_matrix_s2.png`,
`dataset_distribution.png`, `results_summary.xlsx`.

---

## 9. GHI CHÚ IMPLEMENTATION & KHÁC BIỆT SO VỚI PAPER

### Quyết định triển khai
- **Stacking, không end-to-end:** Mixer và LSTM train độc lập; SVM train trên softmax-probs của chúng (lấy trên TRAIN set), evaluate trên TEST set qua toàn pipeline.
- **CNN feature caching:** 4 CNN frozen chạy inference 1 lần để trích feature (5632-d), cache ra numpy. LSTM train trên feature tĩnh — KHÔNG gọi lại CNN mỗi epoch.
- **FEATURE_DIM auto-detect:** `feat_train.shape[1]` thay vì hard-code 5632, để tránh mismatch dimension (lesson từ reproduction trước).
- **Meta-feature = softmax probabilities** (không phải logits), concat theo thứ tự [Mixer, LSTM].
- **`sparse_categorical_crossentropy`** vì label dạng integer (không one-hot).
- **MirroredStrategy:** mọi model build + compile nằm trong `strategy.scope()`. Load/predict ngoài scope là an toàn.

### Khác biệt / điểm rủi ro so với paper
1. **Patch embedding `_compression` (RỦI RO PARAM-MATCH):** Paper Table 7 cho thấy 64→72 resize rồi reshape (9,128) với `_compression` 8272 params, nhưng KHÔNG mô tả cơ chế bằng text. Notebook dùng: resize 64→72 → reshape 9×1728 → Dense(1728→128). Cách này tạo đúng N=9, D=128 nhưng **số params của bước patch projection sẽ KHÁC** giá trị 8272 của paper (Dense 1728→128 = 221,312 params), nên **tổng params Mixer sẽ cao hơn ~810K–915K của paper**. Đây là điểm không thể tái hiện 1:1 do paper thiếu mô tả. Dùng `model.summary()` để xác minh con số thực tế.
2. **DenseNet feature dims:** Notebook dùng dims thực tế của torchvision/Keras: MobileNet=1024, DenseNet121=1024, DenseNet169=1664, DenseNet201=1920 → tổng=5632. Paper text liệt kê dims khác thứ tự (1024/1024/1536/2048) nhưng cùng tổng 5632. Notebook auto-detect nên luôn nhất quán.
3. **Learning rate / epochs:** paper để range, user chốt LR=1e-3, EPOCHS=100 + EarlyStopping.
4. **Augmentation ranges:** paper chỉ nêu loại phép (flip/rotation/zoom/width-shift), không nêu tham số → giả định hợp lý (xem §5). Brightness thêm theo Rule 3.
5. **SVM C grid:** user chốt [0.1,1,2,10,100] (gồm C=2 best của paper Table 11). Paper text grid khác chút ([0.1,1,10,100,1000]).
6. **Class weighting:** paper không đề cập, nhưng Rule 11 workspace auto bật khi imbalance ≥1.5× (gần như chắc bật với dataset plant disease). SVM dùng `class_weight='balanced'` tương ứng.
7. **ROC/AUC:** paper vẽ ROC, nhưng user xác nhận KHÔNG cần (Rule 4 chỉ cần Acc/F1/Recall/classification_report). Dùng `SVC(probability=False)` cho nhanh.
8. **Learning Curve trong Excel:** mỗi scenario có 2 learning curve (Mixer + LSTM). Excel chỉ nhúng được 1 ô/scenario → dùng Mixer LC làm đại diện. Cả 2 LC vẫn được lưu PNG đầy đủ và hiển thị inline trong notebook.
