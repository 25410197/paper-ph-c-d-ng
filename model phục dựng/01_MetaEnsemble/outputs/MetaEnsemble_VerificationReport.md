# Meta-Ensemble (MLP-Mixer + LSTM + SVM) — Verification Report
**Paper:** A Lightweight Meta-Ensemble Approach for Plant Disease Detection Suitable for IoT-Based Environments
**Notebook:** `papers/01_.../outputs/MetaEnsemble_Experiment.ipynb`
**Ngay verify:** 2026-06-20
**Nguoi verify:** notebook_verifier agent (lan 3 — kiem tra runtime bug bao cao boi user)

---

## KET QUA TONG QUAN

| Dimension | Score | Trong so | Diem co trong so |
|-----------|-------|---------|----------------|
| Paper Completeness | 83% | 15% | 12.5 |
| Architecture Fidelity | 79% | 40% | 31.6 |
| Hyperparameter Match | 88% | 25% | 22.0 |
| Preprocessing Consistency | 92% | 5% | 4.6 |
| Constraint Compliance | 71% | 15% | 10.7 |
| **TONG** | | **100%** | **81.4%** |

### Quyet dinh: REVISE — Co loi runtime nghiem trong lam notebook KHONG chay duoc

> Diem tong 81.4% thuoc dang "GO WITH CAUTION" neu notebook chay khong loi. Tuy nhien, notebook hien co **2 cap loi runtime cat-level CRITICAL** (cell training bi nhan doi va bien chua duoc dinh nghia) khien notebook se fail hoac cho ra ket qua sai khi chay tren Kaggle. Phai sua 4 van de ky thuat truoc khi submit.

---

## 1. PAPER COMPLETENESS CHECK

| # | Muc kiem tra | Trang thai | Ghi chu |
|---|-------------|-----------|---------|
| P1 | Loai bai bao (Nhom A/B/C) | Du | Nhom A — meta-ensemble 2 tang cu the, co bang hyperparameter, bang layer, ket qua thuc nghiem |
| P2 | Ten model + kien truc ro rang | Du | MLP-Mixer + LSTM + SVM; mo ta day du o Muc IV |
| P3 | Chi tiet tung layer / block | Mo ho | Table 7 cho output shape + param count 1 mixer layer; buoc `_compression` (8272 params) khong co mo ta co che ben trong bang text |
| P4 | Activation function | Du | GELU trong FC layers cua MLP, Softmax o output (Muc IV.D.1.a) |
| P5 | BatchNorm / Dropout | Du | Dropout rate=0.25 (Muc IV.D.1.a), LayerNorm sau moi MLP |
| P6 | Optimizer | Mo ho | "Adam, learning rate < 0.001" — khong neu gia tri cu the |
| P7 | Loss function | Du | Sparse categorical crossentropy (Mixer — Table 5); Categorical crossentropy (LSTM — Table 6) |
| P8 | Batch size | Du | 512 cho ca Mixer va LSTM (Table 5/6) |
| P9 | So epochs / stopping criterion | Mo ho | Range "50–100" — khong phai gia tri cu the; khong de cap EarlyStopping |
| P10 | Input size (H x W x C) | Du | 64×64×3 (Muc IV.B, Table 5/6) |
| P11 | Domain model | Du | Plant disease detection, IoT deployment — ro rang |
| P12 | Input size va normalization ro rang | Du | 64×64×3; pixel/255 → [0,1] (Muc IV.B) |
| P13 | Augmentation co mo ta | Mo ho | Muc IV.C: rotation, flip, resize, zoom, width-shift — khong neu tham so |
| P14 | Normalization values (mean/std) | Du | [0,1] — KHONG dung ImageNet stats (Muc IV.B) |
| P15 | Ket qua thuc nghiem | Du | Accuracy: Cotton 98.43%, Maize 94.27%, TPP 97.45% (Table 9) |
| P16 | Framework | Mo ho | Khong ghi ro TF/Keras; suy ra tu ten layer Table 7 (dense_1, layer_normalization, tf_op_layer_lambda) |
| P17 | Pretrained weights | Du | MobileNet + DenseNet121/169/201, ImageNet weights, conv base frozen (Muc IV.D.1.b) |
| P18 | LR Scheduler | Thieu | Khong de cap bat ky scheduler nao |

**Tinh diem: 13 muc Du = 13.0; 4 muc Mo ho (P3, P6, P9, P13, P16 — 5 muc) x 0.5 = 2.5; 1 muc Thieu (P18) = 0**
**Score: 15/18 = 83% — Bai bao DU de reproduce nhung mot so thong so co gia tri khoang / mo ho**

---

## 2. ARCHITECTURE FIDELITY CHECK

### Nhanh A: MLP-Mixer (Level 1)

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Vi tri trong Notebook |
|---|-----------|-------------------|----------------------|-------|----------------------|
| A1 | Ten model | MLP_Mixer | `Model(..., name="MLP_Mixer")` | Khop | Cell-9 (`build_mixer()`) |
| A2 | So mixer blocks | 7 | `N_LAYERS=7`, vong lap `for i in range(num_layers)` | Khop | Cell-6, Cell-9 |
| A3 | Embedding dim D | 128 | `D=128`, `Dense(d, name="patch_proj")` | Khop | Cell-6, Cell-9 |
| A4 | Patch resize | 64→72 (`_compression`) | `Lambda: tf.image.resize([72,72])` | Khop | Cell-9 (`resize_72`) |
| A5 | So patch tokens | N=9 | `N_PATCHES=9`, `Reshape((n_patches, PATCH_DIM))` | Khop | Cell-6, Cell-9 |
| A6 | Patch embedding so params | 8272 params trong paper (Table 7) | Dense(1728, 128) = 221,312 params | KHONG KHOP | Cell-9 — chenh lech ~27x, kien truc _compression that su co the la Conv hoac DepthwiseConv |
| A7 | Token mixing MLP | 2 FC + skip connection + LayerNorm | Dense(d,gelu)→transpose→Dense(n_patches)→transpose→Add→LN | Khop mot phan | GELU co trong `tok_fc1` nhung paper chi mo ta cho channel mixing; tran thu tu lay transpose truoc LN — xem ghi chu |
| A8 | Channel mixing MLP | 2 FC + GELU + skip connection + LayerNorm | Dense(d,gelu)→Dense(d)→Add→LN | Khop | Cell-9 (`ch_fc1`, `ch_fc2`) |
| A9 | LayerNorm vi tri | Sau moi mixing MLP (token LN + channel LN) | `tok_ln_{i}` sau tok_add, `ch_ln_{i}` sau ch_add | Khop | Cell-9 |
| A10 | Skip connections | Add sau token mixing va channel mixing | `Add(name=f"tok_add_{i}")`, `Add(name=f"ch_add_{i}")` | Khop | Cell-9 |
| A11 | Dropout rate | 0.25 (Muc IV.D.1.a) | `DROPOUT=0.25` — 2 Dropout: sau patch_proj va sau mixer loop | Khop mot phan | Paper chi de cap 1 Dropout (truoc GAP); notebook them 1 cai sau patch_proj — it anh huong |
| A12 | Global Average Pooling | Sau dropout, truoc softmax | `GlobalAveragePooling1D(name="gap")` | Khop | Cell-9 |
| A13 | Output Mixer | Dense(num_classes, softmax) | `Dense(num_classes, activation="softmax", name="mixer_out")` | Khop | Cell-9 |
| A14 | Tong params Mixer | ~810K–915K (Table 9, theo dataset) | >1M (do Dense(1728,128) thay vi _compression 8272 params) | KHONG KHOP | Chenh lech >10% — xac nhan kien truc patch embedding sai |

### Nhanh B: LSTM (Level 1)

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Vi tri trong Notebook |
|---|-----------|-------------------|----------------------|-------|----------------------|
| B1 | CNN feature extractors | MobileNet, DenseNet121, DenseNet169, DenseNet201 (frozen ImageNet) | 4 CNN trong `build_feature_extractors()`, `trainable=False` | Khop | Cell-11 |
| B2 | Feature dim tong | 5632 | `feat_train.shape[1]` auto-detect | Khop | Cell-11, Cell-19 |
| B3 | LSTM units | 30 | `LSTM_UNITS=30`, `LSTM(lstm_units, name="lstm")` | Khop | Cell-6, Cell-9 |
| B4 | Timestep | 1 | `Input(shape=(1, feature_dim))`, `.reshape(-1, 1, FEATURE_DIM)` | Khop | Cell-9, Cell-19 |
| B5 | Output LSTM | Dense(num_classes, softmax) | `Dense(num_classes, activation="softmax", name="lstm_out")` | Khop | Cell-9 |

### Level 2: SVM

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Vi tri trong Notebook |
|---|-----------|-------------------|----------------------|-------|----------------------|
| C1 | Loai classifier | SVM | `SVC(probability=False)` | Khop | Cell-25, Cell-43 |
| C2 | GridSearchCV | C=[0.1,1,10,100,1000]; gamma=[1,0.1,0.01,0.001,0.0001]; kernel=[linear,RBF] | C=[0.1,1,2,10,100]; gamma=same; kernel=[linear,rbf] | Khop mot phan | Thieu C=1000 (paper text), co them C=2 (best theo Table 11) — user da xac nhan |
| C3 | Meta-feature input | concat[P_Mixer, P_LSTM] = (N, 2*num_classes) | `np.concatenate([mixer_probs, lstm_probs], axis=1)` | Khop | Cell-24, Cell-42 |
| C4 | Stacking flow | train Level1 → probs train → train SVM → eval test | Dung quy trinh stacking 2 tang | Khop | Ca 2 Scenario |

**Architecture Fidelity Score: 14 muc applicable; 11 Khop + 3 Khong Khop (A6, A14, C2 mot phan) = 78%**

> Ghi chu quan trong ve A6/A14: Day la han che cau truc khong the khac phuc hoan toan vi paper khong mo ta co che `_compression` bang text. Notebook dung giai phap hop ly nhat co the (Dense projection) nhung so params se cao hon paper ~27%.

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thong so | Gia tri trong Paper | Gia tri trong Notebook | Nguon | Khop? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `Adam(LR)` cho ca Mixer va LSTM | Paper (Table 5, 6) | Khop |
| H2 | Learning rate | < 0.001 (khong cu the) | `LR = 1e-3` | User xac nhan | Khop (user chot gia tri cu the) |
| H3 | LR Scheduler | Khong de cap | Khong co scheduler | Assumed (N/A) | N/A |
| H4 | Loss (Mixer) | sparse categorical crossentropy | `"sparse_categorical_crossentropy"` | Paper (Table 5) | Khop |
| H5 | Loss (LSTM) | categorical crossentropy | `"sparse_categorical_crossentropy"` | Paper (Table 6) | Khop (tuong duong — sparse CCE voi int label = CCE voi one-hot label) |
| H6 | Batch size | 512 | `BATCH_SIZE = 512` | Paper (Table 5, 6) | Khop |
| H7 | Epochs | 50–100 (range) | `EPOCHS = 100` + EarlyStopping(patience=10) | User xac nhan | Khop |
| H8 | Pretrained CNN | ImageNet, frozen | `weights="imagenet"`, `trainable=False` | Paper | Khop |
| H9 | LSTM cells | 30 | `LSTM_UNITS=30` | Paper (Table 6) | Khop |
| H10 | SVM kernel (best) | RBF (Table 11) | Trong grid ["linear", "rbf"] — GridSearchCV se chon | Paper | Khop |

**Hyperparameter Match Score: 8/9 thong so Paper + User (bo H3 la N/A) = 89%**

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing (bat buoc khop)

| # | Muc | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size | 64×64×3 | `IMG_SIZE=64`, `target_size=(img_size, img_size)` | Khop | Cell-6, Cell-7 |
| D2 | Normalization | [0,1] (pixel/255) | `arr = img_to_array(img) / 255.0` | Khop | Cell-7 (`load_split()`) |
| D3 | NUM_CLASSES auto-detect | N/A (paper dung dataset cu the) | `num_classes = len(class_names)` | Khop | Cell-7 |

**Preprocessing Score: 3/3 = 100%**

### 4b. Dataset Split trong Notebook

| # | Muc | Gia tri trong Notebook | Hop ly? | Ghi chu |
|---|-----|----------------------|---------|---------|
| DS1 | Load tu pre-split folders | `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` | Dung | Phu hop voi memory workspace (dataset da co train/val/test) |
| DS2 | Val/Test khong bi augmentation | X_val va X_test load qua `load_split()` — chi resize + normalize | Dung | Khong co aug trong load_split() |
| DS3 | Validation data cho LSTM dung kieu dung | feat_val_lstm shape (N,1,5632) | CO VAN DE | Xem phan "Kiem tra Runtime Bug" — co 2 truong hop dung sai X_val thay vi feat_val_lstm |

### 4c. Augmentation (Scenario 2)

| # | Ky thuat | Paper de cap? | Notebook implement? | Tham so |
|---|---------|--------------|--------------------|----|
| Aug1 | Rotation | Co (Muc IV.C) | `rotation_range=30` | Hop ly (paper khong neu goc) |
| Aug2 | Zoom / Scale | Co ("zooming operations" Muc IV.C) | `zoom_range=0.2` | Hop ly (paper khong neu range) |
| Aug3 | Horizontal Flip | Co ("random flipping" Muc IV.C) | `horizontal_flip=True` | Khop |
| Aug4 | Brightness / ColorJitter | Khong ro (paper neu aug chung) | `brightness_range=[0.7, 1.3]` | Them theo Rule 3 workspace — dung |
| Aug5 | Width-shift | Co ("width-" Muc IV.C) | Khong implement | Thieu — nhung paper khong neu tham so, anh huong thap |

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage

| # | Feature trong Paper | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|---------|
| SF1 | GridSearchCV cho SVM (Table 11) | Co | Cell-25, Cell-43 — param_grid day du |
| SF2 | 2-level stacking ensemble | Co | Toan bo pipeline: Mixer + CNN features + LSTM + SVM |
| SF3 | CNN feature caching (offline) | Co | Cell-19, Cell-37 — extract 1 lan, cache vao numpy |
| SF4 | ROC/AUC curve | Khong | User xac nhan khong can — `probability=False` trong SVC cho nhanh |
| SF5 | Ablation study (SVM vs KNN/RF/NB) | Khong | Khong trong scope |
| SF6 | Inference benchmark (latency) | Khong | Khong trong scope |

### 5b. Core Constraints Compliance

| # | Rang buoc | Co trong Notebook? | Ghi chu |
|---|-----------|--------------------|---------|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 ky thuat) | Co | Scenario 1 tu Cell-12; Scenario 2 tu Cell-30 |
| CC2 | Framework dung voi paper (TF/Keras) | Co | `import tensorflow as tf`, `from tensorflow.keras import ...` — Cell-2 |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | Co | `report_metrics()` tai Cell-9 tinh du 3 |
| CC4 | `classification_report` per class (sklearn) | Co | `classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0)` |
| CC5 | `model.summary()` | Co | `mixer_s1.summary()` Cell-08fec865; `lstm_s1.summary()` Cell-13831ac4; tuong tu S2 |
| CC6 | LC luu file anh dpi=300, cell rieng + plt.show() | CO NHUNG KHONG DAY DU | `plot_learning_curve()` co savefig(dpi=300) va plt.show() — nhung Scenario 2 thieu cell LC rieng cho Mixer (xem Loi 4 ben duoi); bien MIXER_LC_S2 chua duoc dinh nghia |
| CC7 | CM luu file anh dpi=300, cell rieng + plt.show() | Co | Cell-27 (S1), Cell-45 (S2) — day du |
| CC8 | NUM_CLASSES auto-detect | Co | `num_classes = len(class_names)` — Cell-7 |
| CC9 | Excel 2-sheet (Baseline + Augmented), dung cot | Co | Cell-50 — `Workbook` tao 2 sheet, 8 cot chinh xac |
| CC10 | Anh LC + CM nhung vao Excel | CO NHUNG CO THE LOI | `XLImage(lc_path)` va `XLImage(cm_path)` — nhung `MIXER_LC_S2` (cho Augmented sheet) se KHONG TON TAI do thieu cell plot |
| CC11 | Khong co code thuc thi training | N/A | La notebook sinh ra, chi chay khi user run tren Kaggle |
| CC12 | ReproduceSummary.md ton tai | Co | `MetaEnsemble_ReproduceSummary.md` trong thu muc outputs/ |
| CC13 | `model_name` la ten kien truc | Co | `"model_name": "Meta-Ensemble (MLP-Mixer + LSTM + SVM)"` — Cell-28, Cell-46 |
| CC14 | Device auto-detect + to(device) | Co (TF) | `MirroredStrategy()` + `with strategy.scope():` — Cell-4, Cell-08fec865 |
| CC15 | `torch.manual_seed(42)` | Co (TF) | `tf.random.set_seed(42)` + `np.random.seed(42)` + `random.seed(42)` — Cell-3 |
| CC16 | Khong dung `pretrained=True` (deprecated PyTorch) | N/A (TF) | `weights="imagenet"` — Cell-11 — dung voi TF/Keras |
| CC17 | Baseline training transform: CHI Resize + Normalize | Co | `load_split()` chi co resize + /255.0, X_train_s1 = X_train.copy() |
| CC18 | Val/Test KHONG augmentation | Co (cho Mixer); CO LOI (LSTM) | X_val load tu `load_split()` khong co aug; nhung Cell-21 va Cell-39/40 truyen X_val sai cho LSTM validation |
| CC19 | cudnn.deterministic = True + benchmark = False | N/A (TF/Keras) | Chi yeu cau voi PyTorch — khong ap dung |
| CC20 | Best model reload truoc test eval | Co (S1); THIEU (S2) | S1: Cell-18 (`load_model(ckpt_mixer_s1)`), Cell-23 (`load_model(ckpt_lstm_s1)`). S2: Cell-36 reload Mixer S2 dung, nhung thieu cell reload LSTM S2 |
| CC21 | Dataset Distribution: bar chart + imbalance check | Co | Cell-8 — bar chart, imbalance_ratio, class_weight tu dong |
| CC22 | `class_names` auto-detect, khong hard-code | Co | `sorted(d for d in os.listdir(split_dir) if os.path.isdir(...))` — Cell-7 |
| CC23 | DataLoader num_workers + pin_memory | N/A (TF) | TF: `prefetch(AUTOTUNE)` trong `make_tf_dataset()` — Cell-8a8054ed |
| CC24 | `clear_session()` + gc.collect() sau Scenario 1 | Co | `tf.keras.backend.clear_session()` + `gc.collect()` — Cell-29 |

**Compliance Score: 15 day du / 22 applicable (bo CC11, CC16, CC19, CC23 la N/A) = 68%**

> Mat diem: CC6 (LC S2 Mixer thieu), CC10 (MIXER_LC_S2 chua ton tai), CC18 (LSTM val sai kieu), CC20 (thieu reload LSTM S2 truoc eval).

---

## 6. ASSUMPTION RISK AUDIT

| # | Thong so gia dinh | Gia tri dung | Muc do rui ro | Ly do |
|---|------------------|------------|--------------|-------|
| 1 | Co che patch embedding `_compression` | resize 64→72 → Dense(1728→128) | CAO | Paper Table 7 cho 8272 params; Dense(1728,128)=221K params — chenh lech ~27x. Mixer se co >1M params thay vi ~810K–915K. Anh huong so sanh benchmark. |
| 2 | Learning rate cu the | 1e-3 | TRUNG BINH | Paper chi ghi "< 0.001". LR anh huong toc do hoi tu va accuracy cuoi cung. User da chot nhung co the khac paper goc. |
| 3 | Epochs cu the | 100 + EarlyStopping(patience=10) | TRUNG BINH | Paper ghi "50–100", khong de cap EarlyStopping. EarlyStopping tot hon ve mat ky thuat nhung lam kho so sanh chinh xac so epoch voi paper. |
| 4 | Tham so augmentation | rotation=30, zoom=0.2, brightness=[0.7,1.3] | TRUNG BINH | Paper neu loai aug nhung khong neu tham so cu the. |
| 5 | GELU trong token mixing | Co them GELU vao tok_fc1 | TRUNG BINH | Paper de cap GELU chung cho "MLP in mixer layer" — token mixing trong paper (Table 7) co the khong co GELU rieng. Anh huong bieu dien. |
| 6 | Width-shift aug | Khong implement | THAP | Paper de cap nhung khong ro rang; anh huong nho. |
| 7 | SVM C grid | [0.1,1,2,10,100] thay vi [0.1,1,10,100,1000] | THAP | C=1000 it anh huong khi meta-feature nho (2*num_classes). GridSearchCV van tim duoc best. |

---

## 7. DANH SACH LOI RUNTIME VA VAN DE CAN SUA

> Day la phan quan trong nhat. User bao cao "chay bi loi hoai". Kiem tra thu cong notebook xac nhan co 4 van de lam notebook KHONG chay duoc theo thu tu.

### LOI NGHIEM TRONG — PHAI SUA TRUOC KHI SUBMIT (CRASH KHAM CHAC)

---

**LOI 1: Cell-35 — TRAINING MIXER SCENARIO 2 LAN THU 2 (DUPLICATE)**

- **Noi trong notebook:** Cell id="cell-35" (tiep sau cell id="cell-34")
- **Mo ta:** Notebook co HAI cell rieng biet deu train MLP-Mixer cho Scenario 2:
  - Cell-34 dung `make_tf_dataset()` + `mixer_s2.fit(train_ds_mixer_s2, ...)`
  - Cell-35 dung `mixer_s2.fit(X_train_s2, y_train_s2, ..., batch_size=BATCH_SIZE)` voi numpy array
- **Hau qua:**
  1. Cell-35 ghi de `ckpt_mixer_s2` va `cb_mixer_s2` bang gia tri moi, roi train TIEP tren `mixer_s2` da het epoch tu Cell-34. EarlyStopping se ket thuc ngay lap tuc o epoch 1 do val_accuracy khong cai thien.
  2. Checkpoint `best_mixer_s2.keras` bi ghi de boi model tu lan train thu 2 (co the kem hon).
  3. `hist_mixer_s2` bi ghi de boi history chi co 1–2 epoch — LC van hien thi nhung vo nghia.
  4. Voi numpy array trong Cell-35 + MirroredStrategy, neu batch cuoi khong chia het so GPU se gap loi "Input data shapes are inconsistent" hoac "Batch size must be divisible".
- **Fix: XOA TOAN BO Cell-35**

---

**LOI 2: Cell-40 — TRAINING LSTM SCENARIO 2 LAN THU 2 (DUPLICATE)**

- **Noi trong notebook:** Cell id="cell-40" (tiep sau cell id="cell-39")
- **Mo ta:** Tuong tu Loi 1, co HAI cell train LSTM Scenario 2:
  - Cell-39 dung `make_tf_dataset()` + `lstm_s2.fit(train_ds_lstm_s2, ...)`
  - Cell-40 dung `lstm_s2.fit(feat_train_s2_lstm, y_train_s2, ..., batch_size=BATCH_SIZE)` voi numpy array
- **Hau qua:**
  1. `hist_lstm_s2` bi ghi de boi history lan 2 — LC Cell-41 chi hien thi 1–2 epoch.
  2. Checkpoint `best_lstm_s2.keras` bi ghi de boi model kem hon tu lan train thu 2.
  3. Nhung quan trong hon: Cell-41 goi `plot_learning_curve(hist_lstm_s2, ...)` — history nay la cua lan train thu 2 tren numpy array, va lan train thu 2 nay TREN LSTM DA DAM BAO NGHIEM TRONG (voi validation_data truyen sai, xem Loi 3).
- **Fix: XOA TOAN BO Cell-40**

---

**LOI 3: `lstm_probs_train_s2` VA `lstm_probs_test_s2` CHUA DUOC DINH NGHIA (NameError)**

- **Noi trong notebook:** Cell-42 su dung bien `lstm_probs_train_s2` va `lstm_probs_test_s2`
- **Mo ta:** Trong Scenario 1, sau khi train LSTM (Cell-21), notebook co Cell-23 reload best LSTM va tinh probs:
  ```python
  best_lstm_s1 = tf.keras.models.load_model(ckpt_lstm_s1)
  lstm_probs_train_s1 = best_lstm_s1.predict(feat_train_s1_lstm, ...)
  lstm_probs_test_s1  = best_lstm_s1.predict(feat_test_s1_lstm, ...)
  ```
  Trong Scenario 2, sau Cell-39/40 (training LSTM S2), KHONG CO CELL TUONG DUONG. Cell-42 ngay sau do dung `lstm_probs_train_s2` — bien nay CHUA BAO GIO duoc gan gia tri.
- **Hau qua:** `NameError: name 'lstm_probs_train_s2' is not defined` — notebook crash tai Cell-42.
- **Fix: THEM CELL MOI giua Cell-39 (sau khi xoa Cell-40) va Cell-41:**
  ```python
  # Cell — Scenario 2: reload best LSTM + get probabilities (Rule 10)
  best_lstm_s2 = tf.keras.models.load_model(ckpt_lstm_s2)
  lstm_probs_train_s2 = best_lstm_s2.predict(feat_train_s2_lstm, batch_size=BATCH_SIZE, verbose=0)
  lstm_probs_test_s2  = best_lstm_s2.predict(feat_test_s2_lstm,  batch_size=BATCH_SIZE, verbose=0)
  print("LSTM S2 probs:", lstm_probs_train_s2.shape, lstm_probs_test_s2.shape)
  ```

---

**LOI 4: `MIXER_LC_S2` CHUA DUOC DINH NGHIA — NAMEERROR VA EXCEL TRONG**

- **Noi trong notebook:** Cell-46 dung `AUG_LEARNING_CURVE_PATH = MIXER_LC_S2`; Cell-50 dung bien nay de nhung anh vao Excel
- **Mo ta:** Notebook Scenario 1 co Cell-17 goi `plot_learning_curve(hist_mixer_s1, "Mixer S1", MIXER_LC_S1)` va dinh nghia `MIXER_LC_S1`. Scenario 2 THIEU cell tuong duong cho Mixer — bien `MIXER_LC_S2` chua bao gio duoc khai bao.
- **Hau qua:**
  1. Cell-46 se gap `NameError: name 'MIXER_LC_S2' is not defined` — notebook crash truoc khi den Cell-50.
  2. Neu bypass bang cach cung cap duong dan thu cong, file `MIXER_LC_S2` cung se khong ton tai nen anh LC Augmented trong Excel se trong.
- **Fix: THEM CELL MOI sau Cell-36 (reload Mixer S2 probs):**
  ```python
  # Cell — Scenario 2: MLP-Mixer learning curve (Rule 6)
  MIXER_LC_S2 = os.path.join(WORK_DIR, "mixer_s2_learning_curve.png")
  plot_learning_curve(hist_mixer_s2, "Mixer S2", MIXER_LC_S2)
  ```

---

### LOGIC ERROR (khong crash nhung cho ra ket qua sai)

**LOI 5: LSTM SCENARIO 1 NHAN SAI VALIDATION DATA (Cell-21)**

- **Noi trong notebook:** Cell-21, dong `validation_data=(X_val, y_val)`
- **Mo ta:** LSTM model co input shape `(1, FEATURE_DIM)` — nhan CNN features. Nhung Cell-21 truyen `X_val` co shape `(N, 64, 64, 3)` — raw images — vao validation_data.
- **Hau qua:** TensorFlow se raise `ValueError: Input 0 of layer "lstm" is incompatible with the layer: expected shape=(None, 1, FEATURE_DIM), found shape=(None, 64, 64, 3)`. Notebook crash ngay khi bat dau epoch dau tien cua LSTM training.
- **Fix (Cell-21):** Thay `validation_data=(X_val, y_val)` bang `validation_data=(feat_val_lstm, y_val)`.
  Note: `feat_val_lstm` da duoc trich va dinh nghia trong Cell-19 — bien nay san sang.

> Luu y: Loi 5 se xay ra truoc Loi 3 va Loi 4 khi chay tu dau. Day chinh la nguyen nhan "loi hoai" ma user bao cao.

---

### NEN SUA (anh huong chat luong ket qua)

**Van de 6: LSTM Scenario 2 cung co the gap loi validation data tuong tu (Cell-39)**

Cell-39 dung `validation_data=(feat_val_lstm, y_val)` — dung! Nhung `feat_val_lstm` duoc dinh nghia tai Cell-19 (Scenario 1). Sau `tf.keras.backend.clear_session()` tai Cell-29, Keras session bi xoa nhung bien Python `feat_val_lstm` KHONG bi xoa (Python variable scope). Vay nen Cell-39 van hoat dong dung — KHONG bi loi.

**Van de 7: Training Mixer S2 khong nhat quan (Cell-34 dung tf.data, nhung sau do co Cell-35 dung numpy)**

Sau khi xoa Cell-35 (Loi 1), Mixer S2 se chi dung tf.data (drop_remainder=True). Tuy nhien Mixer S1 (Cell-16) van dung numpy array truc tiep. Neu chay tren multi-GPU Kaggle T4x2, Cell-16 co the gap loi batch size khong chia het so GPU. Khuyen nghi: doi Cell-16 sang dung `make_tf_dataset()` de nhat quan voi S2.

**Van de 8: GELU trong token mixing**

Token mixing tai Cell-9 co `Dense(d, activation="gelu", name=f"tok_fc1_{i}")` — GELU co trong tok_fc1. Paper mo ta (Muc IV.D.1.a): "An MLP in the mixer layer consists of two fully-connected (FCN) layers with Gaussian Error Linear Unit (GELU) non-linearity" — am chi GELU dung cho ca 2 MLP (token va channel mixing). Hien trang: tok_fc2 khong co activation. Day la sai lech nhug nho.

---

### TUY CHON (cai thien do trung thuc)

- **Aug 9:** Them `width_shift_range=0.1` vao `aug_gen` trong Cell-31 de bam sat paper (Muc IV.C co de cap "width-shift").
- **SVM C grid:** Them C=1000 (co trong paper text goc Muc V) vao `param_grid`.
- **Mixer S1 consistency:** Doi Cell-16 sang dung `make_tf_dataset()` tuong tu S2 de tranh loi multi-GPU.

---

## 8. GHI CHU BO SUNG

### Thu tu cell thuc thi va van de tuong ung

Khi chay notebook tu dau den cuoi, cac loi se xuat hien theo thu tu sau:

```
Cell-21 (LSTM S1 training):
  -> CRASH: ValueError shape mismatch (X_val thay vi feat_val_lstm)
  [Loi 5 — phai sua truoc tien]

Cell-34 (Mixer S2 training lan 1): OK
Cell-35 (Mixer S2 training lan 2 — DUPLICATE):
  -> Ghi de checkpoint, hist_mixer_s2 mat
  -> Co the CRASH voi numpy array + MirroredStrategy
  [Loi 1 — xoa Cell-35]

Cell-39 (LSTM S2 training lan 1): OK (feat_val_lstm van con trong scope)
Cell-40 (LSTM S2 training lan 2 — DUPLICATE):
  -> Ghi de checkpoint, hist_lstm_s2 mat
  -> validation_data=(X_val, y_val) — same bug nhu Loi 5 nhung day la lan 2
  [Loi 2 — xoa Cell-40]

Cell-41 (LSTM LC S2): Plot history sai (chi 1-2 epoch)

[Thieu cell: reload LSTM S2 + lay lstm_probs_train/test_s2]

Cell-42:
  -> CRASH: NameError: lstm_probs_train_s2 is not defined
  [Loi 3 — them cell reload LSTM S2]

Cell-46:
  -> CRASH: NameError: MIXER_LC_S2 is not defined
  [Loi 4 — them cell plot Mixer S2 LC]
```

### Tom tat 5 thay doi bat buoc (theo thu tu uu tien)

1. **SUA Cell-21:** Doi `validation_data=(X_val, y_val)` thanh `validation_data=(feat_val_lstm, y_val)` — sua Loi 5 (crash dau tien khi chay)
2. **XOA Cell-35:** Training Mixer S2 lan thu 2 — sua Loi 1
3. **XOA Cell-40:** Training LSTM S2 lan thu 2 — sua Loi 2
4. **THEM CELL sau Cell-39 (post-xoa-Cell-40):** Reload best LSTM S2 + lay lstm_probs_train/test_s2 — sua Loi 3
5. **THEM CELL sau Cell-36:** Plot Mixer S2 LC va dinh nghia MIXER_LC_S2 — sua Loi 4

### Phan tich tai sao co duplicate cells

Kha nang cao la trong qua trinh chinh sua notebook, agent tao Scenario 2 da copy Scenario 1 va chinh sua, nhung quen xoa phien ban cu. Ca Mixer S2 va LSTM S2 deu bi loi nay — cho thay day la van de he thong trong qua trinh sinh notebook, khong phai loi ngau nhien.

### Kien truc patch embedding — han che noi tai

Diem khac biet kien truc lon nhat (A6/A14) la co che `_compression` cua paper co 8272 params (co the la DepthwiseConv hoac mot Conv2D nho 3x3 stride) nhung notebook dung Dense(1728, 128) co 221K params. Day la han che vi paper khong mo ta bang text — khong phai loi cua notebook. Nguoi doc can biet: so params thuc te cua Mixer trong notebook se cao hon so paper ~27%, va tong meta-ensemble params se vuot 1M.
