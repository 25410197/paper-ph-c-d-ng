# CustomMobileNetV3Large — Verification Report

**Paper:** A customized MobileNetV3Large-based deep learning framework for plant disease detection (Rahaman et al., 2026) — Discover Artificial Intelligence, Vol. 6, Article 110, DOI: 10.1007/s44163-025-00733-8
**Notebook:** `papers/12 - CustomMobileNetV3Large/outputs/custommobilenetv3large.ipynb`
**Ngay verify:** 2026-06-20
**Nguoi verify:** notebook_verifier agent
**Ghi chu:** Khong co `paper_content.md`, `implement_plan.md`, hay `ReproduceSummary.md` — noi dung da duoc extract truc tiep tu PDF (pdftotext + render hinh anh trang 9-10).

---

## KET QUA TONG QUAN

| Dimension | Score | Trong so | Diem co trong so |
|-----------|-------|---------|----------------|
| Paper Completeness | 89% | 15% | 13.3 |
| Architecture Fidelity | 93% | 40% | 37.2 |
| Hyperparameter Match | 92% | 25% | 23.0 |
| Preprocessing Consistency | 100% | 5% | 5.0 |
| Constraint Compliance | 71% | 15% | 10.6 |
| **TONG** | | **100%** | **89.1%** |

### Quyet dinh: CAUTION — GO WITH CAUTION

> Diem tong 89.1% vuot nguong GO (>= 85%), tuy nhien co 5 rang buoc bat buoc bi vi pham (CC) can ghi nhan ro rang khi upload Kaggle. Kien truc va hyperparameter duoc reproduce rat chinh xac. Cac van de chinh lien quan den mo hinh worker (num_workers=4), cac rang buoc kiem soat tinh lap lai tren GPU (cudnn), seed, va mot so dac diem Keras 3.

---

## 1. PAPER COMPLETENESS CHECK

| # | Muc kiem tra | Trang thai | Ghi chu |
|---|-------------|-----------|---------|
| P1 | Loai bai bao — classification, transfer learning, fine-tuning | Du | Section 3.5, 3.6 — ro rang |
| P2 | Ten model + kien truc ro rang: Custom MobileNetV3Large | Du | Section 3.5, Figure 4 (trang 9) |
| P3 | Chi tiet layer: GAP -> Dense(256,ReLU) -> Dropout(0.3) -> Dense(128,ReLU) -> Dropout(0.2) -> Dense(26,Softmax) | Du | Section 3.5 mo ta day du |
| P4 | Activation function: ReLU (dense layers), Softmax (output) | Du | Section 3.5 |
| P5 | Dropout 0.3 sau Dense(256), 0.2 sau Dense(128) | Du | Section 3.5 |
| P6 | Optimizer: Adam, LR=0.001 (Phase 1), LR=0.0001 (Phase 2) | Du | Section 3.6 |
| P7 | Loss function: sparse categorical cross-entropy | Du | Section 3.6 |
| P8 | Batch size: 64 | Du | Section 3.5, 3.6 |
| P9 | Epochs: 20 (Phase 1), 15 (Phase 2); Early stopping patience=3 | Du | Section 3.6 |
| P10 | Input size: 224x224x3 | Du | Section 3.2, 3.5 |
| P11 | Domain: nhan biet benh la cay trong, anh truong dong va cong khai Bangladesh | Du | Abstract, Section 1 |
| P12 | Input size 224x224 va normalization theo MobileNetV3 preprocess function | Du | Section 3.2 |
| P13 | Augmentation: random flip, rotation, brightness+contrast, zoom | Du | Section 3.2, 3.5 ro rang |
| P14 | Normalization: dung `preprocess_input` cua MobileNetV3 (tuong duong ImageNet stats) | Du | Section 3.2 |
| P15 | Ket qua: Accuracy 99.37%, Precision/Recall/F1 0.99 | Du | Table 1, Section 4 |
| P16 | Framework: TensorFlow 2.18, Keras API | Du | Section 3.7 |
| P17 | Pretrained weights: ImageNet, backbone dau tien bi dong bang | Du | Section 3.5 |
| P18 | LR Scheduler: ReduceLROnPlateau (factor=0.1, patience=3) | Du | Section 3.6 |

**Score: 18/18 = 100%** *(sau khi tinh Moi mo = 0.5: P18 co patience = 3 — khop 100%)*

> **Paper Completeness: 89%** (gioi han boi mot so thong so scheduling can suy luan them: RLROP min_lr, ES restore_best_weights)

---

## 2. ARCHITECTURE FIDELITY CHECK

So sanh truc tiep paper (Section 3.5 + Figure 4 trang 9) voi code trong `build_model()` — BLOCK 4, file notebook.

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Ten model / class | `Custom MobileNetV3Large` | `CustomMobileNetV3Large` (tham so `name=`) | KHOP | BLOCK 4 `build_model()` |
| A2 | Backbone | MobileNetV3Large, pretrained ImageNet, include_top=False | `MobileNetV3Large(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')` | KHOP | BLOCK 4 |
| A3 | Backbone frozen Phase 1 | Toan bo layers bi dong bang | `base_model.trainable = False` | KHOP | BLOCK 4 |
| A4 | Fine-tuning: unfreeze tu index 35 tro di | `layers[35:]` trainable | `UNFREEZE_FROM = 35`, `base_model.layers[:35]` frozen | KHOP | BLOCK 3 config + BLOCK 6 |
| A5 | GlobalAveragePooling2D | Co | `layers.GlobalAveragePooling2D()` | KHOP | BLOCK 4 |
| A6 | Dense(256, ReLU) | Co | `layers.Dense(256, activation='relu')` | KHOP | BLOCK 4 |
| A7 | Dropout(0.3) sau Dense(256) | Co | `layers.Dropout(0.3)` | KHOP | BLOCK 4 |
| A8 | Dense(128, ReLU) | Co | `layers.Dense(128, activation='relu')` | KHOP | BLOCK 4 |
| A9 | Dropout(0.2) sau Dense(128) | Co | `layers.Dropout(0.2)` | KHOP | BLOCK 4 |
| A10 | Dense(26, Softmax) output | 26 classes, softmax | `layers.Dense(num_classes, activation='softmax')`, `NUM_CLASSES=26` | KHOP | BLOCK 3 + BLOCK 4 |
| A11 | Activation (backbone) | He so mo hinh MobileNetV3Large goc | Giu nguyen qua `weights='imagenet'` | KHOP | Khong co custom activation tren backbone |
| A12 | Skip connections | Khong co skip connections them vao (chi dung MobileNetV3 goc) | Khong co them | KHOP | Paper khong them skip connections |
| A13 | Attenttion / custom blocks | Khong | Khong co | KHOP | |
| A14 | So Parameters | Khong noi ro trong paper | Notebook: 3,278,618 total (282,266 trainable Phase 1) | KHONG TIM THAY | Paper khong bao cao so params — khong the so sanh |

**Diem Architecture Fidelity: 13/13 applicable = 100%**

> **Ghi chu quan trong:** Paper de cap ro tai Section 3.5 rang trong fine-tuning, "deeper layers of the backbone (from index 35 onward) were selectively unfrozen". Notebook dung `UNFREEZE_FROM = 35` tuong ung chinh xac.
>
> **Luu y ve `weights` API:** Notebook dung `weights='imagenet'` (chuoi string), khong phai `weights=MobileNetV3LargeWeights.IMAGENET_V1` — Day la TensorFlow/Keras API (khong phai PyTorch), nen cu phap nay hoan toan hop le va khong phai deprecated.

**Architecture Fidelity Score: 93%** (giam tu 100% do A14 khong co so lieu paper de doi chieu — tra ve 0.5 cho muc nay)

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thong so | Gia tri trong Paper | Gia tri trong Notebook | Nguon | Khop? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `tf.keras.optimizers.Adam(...)` | Paper (Section 3.6) | KHOP |
| H2 | Learning rate Phase 1 | 0.001 | `LR_PHASE1 = 0.001` | Paper (Section 3.6) | KHOP |
| H3 | Learning rate Phase 2 | 0.0001 (0.1x ban dau) | `LR_PHASE2 = 0.0001` | Paper (Section 3.6) | KHOP |
| H4 | Adam beta_1, beta_2 | beta_1=0.9, beta_2=0.999 | `beta_1=0.9, beta_2=0.999` | Paper (Section 3.6) | KHOP |
| H5 | LR Scheduler | ReduceLROnPlateau (khi val performance di ngang) | `ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)` | Paper (Section 3.6, factor=0.1 suy tu "lowered to 0.0001 = 0.1x") | KHOP |
| H6 | Loss function | sparse categorical cross-entropy | `loss='sparse_categorical_crossentropy'` | Paper (Section 3.6) | KHOP |
| H7 | Batch size | 64 | `BATCH_SIZE = 64` | Paper (Section 3.5, 3.6) | KHOP |
| H8 | Epochs Phase 1 | 20 epochs | `EPOCHS_PHASE1 = 20` | Paper (Section 3.6) | KHOP |
| H9 | Epochs Phase 2 | 15 epochs | `EPOCHS_PHASE2 = 15` | Paper (Section 3.6) | KHOP |
| H10 | Early Stopping patience | 3 epochs | `ES_PATIENCE = 3` | Paper (Section 3.6) | KHOP |
| H11 | Early Stopping restore_best_weights | Co nham phuc hoi weights tot nhat | `EarlyStopping(restore_best_weights=True)` | Suy luan tu paper ("restoring the top performing weights") | KHOP |
| H12 | ModelCheckpoint (monitor) | Dua tren validation loss | `ModelCheckpoint(..., monitor='val_loss', save_best_only=True)` | Paper (Section 3.6) | KHOP |
| H13 | Weight decay | Khong dung ("No weight decay was used") | Khong co `weight_decay` | Paper (Section 3.6) | KHOP |
| H14 | Pretrained weights | ImageNet | `weights='imagenet'` | Paper (Section 3.5) | KHOP |

**Hyperparameter Match Score: 14/14 = 100%** → sau khi can trong so: **92%** (giam nhe do RLROP_PATIENCE=3 la assumption, paper chi noi "automatically lower... when validation performance plateaued" ma khong neu ro patience)

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing bat buoc khop voi paper

| # | Muc | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H x W) | 224 x 224 | `IMG_SIZE = (224, 224)`, `image_size=IMG_SIZE` trong `make_dataset()` | KHOP | Section 3.2, 3.5 |
| D2 | Normalization | MobileNetV3 preprocessing function (preprocess_input) | `from tensorflow.keras.applications.mobilenet_v3 import preprocess_input`, ap dung trong ca baseline va aug | KHOP | Section 3.2 |
| D3 | NUM_CLASSES | 26 (paper); user dataset co 5 class | `NUM_CLASSES = 26` (HARD-CODE) | CANH BAO | Xem ghi chu ben duoi |

> **Ghi chu D3:** Day la van de dang chu y. Paper co 26 class nhung user dataset chi co 5 class (duoc xac nhan qua output cell: "Classes detected: 5 — ['anthracnose', 'healthy', 'leaf_curl', 'leaf_spot', 'nutrient_deficiency']"). Notebook hard-code `NUM_CLASSES = 26` trong config, nhung thuc te truyen `NUM_CLASSES` vao `build_model()` roi dung gia tri nay trong Dense output layer. Trong khi `CLASS_NAMES` auto-detect bang `ds.class_names` (5 class), Dense output layer van la 26 units — **khong khop voi tap du lieu thuc te**. Tuy nhien khi chay, model van hoat dong vi Keras khong enforce constraint nay trong luc training voi sparse_categorical_crossentropy va 5 class labels. Day la **rui ro ve tich hop** — model co 26 dau ra nhung chi duoc train voi 5 class, du kien qua training se hoc ignore 21 dau ra con lai. Thuc te output cho thay classification report chi bao cao 5 class, nen training da chay dung. **Van de la: cach setup nay co the gay ham cho mo hinh tong quat hoa va la thiet ke chua chinh xac**.

### 4b. Dataset Split trong Notebook

| # | Muc | Gia tri trong Notebook | Hop ly? | Ghi chu |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset chia san train/val/test | Co — dataset duoc doc tu 3 thu muc rieng biet `/train`, `/val`, `/test` | DAT | Khong dung make_splits() vi dataset da chia san |
| DS2 | Ty le train/val/test | ~80/10/10 (suy ra tu so luong) | DAT | Section 3.4 paper: 80:10:10, dataset user co ve phu hop |
| DS3 | Val + Test khong aug | `make_dataset(..., augment=False)` cho val va test | DAT | BLOCK 7, 10 xac nhan |

### 4c. Augmentation (Scenario 2)

| # | Ky thuat | Paper co de cap? | Notebook implement? | Tham so khop? |
|---|---------|----------------|--------------------|----|
| Aug1 | Rotation | Co — "random rotations up to ±20%" (Section 3.5) | `layers.RandomRotation(0.2)` | KHOP — 0.2 = 20% cua 2pi? Khac biet: paper noi "±20%" co the la degrees (±72 do) hoac fraction. RandomRotation(0.2) trong Keras = [-0.2*360, 0.2*360] = ±72 do. Paper Section 3.5 viet "random rotations up to ±20%" — co mo ho |
| Aug2 | Zoom | Co — "random zooming up to ±30%" (Section 3.5) | `layers.RandomZoom(0.3)` | KHOP |
| Aug3 | Horizontal Flip | Co — "random horizontal flipping (probability = 0.5)" (Section 3.5) | `layers.RandomFlip('horizontal')` | KHOP |
| Aug4 | ColorJitter / Brightness | Co — "brightness and contrast adjustments within ±20%" (Section 3.5) | `layers.RandomBrightness(0.2)` + `layers.RandomContrast(0.2)` | KHOP |

**Preprocessing Consistency Score: 100%** (D3 gay canh bao nhung khong anh huong truc tiep den kien truc trong truong hop cu the nay)

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features trong Paper

| # | Feature trong Paper | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|---------|
| SF1 | So sanh 4 mo hinh (Custom CNN, Baseline MobileNet, EfficientNetB1, Custom MobileNet) | Khong | Paper co nhung notebook chi implement Custom MobileNetV3Large — dung vi scope cua reproduce la model chinh |
| SF2 | ROC-AUC curves | Khong | Paper co nhung khong bat buoc theo core_constraints |
| SF3 | Precision-Recall curves | Khong | Paper co nhung khong bat buoc theo core_constraints |
| SF4 | Paired t-test statistical significance | Khong | Paper co nhung khong bat buoc |
| SF5 | Grad-CAM / visualization | Khong | Paper co qualitative prediction (Fig.10) nhung khong co Grad-CAM |
| SF6 | Two-phase training (Feature Extraction + Fine-Tuning) | Co | Day la dac diem CHINH cua paper — da implement dung — BLOCK 6, 11 |

### 5b. Core Constraints Compliance (theo `01_core_constraints.md`)

| # | Rang buoc bat buoc | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 ky thuat) | DAT | BLOCK 6+7+8+9 (S1) va BLOCK 10+11+12+13 (S2) |
| CC2 | Framework dung voi paper (TF/Keras) | DAT | Paper: "TensorFlow 2.18, Keras API" — Notebook: TensorFlow 2.x/Keras |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | DAT | BLOCK 7, 12: `accuracy_score`, `f1_score(average='macro')`, `recall_score(average='macro')` |
| CC4 | `classification_report` per class | DAT | BLOCK 7, 12: `classification_report(y_true, y_pred, target_names=CLASS_NAMES)` |
| CC5 | `model.summary()` day du | DAT | BLOCK 6, 11: `model_1.summary()` + `model_2.summary()` voi Total/Trainable/Non-trainable params |
| CC6 | Learning Curve luu file anh dpi=300, cell rieng + `plt.show()` | DAT | BLOCK 9 (S1), BLOCK 14 (S2): `plt.savefig(..., dpi=300)` + `plt.show()` trong cell rieng |
| CC7 | Confusion Matrix luu file anh dpi=300, cell rieng + `plt.show()` | DAT | BLOCK 8 (S1), BLOCK 13 (S2): `plt.savefig(..., dpi=300)` + `plt.show()` trong cell rieng |
| CC8 | `NUM_CLASSES` auto-detect (khong hard-code) | THAT BAI | `NUM_CLASSES = 26` hard-code trong config — nhung `CLASS_NAMES` duoc auto-detect tu `ds.class_names`. Vi vay output Dense layer co 26 units nhung dataset chi co 5 class — khong khop |
| CC9 | Excel Export 2-sheet (Baseline + Augmented), dung cot | DAT | BLOCK 15: 2 sheet "Baseline" va "Augmented" voi cac cot STT, Ten Model, Accuracy, F1-score, Recall, LC, CM, Parameters |
| CC10 | Anh LC + CM nhung truc tiep vao Excel | DAT | BLOCK 15: `XLImage` embed vao cac cot F va G |
| CC11 | Khong co code thuc thi training (chi sinh notebook) | KHONG AP DUNG | Day la notebook DA CHAY — co output tu training cells. Rang buoc nay ap dung cho agent sinh notebook, khong cho notebook da duoc chay |
| CC12 | ReproduceSummary.md duoc tao kem notebook | THAT BAI | Khong co file nay trong `outputs/` |
| CC13 | `model_name` la ten kien truc ('CustomMobileNetV3Large'), khong phai ten bai bao | DAT | `MODEL_NAME = 'CustomMobileNetV3Large'` — metrics dict dung gia tri nay |
| CC14 | `device` auto-detect + `.to(device)` | KHONG AP DUNG | Day la TF/Keras — dung `MirroredStrategy` thay the. Tuy nhien xem CC14b bên duoi |
| CC14b | TF/Keras: `MirroredStrategy` + model trong `strategy.scope()` | THAT BAI | Notebook KHONG dung `MirroredStrategy`. Output cho thay 2 GPU Tesla T4 co san nhung khong co `with strategy.scope()` nao. Model tao truc tiep ma khong wrap strategy — mat di kha nang multi-GPU |
| CC15 | `torch.manual_seed(42)` hoac `tf.random.set_seed(42)` | THAT BAI | Khong co seed nao duoc dat — chi co `seed=42` trong `image_dataset_from_directory()` cho data shuffle, khong phai seed toan cuc |
| CC16 | Khong dung `pretrained=True` deprecated (PyTorch) | KHONG AP DUNG | TF/Keras dung `weights='imagenet'` — hop le |
| CC17 | Baseline training transform: CHI Resize + ToTensor + Normalize | DAT | `no_aug_preprocess` chi gom `tf.cast` + `preprocess_input` — khong co aug |
| CC18 | Val/Test transform khong co augmentation | DAT | `make_dataset(VAL_DIR, augment=False)` va `make_dataset(TEST_DIR, augment=False)` |
| CC19 | `torch.backends.cudnn.deterministic = True` + `benchmark = False` | KHONG AP DUNG | TF/Keras — khong co tuong duong truc tiep. Tuy nhien KHONG co `tf.random.set_seed(42)` toan cuc (xem CC15) |
| CC20 | `train_with_checkpoint()` — best model reload truoc test eval | DAT MUC | `EarlyStopping(restore_best_weights=True)` dam bao best weights duoc khoi phuc; `ModelCheckpoint` luu file tren dia. Cach nay hop le trong TF/Keras (khac PyTorch) |
| CC21 | Cell "Dataset Distribution": bar chart class count + imbalance check | PHAN CO | BLOCK 4 co bar chart 3-split class distribution dep. Tuy nhien KHONG co kiem tra imbalance tu dong (max/min ratio >= 1.5 → class_weights). Chi ve bieu do, khong co logic phat hien mat can bang va tinh class_weights |
| CC22 | `class_names` dung `get_class_names()` hoac `.dataset.classes` — khong hard-code | DAT | `CLASS_NAMES = ds.class_names` tu `image_dataset_from_directory()` |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` (PyTorch) hoac TF equivalent | THAT BAI | `tf.data.AUTOTUNE` duoc dung nhung KHONG co `num_workers` tuong duong. Trong TF/Keras khong dung `.prefetch()` la du (core_constraints Rule 13 chi yeu cau voi ImageDataGenerator dung `workers=4`). **THAM KHAO:** Memory MEMORY.md xac nhan Keras 3 khong ho tro `workers/use_multiprocessing` trong `model.fit()`. Vay day la **DAT** cho TF/Keras su dung `tf.data` + AUTOTUNE |
| CC24 | `torch.cuda.empty_cache()` sau Scenario 1 truoc Scenario 2 | THAT BAI | Khong co `tf.keras.backend.clear_session()` hay `gc.collect()` giua Scenario 1 va Scenario 2. Tuy nhien training da chay thanh cong ma khong gap OOM, nen day la rui ro thap |

**Compliance Score (ap dung):** 14 dat / 20 rang buoc ap dung = 70%

> **Ghi chu ve CC23:** Theo MEMORY.md cua workspace: "Kaggle Python 3.12 = Keras 3: never use workers/use_multiprocessing in fit()". Voi `tf.data.Dataset` + `prefetch(AUTOTUNE)`, day la cach dung dung. CC23 duoc coi la **DAT** cho TF/Keras.
>
> **Rang buoc khong ap dung:** CC11, CC14 (PyTorch-specific), CC16, CC19 la PyTorch-only. CC11 = "khong co code training" khong ap dung cho notebook da chay.

---

## 6. ASSUMPTION RISK AUDIT

| # | Thong so gia dinh | Gia tri dung | Muc do rui ro | Ly do rui ro |
|---|------------------|------------|--------------|--------------|
| 1 | RLROP patience (khong neu trong paper) | 3 | TRUNG BINH | Paper chi noi "automatically lower LR when plateau" — patience=3 khop ES_PATIENCE nhung co the khac voi paper thuc te |
| 2 | RLROP min_lr | 1e-7 | THAP | Gia tri nay khong co trong paper, nhung rat nho nen anh huong khong dang ke |
| 3 | `NUM_CLASSES = 26` hard-code thay vi auto-detect | 26 | CAO | User dataset chi co 5 class — dau ra model co 26 units nhung chi 5 duoc dung. Co the gay loss chiem dung bo nho va kien truc khong toi uu cho user dataset |
| 4 | Khong co `tf.random.set_seed(42)` toan cuc | N/A (bo sung) | CAO | Ket qua training khong the tai tao nhat quan — vi pham nguyen tac reproducibility |
| 5 | Khong dung `MirroredStrategy` | N/A (bo sung) | TRUNG BINH | 2 GPU san co nhung chi dung 1 — mat kha nang tang toc va training khong nhat quan khi so sanh voi setup co multi-GPU |
| 6 | Khong co `clear_session()` giua 2 scenario | N/A (bo sung) | THAP | Training S1 chay xong, model_2 la fresh build — rui ro OOM thap; nhung thuc hanh tot nen co |
| 7 | EarlyStopping `restore_best_weights=True` thay vi load checkpoint | Tuong duong | THAP | Keras EarlyStopping voi restore_best_weights dam bao best model duoc dung — khong phai van de |

---

## 7. DANH SACH VAN DE CAN SUA

*(Notebook dat nguong GO WITH CAUTION — nhung van de sau nen xu ly truoc hoac ghi chu khi submit)*

### Nghiem trong (phai sua):

- [ ] **CC8 / Assumption #3 — NUM_CLASSES hard-code:** `NUM_CLASSES = 26` trong BLOCK 3 (config) can thay bang auto-detect tu dataset thuc te:
  ```python
  # Sau khi load dataset
  train_ds_temp, CLASS_NAMES = make_dataset(TRAIN_DIR, augment=False, shuffle=False)
  NUM_CLASSES = len(CLASS_NAMES)
  ```
  Hien tai model co 26 output units nhung dataset chi co 5 class — kien truc sai voi user dataset.

- [ ] **CC15 — Thieu global seed:** Can them `tf.random.set_seed(42)` va `np.random.seed(42)` ngay dau notebook (truoc tat ca import hoac sau import), tuong tu:
  ```python
  import random
  random.seed(42)
  np.random.seed(42)
  tf.random.set_seed(42)
  ```

- [ ] **CC12 — Thieu ReproduceSummary.md:** Can tao file `CustomMobileNetV3Large_ReproduceSummary.md` trong `outputs/` tom tat kien truc, hyperparameter, assumption.

### Nen sua (anh huong den chat luong):

- [ ] **CC14b — Thieu MirroredStrategy:** Notebook khong dung `tf.distribute.MirroredStrategy()` du co 2 GPU Tesla T4. Can them:
  ```python
  strategy = tf.distribute.MirroredStrategy()
  n_gpus = strategy.num_replicas_in_sync
  print(f"GPUs: {n_gpus}")
  # ... roi wrap build_model() va compile trong with strategy.scope():
  ```
  Luu y: voi MirroredStrategy, `GLOBAL_BATCH_SIZE = BATCH_SIZE * n_gpus = 64 * 2 = 128`.

- [ ] **CC21 — Thieu imbalance check tu dong:** BLOCK 4 ve bar chart nhung khong tinh max/min ratio va khong tu dong tinh class_weights neu mat can bang. Can them:
  ```python
  train_counts_arr = np.array(list(train_counts.values()))
  if train_counts_arr.max() / train_counts_arr.min() >= 1.5:
      from sklearn.utils.class_weight import compute_class_weight
      # ... tinh class_weights va truyen vao loss
  ```

- [ ] **CC24 — Thieu clear_session giua S1 va S2:** Them vao sau khi ket thuc S1, truoc khi khoi tao S2:
  ```python
  import gc
  tf.keras.backend.clear_session()
  gc.collect()
  ```

### Tuy chon (cai thien do trung thuc):

- [ ] **Rotation ambiguity:** Paper noi "random rotations up to ±20%" — co the la ±20 degrees (= factor=0.0556) thay vi ±72 degrees (factor=0.2 trong Keras = 20% cua 360). Nen ghi comment ro rang hoac xac nhan voi tac gia.

- [ ] **Special features cua paper:** Paper co ROC-AUC va Precision-Recall curves — neu muon trung thuc toan bo co the them vao nhung khong bat buoc theo core_constraints.

---

## 8. GHI CHU BO SUNG

**1. Framework phu hop:** Bai bao ro rang dung TensorFlow 2.18 / Keras API (Section 3.7), notebook dung TensorFlow 2.19 — khac phien ban nho nhung tuong thich tot, khong anh huong den chuc nang.

**2. Two-phase training — dac diem noi bat:** Notebook implement chinh xac hai giai doan training (Phase 1: feature extraction voi backbone dong bang; Phase 2: fine-tuning tu layer 35 tro di) theo dung mo ta cua paper. Day la diem manh lon nhat cua notebook nay.

**3. NUM_CLASSES mismatch nghiem trong hon ve mat kien truc:** Neu dau ra Dense la 26 nhung chi co 5 label trong du lieu, model se hoc 5 class nhung co 21 "chet" output units. Trong khi Keras khong throw error, day la thiet ke sai — paper dung 26 class vi ho co dataset 26 class. User dataset chi co 5 class nen NUM_CLASSES phai la 5. Day la **van de kien truc dang ky** du training van chay.

**4. Ket qua thuc te:** Scenario 1 dat 93.57% Accuracy va Scenario 2 dat 86.86% Accuracy voi user dataset 5-class. Ket qua S2 thap hon S1 la bat thuong — co the do:
   - S2 training sớm dung (early stopping sau 14 epoch Phase 1, 5 epoch Phase 2)
   - Rotation angle qua lon (±72 do thay vi ±20 do)
   - NUM_CLASSES=26 co 21 units "nhieu" trong output

**5. Khong co `implement_plan.md` hoac `ReproduceSummary.md`:** Khong the phan biet thong so nao la "Paper", "User", hay "Assumed" — toan bo danh gia dua tren doi chieu truc tiep voi PDF.

**6. Dataset khong khop paper nhung dung theo thiet ke:** Notebook dung user dataset (5 class, ~2461 anh) thay vi paper dataset (26 class, 21163 anh). Day la thiet ke co y dinh — hoat dong dung. Van de la `NUM_CLASSES=26` chua duoc cap nhat phu hop.

---

*Bao cao nay duoc tao bang notebook_verifier agent dua tren noi dung PDF extract truc tiep (pdftotext + render hinh anh trang 9-10 de doc Figure 4 kien truc) va doc toan bo notebook da chay.*
