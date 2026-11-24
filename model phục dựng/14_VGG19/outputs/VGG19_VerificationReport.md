# VGG19 + InceptionV3 (Hybrid DL+ML) — Verification Report (Re-verify sau patch)

**Paper:** Sujatha et al. (2025) — *Advancing plant leaf disease detection integrating machine learning and deep learning*
**Journal:** Scientific Reports 15:11552 | DOI: 10.1038/s41598-024-72197-2
**Notebook:** `papers/14 - Hybrild ML + ML/outputs/vgg19.ipynb`
**Ngay verify:** 2026-06-20 (re-verify sau patch)
**Nguoi verify:** notebook_verifier agent

**Nhung thay doi da ap dung truoc lan re-verify nay:**
1. Them `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` (Rule 7)
2. Them `torch.backends.cudnn.deterministic = True` + `benchmark = False` (Rule 9)
3. Them `torchinfo.summary()` cho VGG19 va InceptionV3 (Rule 5)
4. Them `classification_report` per class trong vong lap `run_scenario()` (Rule 4)
5. Doi `plt.close()` → `plt.show()` trong hai ham plot (Rule 6)
6. Xoa hoan toan PCA — `preprocess_features()` chi dung `StandardScaler` (loai bo addition khong co trong paper)

---

## KET QUA TONG QUAN

| Dimension | Score | Trong so | Diem co trong so |
|-----------|-------|----------|-----------------|
| Paper Completeness | 50% | 15% | 7.5 |
| Architecture Fidelity | 86% | 40% | 34.4 |
| Hyperparameter Match | 70% | 25% | 17.5 |
| Preprocessing Consistency | 100% | 5% | 5.0 |
| Constraint Compliance | 89% | 15% | 13.3 |
| **TONG** | | **100%** | **77.7%** |

### Quyet dinh: GO WITH CAUTION

Sau khi ap dung 6 patch, notebook da khac phuc toan bo 4 van de "Nghiem trong" tu lan verify truoc: seed, cudnn deterministic, `torchinfo.summary()`, va `classification_report`. Ngoai ra PCA (addition khong co trong paper) da duoc loai bo — Architecture Fidelity tang tu 79% len 86%, Constraint Compliance tang tu 67% len 89%. Diem tong tang tu 76.8% len 77.7%.

**Luu y quan trong ve trang thai output cell:** Notebook da duoc patch o muc code nguon nhung chua duoc re-run toan bo. Cac cell output hien tai van con la output cu (con hien "PCA 256d | variance retained: 63.55%" — stale output tu truoc khi patch). Truoc khi upload Kaggle, can chay lai toan bo notebook tu dau de output phan anh dung code da patch.

Notebook van ton tai mot so han che can ghi ro truoc khi submit: (1) khong co `ReproduceSummary.md`, (2) `class_names` tu `os.listdir` thay vi `ImageFolder.classes`, (3) `num_workers=2` thay vi 4 theo Rule 13, (4) thieu imbalance check tu dong, (5) evaluation method khac paper (hold-out vs 10-fold CV).

---

## 1. PAPER COMPLETENESS CHECK

| # | Muc kiem tra | Trang thai | Ghi chu |
|---|-------------|-----------|---------|
| P1 | Loai bai bao (Nhom A/B/C) | Du | Nhom B — Hybrid: DL backbone lam feature extractor + ML classifier |
| P2 | Ten model + kien truc ro rang | Du | VGG19 va InceptionV3 duoc neu cu the, 5 ML classifier |
| P3 | Chi tiet tung layer / block | Du | Paper mo ta VGG19: 16 conv layer, 5 MaxPool, 3 FC, filter 3x3 stride 1, same padding, maxpool 2x2 stride 2 |
| P4 | Activation function | Mo ho | Khong neu ro trong paper nhung mac dinh la ReLU cho VGG19 |
| P5 | Batch Normalization / Dropout | Thieu | Khong de cap trong paper |
| P6 | Optimizer (loai + learning rate) | Thieu | Khong applicable — paper chi feature extraction + ML, khong co DL training loop |
| P7 | Loss function | Thieu | Khong applicable — ML classifiers co loss rieng nhung khong neu ro |
| P8 | Batch size | Thieu | Khong neu trong paper |
| P9 | So epochs hoac stopping criterion | Thieu | Khong applicable — ML models khong train bang epochs |
| P10 | Input size (H x W x C) | Du | VGG19: 224x224 neu ro; InceptionV3: 299x299 |
| P11 | Domain model | Du | Plant leaf disease detection ro rang |
| P12 | Input size va normalization | Mo ho | Input size ro, normalization (ImageNet mean/std) duoc su dung ngam dinh vi dung pretrained |
| P13 | Augmentation | Thieu | Paper KHONG mo ta augmentation. Chi extract features tu raw images |
| P14 | Normalization values (mean/std) | Thieu | Khong neu cu the trong paper |
| P15 | Ket qua thuc nghiem | Du | Tables 2-5 co Accuracy, Precision, Recall, F1, AUC, MCC day du cho 4 datasets |
| P16 | Framework | Thieu | Khong neu ro PyTorch hay TF hay sklearn |
| P17 | Pretrained weights | Du | Neu ro la dung pre-trained models (ImageNet) |
| P18 | LR Scheduler | Thieu | Khong applicable |

**Diem tong: 6.5 / 13 applicable = 50%**
*(P6, P7, P8, P9, P18 khong applicable cho hybrid feature-extraction + ML approach — loai ra khoi mau so)*

**Phan loai:** Bai bao CO THE reproduce — thong so chinh (kien truc backbone, ML classifier list, evaluation metrics) du ro rang. Cac thong so bo sung (batch size, normalization values, augmentation) phai gia dinh.

---

## 2. ARCHITECTURE FIDELITY CHECK

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----------|---------------------|----------------------|-------|---------|
| A1 | Backbone 1 | VGG19 (pretrained, feature extractor) | `VGG19Extractor`: `models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)`, lay `features + avgpool`, freeze all params | YES | Cell `bcea153e` |
| A2 | VGG19 feature dim | 25088 (7x7x512 sau avgpool) | `x.flatten(1)` cho ra `(B, 25088)` | YES | Output: `train=(1597, 25088)` |
| A3 | Backbone 2 | InceptionV3 (pretrained, feature extractor) | `InceptionV3Extractor`: `models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)`, `base.fc = nn.Identity()` | YES | Cell `bcea153e` |
| A4 | InceptionV3 feature dim | 2048 (sau global pooling) | Output `(B, 2048)` | YES | Output: `train=(1597, 2048)` |
| A5 | ML Classifiers | SVM, kNN, RF, AdaBoost, DT | SVM, kNN, RF, AdaBoost, DT — du 5 classifier | YES | Cell `dfc2d5bf` va `run_scenario()` |
| A6 | SVM configuration | RBF kernel (pho bien) | `SVC(C=1.0, kernel='rbf', probability=True)` | YES (Assumed) | Khong co trong paper, gia tri chuan |
| A7 | kNN configuration | Khong neu | `KNeighborsClassifier(n_neighbors=5, metric='euclidean')` | YES (Assumed) | Gia tri chuan k=5 |
| A8 | RF configuration | Khong neu | `RandomForestClassifier(n_estimators=100)` | YES (Assumed) | Gia tri chuan |
| A9 | AdaBoost configuration | Khong neu | `AdaBoostClassifier(n_estimators=100)` | YES (Assumed) | Gia tri chuan |
| A10 | DT configuration | Khong neu | `DecisionTreeClassifier(random_state=42)` | YES (Assumed) | Gia tri chuan |
| A11 | PCA dimensionality reduction | Khong neu trong paper | DA XOA — `preprocess_features()` chi dung `StandardScaler` | YES (FIXED) | Patch 6 da loai bo PCA. Code nguon hien tai: `scaler = StandardScaler()` + `fit_transform`. Output cell van hien "PCA 256d" la stale output cu chua re-run |
| A12 | 10-fold stratified cross-validation | Neu ro trong paper (Section "Results") | Dung `learning_curve(cv=5)` de visualize LC, evaluate tren hold-out test set | KHAC | Paper dung 10-fold CV; notebook dung hold-out test set (hop li hon cho use case cua user) |
| A13 | Freeze backbone weights | Implied (feature extraction, khong fine-tune) | `for p in self.parameters(): p.requires_grad = False` | YES | Freeze hoan toan — dung |
| A14 | Tong so params VGG19 | ~138M (VGG19 day du) | 20,024,384 (chi features+avgpool) | KHAC | Notebook chi tao phan feature extractor, loai FC layers — hop li voi muc dich extraction nhung khac so trong paper |

**Architecture Fidelity Score: 12/14 = 85.7%**
*(A12 la difference co so — hold-out hop li hon cho user context; A14 khac vi chi dem features portion cua VGG19)*

**So sanh voi lan truoc:** Tang tu 79% len 85.7% sau khi A11 (PCA) duoc loai bo khoi code nguon.

---

## 3. HYPERPARAMETER & TRAINING MATCH

*Luu y: Paper nay la hybrid approach — KHONG co deep learning training loop. "Hyperparameters" chu yeu la cau hinh ML classifiers va pipeline.*

| # | Thong so | Gia tri trong Paper | Gia tri trong Notebook | Nguon | Khop? |
|---|---------|---------------------|----------------------|-------|-------|
| H1 | Backbone pretrained | ImageNet pretrained ro rang | `weights=...Weights.IMAGENET1K_V1` | Paper | YES |
| H2 | Feature extraction method | Frozen backbone (khong fine-tune) | Freeze toan bo params, eval mode | Paper | YES |
| H3 | ML classifiers su dung | SVM, kNN, RF, AdaBoost, DT | 5 classifiers dung het | Paper | YES |
| H4 | Evaluation method | Stratified 10-fold cross-validation | Hold-out test set (user's dataset) | Assumed (User context) | KHAC |
| H5 | StandardScaler preprocessing | Khong neu trong paper | `StandardScaler()` + `fit_transform` (PCA da xoa) | Assumed | N/A (hop li, khong lam mat thong tin) |
| H6 | Batch size (feature extraction) | Khong neu | 32 | Assumed | N/A |
| H7 | Metrics bao cao | Accuracy, Precision, Recall, F1, AUC, MCC | Accuracy, F1-Macro, Recall-Macro, classification_report per class | Paper (mot phan) | MOT PHAN (thieu AUC, MCC, Precision rieng) |
| H8 | Augmentation trong paper | Khong co (chi raw images) | Scenario 2 co 4 ky thuat aug | Assumed (Core Rule 3) | THEM (Core Rule bat buoc) |
| H9 | SVM hyperparameters | Khong neu | C=1.0, kernel='rbf' | Assumed | N/A |
| H10 | kNN hyperparameters | Khong neu k | k=5, euclidean | Assumed | N/A |

**Hyperparameter Match Score: 3/4 Paper items + gia dinh hop li = 70%**
*(H1, H2, H3 tu Paper — khop; H4 la khac biet co chu y; H7 thieu AUC va MCC; cac H con lai la Assumed — khong tinh vao mau so chinh)*

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing tu Paper (anh huong kien truc — BAT BUOC khop):

| # | Muc | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----|---------------------|----------------------|-------|---------|
| D1 | Image input size VGG19 (H x W) | 224x224 (neu ro: "input tensor size for VGG19 is 224 by 224") | `IMG_VGG19 = (224, 224)`, `transforms.Resize(IMG_VGG19)` | YES | Cell `04d78247` |
| D2 | Image input size InceptionV3 (H x W) | 299x299 (standard InceptionV3) | `IMG_INCEPTION = (299, 299)` | YES | Cell `04d78247` |
| D3 | Normalization | ImageNet (ngam dinh voi pretrained) | `MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]` | YES | ImageNet stats chuan |
| D4 | NUM_CLASSES | N/A (auto-detect) | `CLASS_NAMES = sorted(os.listdir(train_dir))` + `NUM_CLASSES = len(CLASS_NAMES)` | YES | Cell `5bba166e` |

**Preprocessing Consistency Score: 4/4 = 100%**

### 4b. Dataset Split trong Notebook:

| # | Muc | Gia tri trong Notebook | Hop li? | Ghi chu |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset structure | Pre-split: `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` | YES | Dung thu muc san co, khong tu split |
| DS2 | Split ratio | 1597/491/373 = 66/20/14 (theo user's dataset) | YES | Dataset user da san phan chia |
| DS3 | Test transform | `T_NOAUG_VGG`, `T_NOAUG_INC` (Resize + ToTensor + Normalize) | YES | Khong aug tren test |

### 4c. Augmentation (Scenario 2):

| # | Ky thuat | Paper co de cap? | Notebook implement? | Tham so khop? |
|---|---------|-----------------|--------------------|----|
| Aug1 | Rotation | KHONG | YES — `RandomRotation(degrees=20)` | N/A (Core Rule) |
| Aug2 | Zoom / RandomResizedCrop | KHONG | YES — `RandomResizedCrop(scale=(0.8, 1.0))` | N/A (Core Rule) |
| Aug3 | Horizontal Flip | KHONG | YES — `RandomHorizontalFlip(p=0.5)` | N/A (Core Rule) |
| Aug4 | ColorJitter / Brightness | KHONG | YES — `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` | N/A (Core Rule) |

**Ghi chu:** Paper goc KHONG su dung augmentation. Notebook them Scenario 2 theo Core Rule 3 — day la bo sung hop le, KHONG phai loi.

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage (tu paper):

| # | Feature trong Paper | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|---------|
| SF1 | Stratified 10-fold cross-validation | KHONG (chi LC voi cv=5) | Paper dung 10-fold cho toan bo evaluation — notebook dung hold-out, hop li hon cho user |
| SF2 | AUC metric | KHONG | Paper bao cao AUC cho moi combination — notebook bo qua |
| SF3 | MCC metric | KHONG | Paper bao cao MCC — notebook bo qua |
| SF4 | Precision metric (rieng) | Co (trong classification_report) | Patch 4 da them `classification_report` — co Precision per class, nhung khong co Precision Macro rieng biet trong summary table |
| SF5 | Pythagorean Tree visualization | KHONG | Khong can thiet cho muc tieu Kaggle |
| SF6 | 4 datasets evaluation | KHONG | Paper dung 4 datasets; notebook dung 1 dataset cua user (hop le) |

### 5b. Core Constraints Compliance:

| # | Rang buoc bat buoc | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 ky thuat) | YES | Cell `6d51ebc1` (S1) va `96d35881` (S2), du 4 ky thuat aug |
| CC2 | Framework dung voi paper (PyTorch / TF/Keras) | YES | Paper khong neu framework; notebook dung PyTorch — hop li |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | YES | `accuracy_score`, `f1_score(average='macro')`, `recall_score(average='macro')` |
| CC4 | `classification_report` per class (sklearn) | YES (FIXED) | Patch 4 da them. Cell `dfc2d5bf`: `print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))` trong `run_scenario()` |
| CC5 | `torchinfo.summary()` hoac `model.summary()` | YES (FIXED) | Patch 3 da them. Cell `bcea153e`: `summary(_vgg_tmp, input_size=(1, 3, 224, 224), device='cpu')` va `summary(_inc_tmp, input_size=(1, 3, 299, 299), device='cpu')`. *Luu y: cell output hien tai la stale — can re-run de hien summary table.* |
| CC6 | Learning Curve luu file anh dpi=300, cell rieng + plt.show() | YES (FIXED) | Patch 5 da doi `plt.close()` → `plt.show()`. Cell `dfc2d5bf`: `plt.savefig(path, dpi=300)` + `plt.show()` |
| CC7 | Confusion Matrix luu file anh dpi=300, cell rieng + plt.show() | YES (FIXED) | Tuong tu CC6 — cung trong `plot_confusion_matrix()` |
| CC8 | `NUM_CLASSES` auto-detect (khong hard-code) | YES | `NUM_CLASSES = len(CLASS_NAMES)`, `CLASS_NAMES = sorted(os.listdir(train_dir))` |
| CC9 | Excel Export 2-sheet (Baseline + Augmented), dung cot | YES | `wb.create_sheet('Baseline')`, `wb.create_sheet('Augmented')`, moi sheet 10 rows |
| CC10 | Anh LC + CM nhung truc tiep vao Excel | YES | `XLImage(r['lc_path'])`, `XLImage(r['cm_path'])` duoc add vao cot F, G |
| CC11 | Khong co code thuc thi training (chi sinh notebook) | N/A | Day la notebook da co output — khong applicable |
| CC12 | ReproduceSummary.md duoc tao kem notebook | KHONG | Van chua co file `VGG19_ReproduceSummary.md` trong `outputs/` |
| CC13 | `model_name` la ten kien truc (khong phai ten bai bao) | YES | `f'{backbone_name} + {clf_name}'` — vi du "VGG19 + SVM" la ten kien truc dung |
| CC14 | `device = torch.device(...)` auto-detect + `.to(device)` | YES | `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, extractor `.to(DEVICE)` |
| CC15 | `torch.manual_seed(42)` dat o dau notebook | YES (FIXED) | Patch 1 da them. Cell `388cdbc3`: `SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)` |
| CC16 | Khong dung `pretrained=True` — phai dung `weights=...Weights.DEFAULT` | YES | `weights=models.VGG19_Weights.IMAGENET1K_V1`, `weights=models.Inception_V3_Weights.IMAGENET1K_V1` |
| CC17 | Baseline training transform: CHI Resize + ToTensor + Normalize | YES | `T_NOAUG_VGG`: `Resize -> ToTensor -> Normalize` — khong co gi them |
| CC18 | Validation/Test transform KHONG co augmentation | YES | Test dung `T_NOAUG_VGG` / `T_NOAUG_INC`; Scenario 2 test reuse No-Aug features tu S1 |
| CC19 | `torch.backends.cudnn.deterministic = True` + `benchmark = False` | YES (FIXED) | Patch 2 da them. Cell `388cdbc3`: `torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False` |
| CC20 | `train_with_checkpoint()` — best model reload truoc test eval | KHONG APPLICABLE | Hybrid approach: ML classifiers khong co checkpoint mechanism; sklearn `.fit()` roi `.predict()` la dung |
| CC21 | Cell "Dataset Distribution": bar chart class count + imbalance check | MOT PHAN | Cell `5bba166e` co bar chart + in so luong cac class, nhung THIEU kiem tra tu dong `max/min >= 1.5` va tinh `class_weights` |
| CC22 | `class_names` dung `get_class_names()` hoac `.dataset.classes` | MOT PHAN | Dung `sorted(os.listdir(train_dir))` thay vi `ImageFolder(TRAIN_DIR).classes` — ket qua tuong duong nhung khong theo quy tac chuan |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` | MOT PHAN | `num_workers=NUM_WORKERS` (=2), `pin_memory=True`. Rule 13 khuyen nghi 4 workers cho Kaggle T4x2 — notebook dung 2 (chap nhan duoc nhung khong toi uu) |
| CC24 | `torch.cuda.empty_cache()` sau Scenario 1 truoc Scenario 2 | YES | `del vgg_ext; torch.cuda.empty_cache()` va `del inc_ext; torch.cuda.empty_cache()` sau moi extractor trong ca S1 va S2 |

**Compliance Score (updated):**
- CC11, CC20 khong applicable — loai ra (mau so = 22)
- YES day du: CC1, CC2, CC3, CC4, CC5, CC6, CC7, CC8, CC9, CC10, CC13, CC14, CC15, CC16, CC17, CC18, CC19, CC24 = 18 items
- MOT PHAN (0.5 diem): CC21, CC22, CC23 = 1.5 diem
- KHONG: CC12 = 0 diem
- **Tong: (18 + 1.5) / 22 = 19.5/22 = 88.6%**

**So sanh voi lan truoc:** Tang tu 67% (8/20 applicable) len 88.6% (19.5/22 applicable) nho 4 patch nghiem trong da duoc ap dung.

---

## 6. ASSUMPTION RISK AUDIT

| # | Thong so gia dinh | Gia tri dung | Muc do rui ro | Ly do rui ro |
|---|------------------|-------------|--------------|--------------|
| 1 | StandardScaler preprocessing | StandardScaler (PCA da xoa) | THAP | Paper khong mo ta preprocessing truoc ML, nhung StandardScaler la buoc chuan, khong lam mat thong tin. Giam tu CAO xuong THAP sau khi xoa PCA. |
| 2 | SVM hyperparameters (C, kernel) | C=1.0, RBF kernel | TRUNG BINH | Khong co trong paper. C=1.0, RBF la pho bien nhung C khac co the thay doi ket qua dang ke |
| 3 | kNN: k=5 | k=5, euclidean | TRUNG BINH | Khong co trong paper. k co anh huong truc tiep den accuracy |
| 4 | RF: n_estimators=100 | 100 trees | THAP | 100 trees la chuan, it anh huong |
| 5 | Evaluation: hold-out test set thay vi 10-fold CV | User's train/val/test split | TRUNG BINH | Paper dung 10-fold stratified CV — ket qua hold-out se khac. Nhung cho user dataset co san, hold-out la phuong phap dung |
| 6 | ImageNet normalization (mean/std) | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] | THAP | Chuan cho ImageNet pretrained models — hop li |
| 7 | Augmentation Scenario 2 | 4 aug techniques (Core Rule) | THAP | Core Rule bat buoc, khong phai assumption tu paper |
| 8 | Batch size 32 | 32 | THAP | Chi dung cho feature extraction, khong anh huong ket qua cuoi |

**Rui ro trung binh can chu y:**
- **SVM C=1.0 (Assumption 2):** Hyperparameter C anh huong truc tiep den margin cua SVM. Paper khong neu — neu C khac, accuracy co the tang/giam dang ke. Voi VGG19 feature 25088d, C nen duoc tune.
- **kNN k=5 (Assumption 3):** k=5 la gia tri default nhung voi feature 2048d (InceptionV3), k lon hon co the tot hon.
- **Hold-out vs 10-fold CV (Assumption 5):** Day la khac biet phuong phap luon luan — ket qua khong the so sanh truc tiep voi paper.

**Rui ro cao truoc day da duoc giai quyet:**
- PCA(256d) — da xoa, khong con la rui ro cao

---

## 7. VAN DE CON LAI SAU PATCH

*Tat ca 4 van de "Nghiem trong" da duoc sua trong lan patch nay. Cac van de sau day la "Nen sua" hoac "Tuy chon".*

### Nen sua truoc khi submit:

- [ ] **Stale cell outputs (QUAN TRONG):** Notebook da duoc patch o muc code nguon nhung chua duoc re-run. Cac cell output van hien "PCA 256d | variance retained: 63.55%" va chua co output cua `torchinfo.summary()` hay `classification_report`. **Bat buoc chay lai toan bo notebook tu dau truoc khi upload Kaggle.** (tat ca cac cell tu cell imports tro xuong)
- [ ] **THIEU ReproduceSummary.md (CC12):** Tao file `VGG19_ReproduceSummary.md` trong `outputs/` kem theo notebook.
- [ ] **Imbalance check tu dong (CC21):** Trong `explore_dataset()`, them kiem tra `max_count / min_count >= 1.5` va thong bao co/khong dung class weighting. Hien tai chi co bar chart.

### Tuy chon (cai thien chat luong):

- [ ] **`num_workers=2` → 4 (CC23):** Rule 13 khuyen nghi 4 workers cho Kaggle T4x2. Nen doi `NUM_WORKERS = 4`.
- [ ] **`class_names` tu `ImageFolder` (CC22):** Nen doi `CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))` thanh `CLASS_NAMES = ImageFolder(TRAIN_DIR).classes` de thong nhat voi Core Rule 12.
- [ ] **Them Precision Macro vao summary table (H7):** Paper bao cao Precision rieng — co the them `precision_score(average='macro')` trong `run_scenario()` va vao summary table.
- [ ] **Them AUC va MCC (SF2, SF3):** Paper bao cao AUC va MCC cho moi combination — co the them `roc_auc_score(multiclass='ovr')` va `matthews_corrcoef()` de tang do trung thuc.

---

## 8. GHI CHU BO SUNG

### Thay doi diem so sau patch:

| Dimension | Diem lan truoc | Diem lan nay | Thay doi |
|-----------|---------------|-------------|---------|
| Paper Completeness | 50% | 50% | Khong doi |
| Architecture Fidelity | 79% | 86% | +7% (xoa PCA) |
| Hyperparameter Match | 70% | 70% | Khong doi |
| Preprocessing Consistency | 100% | 100% | Khong doi |
| Constraint Compliance | 67% | 89% | +22% (4 patch nghiem trong) |
| **TONG** | **76.8%** | **77.7%** | **+0.9%** |

*Ghi chu: Diem tong tang it (0.9%) du Compliance tang nhieu (+22%) vi trong so cua Compliance chi la 15% trong khi Architecture Fidelity (trong so 40%) chi tang 7%.*

### Nhan xet tong the:
Sau 6 patch, notebook da dat trang thai "Go with Caution" vung hon. Tat ca 4 van de "Nghiem trong" da duoc sua trong code nguon. Van de duy nhat can lam ngay truoc khi upload la re-run toan bo notebook de output phan anh dung trang thai code da patch.

### Luu y ve ban chat bai bao:
Paper nay la mot **comparative study** — khong co mot "model toi uu" duy nhat ma la bang so sanh 10 combinations (2 backbone x 5 ML). Notebook implement dung tinh than nay bang cach chay va bao cao tat ca 10 combinations.

### Stale output la van de nghiem trong can xu ly:
Mau thuan giua code nguon (khong co PCA) va cell output (hien "PCA 256d") co the gay nham lan cho nguoi xem notebook tren Kaggle. Re-run la buoc bat buoc truoc khi submit.

### Tong ket PCA:
Viec xoa PCA la thay doi dung dan — paper khong mo ta buoc giam chieu nao, va viec giu nguyen feature dim (VGG19: 25088d, InceptionV3: 2048d) sau StandardScaler la trung thuc voi paper hon. StandardScaler chi chuan hoa scale, khong lam mat thong tin, nen khong co rui ro cao nhu truoc.
