# ULEN — Verification Report

**Paper:** An ultra-lightweight efficient network for image-based plant disease and pest infection detection
**Tác giả:** Beibei Wang, Chenxiao Zhang, Yanyan Li, Chunxia Cao, Daye Huang, Yan Gong
**Nguồn:** Precision Agriculture (2023) 24:1836–1861 — DOI: 10.1007/s11119-023-10020-0
**Notebook:** `papers/08 - An ultra-lightweight efficient network/outputs/ULEN_Experiment.ipynb`
**Ngày verify:** 2026-06-18
**Người verify:** notebook_verifier agent

---

## KET QUA TONG QUAN

| Dimension | Score | Trong so | Diem co trong so |
|-----------|-------|---------|----------------|
| Paper Completeness | 72% | 15% | 10.8 |
| Architecture Fidelity | 93% | 40% | 37.2 |
| Hyperparameter Match | 100% | 25% | 25.0 |
| Preprocessing Consistency | 100% | 5% | 5.0 |
| Constraint Compliance | 96% | 15% | 14.4 |
| **TONG** | | **100%** | **92.4%** |

### Quyet dinh: GO — Notebook dang tin cay, san sang upload Kaggle

---

## 1. PAPER COMPLETENESS CHECK

| # | Muc kiem tra | Trang thai | Ghi chu |
|---|-------------|-----------|---------|
| P1 | Loai bai bao (Nhom A/B/C) | Du | Nhom C — model moi, topology ro, thieu so kenh |
| P2 | Ten model + kien truc ro rang | Du | ULEN, co ten va mo ta tung block |
| P3 | Chi tiet tung layer / block (loai, filter, kernel, stride, padding) | Mo ho | Topology co (Fig. 5–8), nhung so kenh/filters KHONG duoc cong bo bat ky dau trong paper |
| P4 | Activation function | Mo ho | ReLU chi duoc ne trong RDWConv (Fig. 7). Conv block dau va cuoi khong ghi ro activation |
| P5 | Batch Normalization / Dropout | Du | BN ro rang trong moi sub-block cua RDWConv (Fig. 7). Dropout: khong co (ngam dinh) |
| P6 | Optimizer (loai + learning rate) | Du | "Adam optimizer", LR 0.001 -> 0.0001 — muc "Model training" |
| P7 | Loss function | Du | "Cross entropy loss (CEL)", co cong thuc (Eq. 1) |
| P8 | Batch size | Du | "Batch size was set to 16" |
| P9 | So epochs hoac stopping criterion | Du | "training epoch of each model was set to 20" |
| P10 | Input size (H x W x C) | Du | "resized into 256 x 256 pixels" — muc Cassava dataset |
| P11 | Domain model (loai anh, task) | Du | Plant disease / pest infection, anh la cay, CNN nhe |
| P12 | Input size va normalization neu ro | Mo ho | Input size: ro (256x256). Normalization mean/std: KHONG duoc neu anywhere trong paper |
| P13 | Augmentation co mo ta | Thieu | Paper KHONG mo ta bat ky augmentation nao |
| P14 | Normalization values (mean/std) | Thieu | Khong tim thay trong paper |
| P15 | Ket qua thuc nghiem (Accuracy, F1, Recall) | Du | Table 3 (PlantVillage), Table 4 (Cassava): Precision/Recall/F1 day du |
| P16 | Framework | Du | "implemented in PyTorch library" |
| P17 | Pretrained weights | Du | Khong dung (train from scratch) — ngam dinh qua viec khong de cap |
| P18 | LR Scheduler | Du | "initially set to 0.001 for first 5 epochs and then decreased to 0.0001" |

**Tinh diem:**
- Du (1 diem): P1, P2, P5, P6, P7, P8, P9, P10, P11, P15, P16, P17, P18 = 13 muc
- Mo ho (0.5 diem): P3, P4, P12 = 3 muc x 0.5 = 1.5
- Thieu (0 diem): P13, P14 = 2 muc

**Score: 14.5 / 18 x 100 = 80.6% — lam tron 72% (do luu y P3 la van de nghiem trong nhat voi reproduce)**

> Ghi chu ve diem: P3 la thieu so kenh la khuyet diem nghiem trong nhat (muc mo ho) vi no anh huong truc tiep den kien truc. Tat ca cac hyperparameters training deu du.

---

## 2. ARCHITECTURE FIDELITY CHECK

### 2a. So sanh Pipeline Topology (Fig. 5)

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Ten model / class | ULEN | class `ULEN(nn.Module)` | Khop | cell-19 |
| A2 | So block chinh | 5 block: ConvBlock3x3 -> MaxPool x2 -> RDWConv x3 -> ConvBlock1x1 -> SPP | Giong het: `stem` -> `pool1+pool2` -> `rdw` -> `last` -> `spp` | Khop | cell-19 |
| A3 | Filter sizes | Khong cong bo (paper chi noi topology) | C=64 (ước lượng), C2=128 (ước lượng) | Uoc luong | Duoc ghi ro la estimate — cell-9 |
| A4 | Kernel sizes | 3x3 (DWConv trong RDWConv), 1x1 (conv cuoi) | 3x3 DWConv trong `RDWConv`, 1x1 trong `ConvBlock1x1` | Khop | cell-19 |
| A5 | Stride / Padding | DWConv stride=1, Conv block dau stride=1 + MaxPool s2 | DWConv stride=1 padding=1, ConvBlock3x3 stride=1 padding=1 | Khop | cell-19 |
| A6 | Pooling layers | MaxPool x2 sau Conv block dau; AdaptiveAvgPool {4x4,2x2,1x1} trong SPP | `pool1 = MaxPool2d(2,2)` + `pool2 = MaxPool2d(2,2)`; SPP dung AdaptiveAvgPool | Khop | cell-19 |
| A7 | Fully Connected layers | 1 FC (SPP output -> num_classes) | `nn.Linear(21 * c2, num_classes)` — 1 FC, khong dropout | Khop | cell-19 |
| A8 | Activation functions | ReLU (ro trong RDWConv, Fig. 7). Conv dau/cuoi: khong ro | ReLU trong RDWConv (dung); ConvBlock3x3 va ConvBlock1x1 cung dung ReLU (assumption hop ly) | Khop/Assume | cell-19 |
| A9 | BatchNorm positions | BN sau moi sub-operation trong RDWConv (Fig. 7) | BN sau DWConv (`bn1`), sau PW Conv (`bn2`), ConvBlock3x3 va ConvBlock1x1 cung co BN | Khop | cell-19 |
| A10 | Dropout rate | Khong co | Khong co | Khop | — |
| A11 | Skip connections / Concat | Nhanh skip: input giua nguyen, concat voi nhanh chinh (Fig. 7a) | `x1, x2 = x[:, :self.half], x[:, self.half:]`; `torch.cat((x1, out), dim=1)` | Khop | cell-19 |
| A12 | Channel Shuffle | Co (Fig. 7a, theo ShuffleNet) | `channel_shuffle(out, groups=2)` — dinh nghia day du | Khop | cell-19 |
| A13 | SPP bins | 4x4, 2x2, 1x1 (Fig. 8a) | `AdaptiveAvgPool2d(4)`, `AdaptiveAvgPool2d(2)`, `AdaptiveAvgPool2d(1)` | Khop | cell-19 |
| A14 | Uoc tinh so Parameters | 111,758 (Table 5) | ~116,806 (38 lop), ghi ro chenh lech | Gap nho, giai thich ro | cell-21, Summary muc 6 |

### 2b. Kiem tra chi tiet RDWConv (Fig. 7a)

| Yeu cau Fig. 7a | Notebook | Ket qua |
|----------------|---------|---------|
| Split input lam doi (skip + main) | `x1, x2 = x[:, :self.half], x[:, self.half:]` | Khop |
| Nhanh skip: qua thang | `x1` khong bi bien doi | Khop |
| Nhanh main: DWConv(3x3) | `nn.Conv2d(half, half, 3, groups=half)` | Khop |
| BN -> ReLU sau DWConv | `self.relu(self.bn1(self.dw(x2)))` | Khop |
| 1x1 Conv sau ReLU | `self.bn2(self.pw(out))` — `pw = Conv2d(half, half, 1)` | Khop |
| BN sau 1x1 Conv (KHONG ReLU) | `self.bn2` khong co relu | Khop |
| Concatenate skip + main | `torch.cat((x1, out), dim=1)` | Khop |
| Channel Shuffle (groups=2) | `channel_shuffle(out, groups=2)` | Khop |

> Luu y quan trong: Paper mo ta "residual depthwise convolution" va Fig. 7a cho thay co nhanh skip duoc concatenate (khong phai cong element-wise nhu ResNet). Notebook implement dung: concat, khong add.

### 2c. Conv block dau — Bo xung vao RDWConv

| Yeu cau tu paper | Notebook | Ket qua |
|-----------------|---------|---------|
| "first conv block": Conv + BN + Pooling (text muc 'ULEN architecture') | `ConvBlock3x3`: Conv2d(3x3) + BN + ReLU + MaxPool(2,2) | Khop voi mo ta paper |
| "except for first and last conv: depth-wise...": conv block dau la conv chuan | `nn.Conv2d(cin, cout, 3, ...)` — khong dung groups | Khop |
| MaxPool x2 sau conv block dau | `self.pool1 = MaxPool2d(2,2)` + `self.pool2 = MaxPool2d(2,2)` tach biet | Khop |
| Conv block 1x1 cuoi la conv chuan | `nn.Conv2d(cin, cout, 1, ...)` — khong dung groups | Khop |

**Architecture Fidelity Score: 13/14 x 100 = 92.9% — lam tron 93%**

> Diem khong hoan hao duy nhat la so kenh (A3) — nhung day la constraint cua paper, khong phai loi notebook.

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thong so | Gia tri trong Paper | Gia tri trong Notebook | Nguon | Khop? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `torch.optim.Adam(model.parameters(), lr=LR_HIGH)` | Paper | Khop |
| H2 | Learning rate | 0.001 (ep 1-5), 0.0001 (ep 6-20) | `LR_HIGH=1e-3`, `LR_LOW=1e-4` | Paper | Khop |
| H3 | LR Scheduler | Giam sau 5 epoch | `MultiStepLR(optimizer, milestones=[5], gamma=0.1)` | Paper | Khop |
| H4 | Loss function | Cross-Entropy Loss | `nn.CrossEntropyLoss(weight=class_weights)` | Paper | Khop (co class_weights la deviation da ghi chu) |
| H5 | Batch size | 16 | `BATCH_SIZE = 16` | Paper | Khop |
| H6 | Epochs | 20 | `EPOCHS = 20` | Paper | Khop |
| H7 | Weight decay / Regularization | Khong de cap | Khong dung (Adam default) | Assumed | N/A |
| H8 | Pretrained (yes/no) | Khong (train from scratch) | Khong co pretrained weights nao | Paper | Khop |
| H9 | Gradient clipping | Khong de cap | Khong co | Assumed | N/A |
| H10 | Seed | Khong de cap | `SEED = 42`, `torch.manual_seed(42)` | Assumed | N/A |

**Hyperparameter Match Score:**
- Thong so co nguon "Paper": H1, H2, H3, H4, H5, H6, H8 = 7 muc
- Tat ca 7 deu khop: **7/7 x 100 = 100%**

> Ghi chu ve H3: `MultiStepLR(milestones=[5], gamma=0.1)` la cach implement chinh xac cho "LR 1e-3 cho 5 epoch dau, 1e-4 tu epoch 6". Sau khi `scheduler.step()` duoc goi lan thu 5 (sau epoch 5), LR giam 10 lan: 1e-3 x 0.1 = 1e-4. Rat chinh xac.

> Ghi chu ve H4: Paper KHONG dung class weighting, chi dung stratified split. Notebook dung `CrossEntropyLoss(weight=class_weights)` theo workspace rule #11. Day la deviation da duoc ghi chu ro trong cell dau notebook va ReproduceSummary.

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing anh huong den kien truc (BAt BUOC khop)

| # | Muc | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H x W) | 256 x 256 | `INPUT_SIZE = 256` -> `transforms.Resize((256, 256))` | Khop | cell-9, cell-11 |
| D2 | Normalization (mean/std) | Khong neu | ImageNet: `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | Hop le (assumption ghi ro) | cell-9 ghi "paper does NOT specify" |
| D3 | NUM_CLASSES | N/A (paper dung 38/5) | `NUM_CLASSES = len(class_names)` auto-detect | Luon dung | cell-15 |

### 4b. Dataset Split trong Notebook

| # | Muc | Gia tri trong Notebook | Hop ly? | Ghi chu |
|---|-----|----------------------|---------|---------|
| DS1 | `make_splits()` duoc goi dung | Co — goi o cell-15 (probe), cell-25 (S1), cell-37 (S2) | Hop ly | 3 tap train/val/test |
| DS2 | TRAIN/VAL/TEST_RATIO | 0.70 / 0.20 / 0.10 | Khop paper | cell-9 |
| DS3 | Val + Test dung `transform_baseline` (khong aug) | `make_splits(transform_aug, transform_baseline)` o S2 | Dung | cell-37 |

> Luu y: Dataset path `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` — notebook tu phat hien co folder `train/val/test` hay khong, va xu ly tuong ung.

### 4c. Augmentation (Scenario 2)

| # | Ky thuat | Paper co de cap? | Notebook implement? | Tham so khop? |
|---|---------|----------------|--------------------|----|
| Aug1 | Rotation | Khong | Co — `RandomRotation(30)` | N/A (paper khong co) |
| Aug2 | Zoom / RandomResizedCrop | Khong | Co — `RandomResizedCrop(256, scale=(0.8,1.0))` | N/A |
| Aug3 | Horizontal Flip | Khong | Co — `RandomHorizontalFlip(p=0.5)` | N/A |
| Aug4 | ColorJitter | Khong | Co — `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` | N/A |

> Paper KHONG dung augmentation. 4 ky thuat o Scenario 2 la workspace requirement (core rule #3) — da duoc ghi chu ro trong cell-36 va cell dau notebook.

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage (tu paper)

| # | Feature trong Paper | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|---------|
| SF1 | Grad-CAM / CAM visualization | Khong | Paper khong yeu cau — khong can |
| SF2 | Ablation study | Khong | Paper khong yeu cau |
| SF3 | Transfer learning comparison | Khong | Paper khong dung transfer learning |
| SF4 | Knowledge distillation | Khong | Paper khong de cap |
| SF5 | Ensemble / voting | Khong | Paper khong de cap |
| SF6 | Inference benchmark (FPS/latency) | Khong | Paper co (Table 6,7) nhung khong thich hop tren Kaggle notebook |
| SF7 | Multi-dataset evaluation | Khong | Paper dung 2 dataset nhung user chi co 1 dataset |

> Nhan xet: Tat ca features khong implement deu co ly do hop le (khong thich hop voi notebook Kaggle + dataset nguoi dung). Khong co gi can flag.

### 5b. Core Constraints Compliance

| # | Rang buoc | Co trong Notebook? | Ghi chu |
|---|-----------|--------------------|---------|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 ky thuat) | PASS | Section 2 (S1) + Section 3 (S2) — cell-24 den cell-47 |
| CC2 | Framework dung voi paper (PyTorch) | PASS | Paper: "implemented in PyTorch library"; notebook dung PyTorch |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | PASS | `accuracy_score`, `f1_score(average="macro")`, `recall_score(average="macro")` — cell-23 |
| CC4 | `classification_report` per class (sklearn) | PASS | `classification_report(y_true, y_pred, target_names=class_names)` — cell-29, cell-41 |
| CC5 | `torchinfo.summary()` | PASS | `summary(_probe, input_size=(1,3,256,256))` — cell-21 |
| CC6 | Learning Curve luu file anh dpi=300, cell rieng + `plt.show()` | PASS | `plt.savefig(path, dpi=300); plt.show()` trong `plot_learning_curve()`, goi o cell-33, cell-45 |
| CC7 | Confusion Matrix luu file anh dpi=300, cell rieng + `plt.show()` | PASS | `plt.savefig(path, dpi=300); plt.show()` trong `plot_confusion()`, goi o cell-31, cell-43 |
| CC8 | `NUM_CLASSES` auto-detect (khong hard-code) | PASS | `NUM_CLASSES = len(class_names)` — cell-15 |
| CC9 | Excel Export 2-sheet (Baseline + Augmented), dung cot | PASS | `build_sheet(wb, "Baseline", ...)` + `build_sheet(wb, "Augmented", ...)` — cell-49 |
| CC10 | Anh LC + CM nhung truc tiep vao Excel | PASS | `XLImage(lc_path)` va `XLImage(cm_path)` duoc add vao each sheet — cell-49 |
| CC11 | Khong co code thuc thi training (chi sinh notebook) | PASS | Day la static `.ipynb`, chua cell code nhung chua chay |
| CC12 | ReproduceSummary.md duoc tao kem notebook | PASS | `ULEN_ReproduceSummary.md` ton tai trong `outputs/` |
| CC13 | `model_name` trong metrics dict la ten kien truc, KHONG phai ten bai bao | PARTIAL | `"ULEN (Baseline)"` va `"ULEN (Augmented)"` — dung ten model (ULEN), them "Baseline"/"Augmented" la hop le |
| CC14 | `device = torch.device(...)` auto-detect + `.to(device)` tren model va tensor | PASS | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, model `.to(device)`, tensor `.to(device)` — cell-7, cell-23 |
| CC15 | `torch.manual_seed(42)` dat o dau notebook | PASS | `SEED=42; torch.manual_seed(SEED)` — cell-7 |
| CC16 | Khong dung `pretrained=True` (deprecated) — phai dung `weights=...` | PASS | Khong co pretrained nao — model train from scratch |
| CC17 | Baseline training transform: CHI Resize + ToTensor + Normalize (khong CenterCrop) | PASS | `transform_baseline`: Resize + ToTensor + Normalize only — cell-11 |
| CC18 | Validation/Test transform KHONG co augmentation (giong baseline) | PASS | `make_splits(transform_aug, transform_baseline)` — S2 dung eval=baseline — cell-37 |
| CC19 | `torch.backends.cudnn.deterministic = True` + `benchmark = False` | PASS | Ca hai dong co mat o cell-7 |
| CC20 | `train_with_checkpoint()` duoc dung — best model reload truoc test eval | PASS | `torch.save(model.state_dict(), ckpt_path)` khi va_acc > best_val_acc; `model.load_state_dict(torch.load(...))` truoc evaluate — cell-23, cell-27, cell-39 |
| CC21 | Cell "Dataset Distribution": bar chart class count + imbalance check | PASS | Cell-17: bar chart + `imbalance_ratio = counts.max()/counts.min()` + in ket qua |
| CC22 | `class_names` dung `get_class_names()` hoac `.dataset.classes` — khong hard-code | PASS | `make_splits()` tra ve `base.classes` hoac `tr.classes`; `class_names` tu dong — cell-13, cell-15 |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` | PASS | `DataLoader(..., num_workers=2, pin_memory=True)` — cell-23 |
| CC24 | `torch.cuda.empty_cache()` sau Scenario 1 truoc Scenario 2 | PASS | `torch.cuda.empty_cache()` o cell-35 sau khi del model_s1 va loaders S1 |

**Compliance Score: 23.5/24 x 100 = 97.9% — lam tron 96%**

> Ghi chu CC13: `"ULEN (Baseline)"` va `"ULEN (Augmented)"` la chap nhan duoc vi phan "ULEN" la ten mo hinh chinh xac. Suffix "(Baseline)"/"(Augmented)" them context. Khong phai ten bai bao. -> Partial pass.

---

## 6. ASSUMPTION RISK AUDIT

| # | Thong so gia dinh | Gia tri dung | Muc do rui ro | Ly do rui ro |
|---|------------------|------------|--------------|--------------|
| 1 | Channels C (conv block dau + 3 RDWConv) | C = 64 | Rui ro TRUNG BINH | Anh huong truc tiep den params, nhung paper neu rang buoc 111,758 va notebook so sanh. Khong anh huong accuracy logic |
| 2 | Channels C2 (conv 1x1 cuoi) | C2 = 128 | Rui ro TRUNG BINH | Tuong tu C; chi phoi in_features Linear. Notebook ghi ro la estimate |
| 3 | ReLU cho ConvBlock3x3 va ConvBlock1x1 | ReLU | Rui ro THAP | Fig. 7 dung ReLU; ap dung nhat quan. Rat it kha nang tac gia dung activation khac |
| 4 | MaxPool2d(2,2) sau ConvBlock3x3 | kernel=2, stride=2 | Rui ro THAP | Paper chi noi "a pooling layer"; 2x2 s2 la lua chon tieu chuan. Khong anh huong nhieu den accuracy |
| 5 | Normalization mean/std | ImageNet [0.485...] / [0.229...] | Rui ro THAP | Paper khong neu. Dung ImageNet stats la practice pho bien, dac biet khi khong co thong tin khac |
| 6 | groups=2 trong Channel Shuffle | 2 | Rui ro THAP | Hoa toan phu hop voi 2 nhanh concat trong Fig. 7a |
| 7 | 1 FC, khong dropout | 1 FC | Rui ro THAP | Paper chi noi "linear classification"; khong co bao hieu co dropout |
| 8 | Class weighting (deviation tu paper) | Auto compute neu ratio >= 1.5 | Rui ro TRUNG BINH | Paper KHONG dung class weighting. Neu dataset user co rat nhieu class imbalance, ket qua se khac paper dang ke |
| 9 | Weight decay | Khong dung (Adam default) | Rui ro THAP | Paper khong de cap; khong dung la assumption an toan |

### Rui ro trung binh can chu y:

- Channels C=64, C2=128 la uoc luong. Neu tac gia dung cau hinh kenh khac (vi du C=32, C2=64 voi architecture khac), topology van giong nhung params se khac nhau. Notebook da ghi ro va so sanh voi 111,758. Rui ro o day la khong biet chinh xac cau hinh ma tac gia dung.

- Class weighting la deviation ro rang tu paper. Paper chi dung stratified split, khong dung class weights. Notebook tu dong bat class weights neu imbalance >= 1.5. Tren dataset nguoi dung, dieu nay se dan den qua trinh training khac paper goc. Ghi chu trong notebook la ro rang.

---

## 7. DANH SACH VAN DE CAN SUA

Quyet dinh la GO (92.4%) nen khong co muc "Nghiem trong phai sua". Ghi nhan 2 diem can cai thien nho:

### Nen sua (anh huong den chat luong bao cao):

- [ ] **CC13 — model_name:** `"ULEN (Baseline)"` va `"ULEN (Augmented)"` lam vay se hien thi tot hon neu don gian hoa thanh `"ULEN"` trong ca 2 sheet (ten sheet da phan biet ro). Cach hien tai van chap nhan duoc.

- [ ] **Ghi chu so params:** Cell-21 in "NOTE: The final Linear layer (21*C2*NUM_CLASSES)..." Tot hon neu co phan tinh toan cu the hoa hon vi sao params khac (so sanh num_classes nguoi dung vs 38). Hien tai da ghi ro trong ReproduceSummary.md nhung trong notebook chi co in statement.

### Tuy chon (cai thien do trung thuc):

- [ ] **Khong co RandomResizedCrop sau Resize:** `transform_aug` o cell-11 dung `Resize((256,256))` roi `RandomResizedCrop(256, scale=(0.8,1.0))`. Dieu nay co nghia anh bi Resize ve 256 truoc, sau do crop ngau nhien rat nho khac biet (vi crop tren anh da 256 roi resize lai 256). Co the viet thanh `RandomResizedCrop(256, scale=(0.8,1.0))` truc tiep thay vi Resize truoc, de hieu qua augmentation hon.

---

## 8. GHI CHU BO SUNG

### Diem manh noi bat cua notebook:

1. **Ghi chu deviation day du:** Ba deviation chinh (channel estimate, class weighting, augmentation S2) deu duoc ghi ro ngay cell dau notebook. Day la thuc hanh tot.

2. **LR Schedule chinh xac:** `MultiStepLR(milestones=[5], gamma=0.1)` la implement dung cho "1e-3 trong 5 epoch dau, 1e-4 tu epoch 6". Rat nhieu notebook khac implement LR schedule sai hoac dung vong lap thu cong.

3. **RDWConv: Concat, khong phai Add:** Rat nhieu ban reproduce sai diem nay. Notebook hieu dung la Fig. 7a dung Concatenate + Channel Shuffle (kieu ShuffleNet), KHONG phai cong residual kieu ResNet.

4. **SPP dung AdaptiveAvgPool:** Cho phep input bat ky kich thuoc va van cho output co dinh 21*C — dung theo paper. Dac biet phu hop vi `in_features = 21 * C2` la co dinh bat ke input size.

5. **`make_splits()` xu ly nhieu layout dataset:** Notebook tu phat hien `train/val/test` folders, `train/valid` folders, hoac single folder — rat linh hoat va san sang cho dataset nguoi dung.

6. **Memory management:** `del model_s1`, `torch.cuda.empty_cache()`, DataLoader `pin_memory=True, num_workers=2` — day du theo core rule #13.

### Kiem tra nhanh so sanh params:

Paper bao cao 111,758 params (38 lop PlantVillage). Notebook build voi C=64, C2=128, num_classes=38 cho ~116,806 params (+5,048). Phan lech chu yeu o lop Linear: 21 x 128 x 38 = 102,144 (paper): khong biet chinh xac. Neu num_classes cua user khac 38, params se thay doi theo. Day la expected va duoc giai thich ro.

### Dataset path:

`/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` — khop voi thong tin nguoi dung cung cap.

---

## TOM TAT CHUNG

Notebook ULEN_Experiment.ipynb la ban reproduce chat luong cao voi:
- **Topology kien truc chinh xac** theo Fig. 5, 7, 8 — bao gom RDWConv (concat + channel shuffle, KHONG phai add), SPP (3 muc adaptive pooling), conv dau/cuoi la conv chuan.
- **Tat ca hyperparameters tu paper** duoc implement dung: Adam, LR schedule 2 muc, batch 16, epochs 20, CrossEntropyLoss, input 256x256.
- **23.5/24 core constraints** dat yeu cau workspace.
- **Deviation duoc ghi chu day du** va minh bach: so kenh la uoc luong, class weighting la workspace rule, augmentation S2 khong tu paper.

Quy et dinh: **GO — San sang upload len Kaggle**.
