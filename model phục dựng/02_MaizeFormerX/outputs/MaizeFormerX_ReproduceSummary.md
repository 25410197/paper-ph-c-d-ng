# MaizeFormerX — Reproduce Summary

**Paper:** MaizeFormerX: a lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis  
**Authors:** Md Mostafizur Rahman, Md Najmul Gony, Md Shafiq Ullah, Sd Maria Khatun Shuvra, Rezaul Haque, Md. Redwan Ahmed, S. M. Masfequier, Rahman Swapno, M. Murugappan, Muhammad E. H. Chowdhury, Gomesh Nair, V. Saravanan  
**Source:** Scientific Reports (2026) 16:15160 — DOI: 10.1038/s41598-026-44550-0  
**Ngay phuc dung:** 2026-05-27  
**Notebook xuat:** `papers/MaizeFormerX a lightweight/outputs/MaizeFormerX_Experiment.ipynb`

---

## 1. THONG TIN BAI BAO

| Muc | Chi tiet |
|-----|---------|
| Tieu de | MaizeFormerX: a lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis |
| Tac gia | Md Mostafizur Rahman et al. (11 tac gia) |
| Journal / Conference | Scientific Reports (Nature Publishing Group) |
| Nam | 2026 |
| DOI / Link | https://doi.org/10.1038/s41598-026-44550-0 |
| Nhom phan loai | Nhom A — co model moi, du thong tin de phuc dung |

---

## 2. KIEN TRUC MODEL

**Ten model:** `MaizeFormerX`  
**Loai:** Hybrid Vision Transformer (custom, khong dung pretrained backbone ngoai)  
**Framework:** PyTorch (cu the: Python 3.10, PyTorch, CUDA 12.1 — ghi ro trong paper)  
**Module dac biet:** MultiScalePatchEmbed + Cross-Scale Attention Fusion (CSAF)  
**Nguon xac nhan:** github.com/rezaul-h/MaizeFormerX/ (xac nhan 2026-05-27)

### Mo ta kien truc (xac nhan tu GitHub)

**Buoc 1 — MultiScalePatchEmbed** (src/models/maizeformerx/multi_scale_patch_embed.py):
- patch_sizes = [8, 16, 32], strides = [8, 16, 32]
- embed_dims = [64, 96, 128] — MOI scale co dimension RIENG (khong phai chung D)
- Moi branch: Conv2d(3, embed_dim_i, kernel_size=ps_i, stride=ps_i) + LayerNorm
- Output: list 3 tensor [B, N1, 64], [B, N2, 96], [B, N3, 128]
- Voi IMG_SIZE=224: N1=28^2=784, N2=14^2=196, N3=7^2=49

**Buoc 2 — CSAF** (src/models/maizeformerx/csaf.py):
- Input: list [[B, N1, 64], [B, N2, 96], [B, N3, 128]]
- fusion_dim = 192
- Co che: Project moi scale rieng: Linear(64->192), Linear(96->192), Linear(128->192)
- Concat tat ca: [B, N1+N2+N3, 192] = [B, 1029, 192]
- Self-attention tren toan bo 1029 tokens (num_heads=4, KHONG phai cross-attention)
- attn_dropout=0.1, proj_dropout=0.1
- residual=True: output = LayerNorm(concat + attn_out)

**Buoc 3 — Transformer Encoder** (src/models/maizeformerx/encoder.py):
- depth=6, num_heads=6, dim=192, mlp_ratio=4.0
- drop_path: linear schedule tu 0 den 0.1 qua 6 layers
- attn_dropout=0.1, proj_dropout=0.1, qkv_bias=True
- Pre-norm architecture: LayerNorm -> MHSA -> DropPath -> Residual -> LayerNorm -> MLP -> DropPath -> Residual

**Buoc 4 — Classifier:**
- Them CLS token (learnable nn.Parameter(zeros(1,1,192))) truoc encoder
- Positional embedding cho [CLS + 1029 patch tokens] = 1030 tokens
- Pooling: dung CLS token output[:, 0] — KHONG phai GlobalAvgPool
- Dropout(0.3) -> Linear(192, num_classes)

### Architecture flow (xac nhan tu GitHub)
```
Input [B, 3, 224, 224]
  -> MultiScalePatchEmbed -> [[B,784,64], [B,196,96], [B,49,128]]
  -> CSAF (project + concat + self-attn) -> [B, 1029, 192]
  -> prepend CLS token -> [B, 1030, 192]
  -> + positional embedding
  -> TransformerEncoder (6 layers, 6 heads) -> [B, 1030, 192]
  -> take CLS token [:, 0] -> [B, 192]
  -> Dropout(0.3) -> Linear(192, num_classes)
```

### Bang kien truc (xac nhan tu GitHub)

| Module | Chi tiet | Nguon |
|--------|---------|-------|
| MultiScalePatchEmbed | patch_sizes=[8,16,32], embed_dims=[64,96,128] (per-scale) | multi_scale_patch_embed.py |
| CSAF | fusion_dim=192, Linear proj x3, concat, self-attn (4 heads), residual+LN | csaf.py |
| CLS token | nn.Parameter(zeros(1,1,192)), prepended truoc encoder | encoder.py |
| Positional embed | nn.Parameter(zeros(1,1030,192)) cho toan bo sequence | encoder.py |
| Transformer Encoder | depth=6, num_heads=6, dim=192, mlp_ratio=4.0, drop_path linear 0->0.1 | encoder.py |
| Classifier | CLS[:, 0] -> Dropout(0.3) -> Linear(192, C) | encoder.py |

**Total params (paper):** 9.6M  
**GFLOPs (paper):** 1.05 @ 224x224  
**CPU Latency (paper):** 16.4 ms (i9-13900K)  
**GPU Latency (paper):** 1.42 ms (RTX 4090)

---

## 3. HYPERPARAMETERS

### Tu truc tiep tu paper (Table 4)

| Thong so | Gia tri | Vi tri trong paper |
|---------|--------|-------------------|
| Framework | PyTorch + CUDA 12.1 | Muc "Experimental models" |
| Optimizer | Adam | Table 4 |
| Loss function | CrossEntropy + weight decay regularization | Paper equation |
| Batch size | 32 | Table 4 |
| Learning rate | 5e-5 | Table 4 |
| Weight decay | 5e-5 | Table 4 |
| Dropout rate | 0.3 | Table 4 |
| Max Epochs | 30 | Table 4 |
| LR Schedule | CosineAnnealingLR | Table 4 |
| Warmup steps | 300 | Table 4 |
| Early stopping patience | 4 | Table 4 |
| Input size | 224x224 | Muc "Data preprocessing" |
| Normalization | Z-score / min-max (paper mo ta ca hai) | Muc "Data preprocessing" |
| Train/Val/Test split | 80% / 5% / 15% | Muc "Data preprocessing" |

**Ghi chu ve normalization:** Paper mo ta ca min-max va z-score normalization nhu la preprocessing bao gom phan tich histogram equalization, grayscale conversion, Gaussian filtering. Tuy nhien, viec dung pretrained ViT backbone va input 224x224 nhat quan voi ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Notebook su dung ImageNet stats vi day la default cho ViT-based architectures va paper noi "pretrained ViT backbone."

### Thong so xac nhan tu GitHub (khong con la assumption)

| Thong so | Gia tri | Nguon xac nhan |
|---------|--------|---------------|
| patch_sizes | [8, 16, 32] | multi_scale_patch_embed.py |
| embed_dims (per scale) | [64, 96, 128] | multi_scale_patch_embed.py |
| fusion_dim (CSAF) | 192 | csaf.py |
| num_heads_csaf | 4 | csaf.py |
| CSAF mechanism | self-attention (KHONG phai cross-attention) | csaf.py |
| num_layers | 6 | encoder.py depth=6 |
| num_heads_encoder | 6 (KHONG phai 8) | encoder.py |
| dim encoder | 192 (= fusion_dim) | encoder.py |
| mlp_ratio | 4.0 | encoder.py |
| drop_path | linear schedule 0->0.1 | encoder.py |
| attn_dropout | 0.1 | encoder.py + csaf.py |
| proj_dropout | 0.1 | encoder.py + csaf.py |
| qkv_bias | True | encoder.py |
| CLS token | co, learnable parameter | encoder.py |
| Pooling | CLS token [:, 0] (KHONG phai GAP) | encoder.py |

### Gia tri van con gia dinh (khong co trong paper hoac GitHub)

| Thong so | Gia tri dung | Ly do gia dinh |
|---------|------------|--------------|
| CosineAnnealingLR T_max | 30 (= MAX_EPOCHS) | Paper khong neu T_max cu the; dung total epochs |
| Warmup schedule | Linear warmup 0->LR trong 300 steps | Paper neu 300 warmup steps nhung khong mo ta loai warmup |
| Normalization | ImageNet mean/std | Train-from-scratch nhung architecture ViT-based; ImageNet stats la default an toan |

---

## 4. DATASET

### 4a. Dataset trong Paper (chi de tham khao — KHONG dung truc tiep)

| Muc | Chi tiet tu Paper |
|-----|---------|
| Dataset 1 | Dataverse — 15,822 images, 3 classes (Healthy/MLB/MSV) |
| Dataset 2 | Tanzania — 9,356 images, 3 classes (Healthy/MLN/MSV) |
| Dataset 3 | Plagues Maiz — 850 images, 16 classes (bacteria/fungi/virus subtypes) |
| Train/Val/Test (paper) | 80% / 5% / 15% |
| Input size (paper) | 224x224 |
| Normalization (paper) | Min-max [0,1] + Z-score + ImageNet stats (ViT backbone) |

### 4b. Dataset Thuc Te (dataset cua nguoi dung — dien sau khi co)

| Muc | Chi tiet |
|-----|---------|
| Ten dataset | (dien ten dataset cua nguoi dung) |
| So classes | Auto-detect tu folder (xem NUM_CLASSES trong notebook) |
| Danh sach classes | Auto-detect — xem `class_names` variable |
| Tong so anh | ... |
| Train / Val / Test | 75% / 15% / 10% (mac dinh; hay thay neu dataset co folder rieng) |
| Image size | 224x224 (theo paper) |
| Normalization | ImageNet mean/std |

---

## 5. AUGMENTATION (Scenario 2)

**Nguon:** Paper Table 2 (Dataverse dataset config — phu hop nhat voi bai toan 3-class)

| Ky thuat | Tham so | Nguon |
|---------|--------|-------|
| RandomHorizontalFlip | p=0.5 | Table 2 (Dataverse) |
| RandomRotation | degrees=30 (± 30 degrees) | Table 2 (Dataverse) |
| RandomResizedCrop | scale=(0.9, 1.0), size=224 | Table 2 (Zoom/Scale 0.9-1.1) |
| ColorJitter | brightness=0.2 | Table 2 (Brightness ±20%) |

**Ghi chu:** Paper dung augmentation dataset-specific. Notebook ap dung config cua Dataverse (pho bien nhat, 3 class co ban). Neu user dataset co nhieu class hon, co the them ColorJitter(contrast=0.2) theo Tanzania config.

---

## 6. KET QUA GOC TRONG PAPER (chi de tham khao — KHONG the so sanh truc tiep)

> **LU Y:** Ket qua duoi day la cua tac gia tren dataset goc cua paper (khac dataset cua ban). Khong the so sanh truc tiep.

| Dataset (cua paper) | Aug | Accuracy | F1-Score | MCC | Parameters |
|--------------------|-----|---------|---------|-----|-----------|
| Dataverse | x2 | 96.4±0.9% | ~97.97% | 93.6% | 9.6M |
| Dataverse | x4 | 96.1±1.0% | ~97.39% | 93.2% | 9.6M |
| Dataverse | x6 | **97.8±0.8%** | ~98.60% | **95.3%** | 9.6M |
| Tanzania | x2 | 96.9±0.9% | ~97.83% | 94.4% | 9.6M |
| Tanzania | x4 | **97.5±0.9%** | ~97.48% | 94.5% | 9.6M |
| Plagues Maiz | x4 | **96.9±0.9%** | ~95.2% | 94.2% | 9.6M |
| No-aug (Dataverse) | — | 94.0±1.0% | 93.4% | 91.6% | 9.6M |

### Ket qua thuc te tren dataset cua ban (dien sau khi chay notebook)

| Scenario | Accuracy | F1-Score (Macro) | Recall (Macro) | Parameters |
|----------|---------|-----------------|---------------|-----------|
| Baseline (No-Aug) | — | — | — | — |
| Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [x] Grad-CAM Visualization — Paper Fig 19-23: Grad-CAM tai CSAF output va encoder output. Notebook dung `pytorch-grad-cam` voi `reshape_transform` cho transformer output.
- [x] Class Imbalance Auto-Detection — Paper co xu ly class imbalance (Plagues Maiz ratio 17:1). Notebook tu dong tinh va ap dung `compute_class_weight`.
- [x] Dataset Distribution Bar Chart — Visualization so luong mau theo class.
- [ ] Ablation Study — Paper co Table 12 (w/o CSAF / w/o Transformer / w/o Multi-Scale). Khong implement vi can 3 model variant rieng biet, tang thoi gian chay len 3x.
- [ ] Cross-dataset Generalization — Paper Table 11. Khong implement vi user chi co 1 dataset.
- [ ] Web Application — Paper trien khai Grad-CAM web app. Ngoai pham vi notebook Kaggle.

---

## 8. OUTPUT FILES

```
papers/MaizeFormerX a lightweight/
├── inputs/
│   └── s41598-026-44550-0.pdf
└── outputs/
    ├── MaizeFormerX_Experiment.ipynb        <- notebook upload len Kaggle
    └── MaizeFormerX_ReproduceSummary.md     <- file nay
```

Sau khi chay xong tren Kaggle (`/kaggle/working/`):
```
results_summary.xlsx            <- Excel 2-sheet (Baseline + Augmented), anh nhung
baseline_learning_curve.png
baseline_confusion_matrix.png
aug_learning_curve.png
aug_confusion_matrix.png
gradcam_visualization.png
dataset_distribution.png
best_model_s1.pth               <- best checkpoint Scenario 1
best_model_s2.pth               <- best checkpoint Scenario 2
```

---

## 9. GHI CHU IMPLEMENTATION

### Cap nhat kien truc (2026-05-27) — xac nhan tu GitHub

Notebook phien ban truoc dung thong so sai (reverse-engineered tu param budget):
- SAI: patch_sizes=[4,8,16], embed_dim=256 (shared), cross-attention CSAF, GAP pooling, num_heads=8
- DUNG (GitHub): patch_sizes=[8,16,32], embed_dims=[64,96,128] (per-scale), fusion_dim=192, self-attention CSAF, CLS token pooling, num_heads_encoder=6, num_heads_csaf=4

### Quyet dinh implementation con lai

1. **Positional embedding:** Them learnable pos_embed cho [CLS + 1029 patch tokens] = 1030 tokens. GitHub co the dung hoac khong dung — standard practice nen giu.

2. **Pre-norm architecture:** Xac nhan qua encoder.py dung pre-norm (LayerNorm truoc MHSA va MLP). Dung DropPath (Stochastic Depth) voi linear schedule.

3. **Normalization (ImageNet mean/std):** Model train-from-scratch. ImageNet stats duoc giu de nhat quan voi cac experiment truoc. Co the thu nghiem voi custom dataset stats neu accuracy thap.

4. **Warmup:** Linear warmup trong 300 steps (tu paper Table 4). Loai warmup (linear vs cosine) khong anh huong nhieu den final accuracy.

5. **Grad-CAM reshape_transform:** Skip CLS token (index 0), lay fine-scale tokens [1:785] (N1=784, patch_size=8, grid 28x28), reshape thanh [B, 192, 28, 28] de GradCAM co the xu ly.

### Risk flags (sau khi xac nhan GitHub)
| Rui ro | Muc do | Xu ly |
|--------|--------|-------|
| Total params co the khac 9.6M (do per-scale embed_dims khac nhau) | THAP | Kiem tra sau khi chay `torchinfo.summary()` |
| Warmup linear vs cosine | THAP | It anh huong den final accuracy |
| T_max CosineAnnealing = epochs vs steps | THAP | Thuong dung epochs |
| Grad-CAM reshape_transform co the can chinh neu token layout khac | THAP | Co try/except fallback trong notebook |

### Lien ket tham khao
- Code chinh thuc: https://github.com/rezaul-h/MaizeFormerX/
- Dataverse dataset: https://doi.org/10.7910/DVN/LPGHKK
- Tanzania dataset: https://doi.org/10.17632/fkw49mz3xs.1
- Plagues Maiz: https://doi.org/10.6084/m9.figshare.10314539.v3
