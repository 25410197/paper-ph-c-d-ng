# TrIncNet — Reproduce Summary
**Paper:** TrIncNet: a lightweight vision transformer network for identification of plant diseases
**Authors:** Pushkar Gole, Punam Bedi, Sudeep Marwaha, Md. Ashraful Haque, Chandan Kumar Deb
**Source:** Frontiers in Plant Science, Vol. 14, 2023 — DOI: 10.3389/fpls.2023.1221557
**Reproduction date:** 2026-06-16
**Notebook output:** `papers/03 - TrIncNet/outputs/TrIncNet_Experiment.ipynb`

---

## 1. PAPER INFORMATION

| Field | Detail |
|-------|---------|
| Title | TrIncNet: a lightweight vision transformer network for identification of plant diseases |
| Authors | Gole P., Bedi P., Marwaha S., Haque MA., Deb CK. |
| Journal / Conference | Frontiers in Plant Science |
| Year | 2023 (Published 27 July 2023) |
| DOI / Link | https://doi.org/10.3389/fpls.2023.1221557 |
| Classification Group | Group A (novel architecture with detailed hyperparameters, reproducible) |

---

## 2. MODEL ARCHITECTURE

**Model name:** `TrIncNet` (Trans-Inception Network)
**Type:** Hybrid ViT + Inception (Vision Transformer with Inception module replacing MLP)
**Framework:** TensorFlow 2.13 / Keras (paper used TF 2.6.0 — confirmed in Section 4.3)
**Special modules:** Trans-Inception block — MHA + LayerNorm + Inception module + skip connection

### Architecture Flow

```
Input (256, 256, 3)
  -> /255 normalization
  -> Patch Embedding: Conv2D(filters=256, kernel=16, stride=16) -> (batch, 16, 16, 256)
  -> Reshape to sequence: (batch, 256, 256)
  -> Prepend learnable CLS token: (batch, 257, 256)
  -> Add learnable positional embedding (1, 257, 256)
  -> 2x Trans-Inception Block
        [each block]:
          -> LayerNorm(eps=1e-6)
          -> MultiHeadAttention(num_heads=12, key_dim=21)
          -> Add (residual around attention)
          -> LayerNorm(eps=1e-6)
          -> Separate CLS token; reshape patch tokens (batch,256,256) -> (batch,16,16,256)
          -> Inception Module (4 branches, concat=224 channels)
          -> Reshape back to (batch, 256, 224)
          -> Dense projection (224 -> 256) for CLS+patch tokens
          -> Reattach CLS token: (batch, 257, 256)
          -> Add (skip connection around whole block)
  -> Extract CLS token [batch, 0, :]  -> (batch, 256)
  -> Dense(num_classes, activation='softmax')
```

### Inception Module Detail

| Branch | Operations | Filters | Output channels |
|--------|-----------|---------|----------------|
| 1 | Conv2D 1x1, same, relu | 96 | 96 |
| 2 | Conv2D 1x1 (reduce), same, relu -> Conv2D 3x3, same, relu | 16 -> 64 | 64 |
| 3 | Conv2D 1x1 (reduce), same, relu -> Conv2D 5x5, same, relu | 128 -> 32 | 32 |
| 4 | MaxPool2D 3x3 (stride=1, same) -> Conv2D 1x1, same, relu | 32 | 32 |
| — | Concatenate along channel axis | — | **224 total** |

Post-Inception: Dense projection 224 -> 256 before residual add (embed_dim match).

**Total params (paper):** ~6.95M

---

## 3. HYPERPARAMETERS

### Confirmed directly from paper

| Parameter | Value | Location in paper |
|-----------|-------|------------------|
| Framework | TensorFlow 2.6.0 + Keras | Section 4.3 |
| Input size | 256 x 256 x 3 | Table 1, Table 2, Section 4.2.1 |
| Patch size | 16 x 16 | Table 1, Table 2 |
| Embedding dim (N) | 256 | Table 2 |
| Num Trans-Inception blocks | 2 | Table 2 |
| Num attention heads (m) | 12 | Table 2, Section 3.2.1 |
| LayerNorm epsilon | 1e-6 | Table 2 |
| Inception filter values | 96, 16, 64, 128, 32, 32 | Table 2 |
| Inception padding | same | Table 2 |
| Inception stride | 1 x 1 | Table 2 |
| Activation (output) | softmax | Table 2 |
| Activation (hidden) | relu | Table 2 |
| Optimizer | Adam | Section 4.3 |
| Loss function | Categorical Cross-Entropy | Section 4.3 |
| Batch size | 32 | Section 4.3 |
| Max epochs | 500 | Section 4.3 |
| Early stopping patience | 20 (monitor: val_accuracy) | Section 4.3 |
| Image normalization | /255 — range [0, 1] | Section 4.2.2 |
| Data split | 70% train / 15% val / 15% test | Section 4.2.4 |

### Assumed values (not stated in paper)

| Parameter | Value used | Reason |
|-----------|-----------|--------|
| Learning rate | 1e-4 | Confirmed by user (paper only states "Adam optimizer") |
| CLS token | Yes, learnable tf.Variable | ViT standard (Dosovitskiy 2021); paper references ViT architecture |
| Positional embedding | Learnable tf.Variable (1, 257, 256) | ViT standard; paper does not specify sinusoidal vs learnable |
| key_dim per head | 256 // 12 = 21 | Standard ViT formula |
| Projection layer after Inception | Dense(256, relu) | Required to reconcile 224 concat channels with 256 embed_dim for residual add |
| Dataset split | Pre-split by user (train/val/test folders) | User provided pre-split dataset — no re-splitting applied |

---

## 4. DATASET

### 4a. Datasets in Paper (reference only)

| Field | Paper Detail |
|-------|-------------|
| Dataset 1 | PlantVillage — 38 classes, 54,503 images, 14 plant species |
| Dataset 2 | Maize — 4 classes (Healthy, TLB, BLSB, MLB), 13,971 images (augmented to 100,000) |
| Split ratio (paper) | 70% train / 15% val / 15% test (random) |
| Input size | 256 x 256 |
| Normalization (paper) | /255 — range [0, 1] |
| Augmentation (paper) | Offline for Maize only (rotation, flipping); PlantVillage no augmentation |

### 4b. Actual Dataset (user's Kaggle dataset)

| Field | Detail |
|-------|---------|
| Dataset name | dataset_smartcrop_plit_300 |
| Dataset path | `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` |
| Pre-split | Yes — 3 folders: `train/`, `val/`, `test/` |
| Num classes | Auto-detected from `train/` subdirectories |
| Class names | Auto-detected (`sorted(os.listdir(TRAIN_DIR))`) |
| Image size | 256 x 256 (resized by ImageDataGenerator) |
| Normalization | /255 (same as paper) |

---

## 5. AUGMENTATION (Scenario 2)

| Technique | Parameter | Source |
|-----------|-----------|--------|
| Rotation | rotation_range=30 | Required by CORE RULE #3 (paper mentions rotation in Section 4.2.3 without specific value) |
| Zoom / Scale | zoom_range=0.2 | Required by CORE RULE #3 |
| Horizontal Flip | horizontal_flip=True | Required by CORE RULE #3 (paper mentions "flipping" in Section 4.2.3) |
| Brightness | brightness_range=[0.8, 1.2] | Required by CORE RULE #3 |
| fill_mode | nearest | Keras default |

Val/Test generators: no augmentation (rescale=/255 only), identical to Scenario 1.

---

## 6. PAPER RESULTS (reference only — different dataset)

| Dataset (paper) | Accuracy | F1-Score | Recall | Parameters |
|-----------------|---------|---------|--------|-----------|
| PlantVillage (38 class) | 99.93% | 99.91% | 99.91% | 6.95M |
| Maize (4 class) | 96.93% | 96.90% | 96.83% | 6.95M |

### Actual Results (fill after running notebook)

| Scenario | Accuracy | F1-Score | Recall | Parameters |
|----------|---------|---------|--------|-----------|
| Scenario 1 — Baseline (No-Aug) | — | — | — | — |
| Scenario 2 — Augmented | — | — | — | — |

---

## 7. SPECIAL FEATURES IMPLEMENTED

- [x] Class imbalance auto-detection (max/min >= 1.5 threshold, sklearn compute_class_weight)
- [x] Class names auto-detect from train directory
- [x] Best model checkpoint (val_accuracy) — reload before test eval (best_model_s1.h5, best_model_s2.h5)
- [x] GPU memory management (tf.keras.backend.clear_session() + gc.collect() between scenarios)
- [ ] Grad-CAM Visualization — not implemented (paper has ablation feature maps but no Grad-CAM)
- [ ] Ablation Study — not implemented (paper ablation is feature map visualization, not quantitative per-variant)

---

## 8. OUTPUT FILES

```
papers/03 - TrIncNet/
├── input/
│   ├── fpls-14-1221557.pdf
│   ├── paper_content.md        <- extracted content
│   └── implement_plan.md       <- architecture analysis
└── outputs/
    ├── TrIncNet_Experiment.ipynb
    └── TrIncNet_ReproduceSummary.md
```

Kaggle working directory outputs (generated at runtime):
```
/kaggle/working/
├── best_model_s1.h5            <- best checkpoint Scenario 1
├── best_model_s2.h5            <- best checkpoint Scenario 2
├── learning_curve_s1.png       <- dpi=300
├── confusion_matrix_s1.png     <- dpi=300
├── learning_curve_s2.png       <- dpi=300
├── confusion_matrix_s2.png     <- dpi=300
└── results_summary.xlsx        <- 2 sheets: Baseline + Augmented
```

---

## 9. IMPLEMENTATION NOTES

### 9.1 Channel mismatch — Inception output vs residual

The Inception module's 4 branches produce: 96 + 64 + 32 + 32 = **224 channels**, but the
residual (skip connection) requires **256 channels** (embed_dim). Resolution: a Dense(256, relu)
projection is applied to both patch tokens (after Inception reshape) and the CLS token before
the residual Add. This is the key non-trivial wiring decision not stated in the paper.

### 9.2 Inception filter mapping (Table 2 interpretation)

Paper Table 2 gives 6 filter values (96, 16, 64, 128, 32, 32) without assignment to specific
convolutions. Mapped per Appendix Table A1 Inception-v1 convention:
- F1=96 (op1: 1x1 direct), F2=16 (op2: reduce for 3x3), F5=64 (op5: 3x3 output),
- F3=128 (op3: reduce for 5x5), F6=32 (op6: 5x5 output), F7=32 (op7: after maxpool)

Note: F3=128 is an unusually large reduce filter for a 5x5 branch (GoogLeNet typically uses
small reduces for 5x5). If total params deviate significantly from 6.95M, reducing F3 (e.g.,
to 32) is the recommended first adjustment as stated in the implement_plan.md.

### 9.3 CLS token and positional embedding — Keras functional API limitation

tf.Variable cannot be directly registered as a Keras layer weight in the functional API.
Lambda layers are used to broadcast the CLS token and add positional embedding. This means
CLS token and positional embedding weights appear as non-trainable in model.summary() when
defined outside the functional graph. An alternative approach (EmbeddingLayer subclass) would
register them properly but adds complexity.

### 9.4 Normalization — NOT ImageNet

Paper Section 4.2.2 explicitly states /255 normalization to range [0, 1].
ImageNet mean/std (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is NOT used.
This is intentional — TrIncNet trains from scratch (no pretrained backbone).

### 9.5 Dataset split

The user's dataset is pre-split into train/val/test folders. The 70/15/15 split ratio from
the paper is not re-applied. The notebook uses the folders as-is.

### 9.6 Early stopping on Kaggle GPU

Paper trains on NVidia Tesla V100-SXM2 32 GB (DGX Server). Kaggle provides T4/P100 GPUs.
With max_epochs=500 and early stopping patience=20, actual training epochs depend on
convergence. If time constraint is a concern on Kaggle (9-12h limit), consider reducing
EPOCHS in Cell 3 before running.
