# ReproduceSummary — CustomMobileNetV3Large

**Paper:** A customized MobileNetV3Large-based deep learning framework for plant disease detection  
**Authors:** Rahaman et al. (2026) — Discover Artificial Intelligence  
**Framework:** TensorFlow 2.x / Keras  
**Notebook:** `custommobilenetv3large.ipynb`

---

## Architecture (reproduced from paper)

| Component | Paper | Notebook |
|-----------|-------|----------|
| Backbone | MobileNetV3Large (ImageNet pretrained, frozen) | ✅ `MobileNetV3Large(include_top=False, weights='imagenet')` |
| Pooling | Global Average Pooling | ✅ `GlobalAveragePooling2D` |
| FC1 | Dense(256, ReLU) + Dropout(0.3) | ✅ |
| FC2 | Dense(128, ReLU) + Dropout(0.2) | ✅ |
| Output | Dense(N_classes, Softmax) | ✅ auto-detected |
| Total Params | ~3.28M | ✅ 3,278,618 |

## Training Protocol

| Setting | Paper | Notebook |
|---------|-------|----------|
| Phase 1 LR | 0.001 | ✅ |
| Phase 1 Epochs | 20 | ✅ |
| Phase 2 LR | 0.0001 | ✅ |
| Phase 2 Epochs | 15 | ✅ |
| Unfreeze from | Layer 35 | ✅ |
| Optimizer | Adam (β1=0.9, β2=0.999) | ✅ |
| Loss | Sparse Categorical CrossEntropy | ✅ |
| EarlyStopping | patience=3 | ✅ |
| ReduceLROnPlateau | factor=0.1, patience=3 | ✅ |

## Augmentation (Scenario 2)

| Technique | Paper | Notebook |
|-----------|-------|----------|
| Horizontal Flip | ✅ | ✅ `RandomFlip('horizontal')` |
| Rotation ±20% | ✅ | ✅ `RandomRotation(0.2)` |
| Zoom ±30% | ✅ | ✅ `RandomZoom(0.3)` |
| Brightness ±20% | ✅ | ✅ `RandomBrightness(0.2)` |
| Contrast ±20% | ✅ | ✅ `RandomContrast(0.2)` |

## Results on User Dataset (5 classes)

| Scenario | Accuracy | F1 (Macro) | Recall (Macro) |
|----------|----------|------------|----------------|
| Baseline (No-Aug) | 93.57% | 0.9365 | 0.9357 |
| Augmented | 86.86% | 0.8694 | 0.8677 |

**Dataset:** 5 classes — anthracnose, healthy, leaf_curl, leaf_spot, nutrient_deficiency  
**Split:** train=1597 / val=491 / test=373

## Deviations from Paper

- **NUM_CLASSES:** Paper uses 26 classes; adapted to user's 5-class dataset (`len(CLASS_NAMES)` auto-detected)
- **Dataset:** User's own plant disease dataset, not the paper's original hybrid dataset

## Workspace Compliance

- ✅ `tf.random.set_seed(42)` + `np.random.seed(42)` + `random.seed(42)`
- ✅ `MirroredStrategy` (2× Tesla T4)
- ✅ `strategy.scope()` wrapping all `build_model()` and `compile()` calls
- ✅ Class imbalance auto-detection (ratio < 1.5 → no weighting on this dataset)
- ✅ `class_weight=CLASS_WEIGHTS_DICT` passed to all `model.fit()` calls
- ✅ `tf.keras.backend.clear_session(); gc.collect()` between Scenario 1 and 2
- ✅ `NUM_CLASSES` auto-detected from `os.listdir(TRAIN_DIR)`
- ✅ `CLASS_NAMES` auto-detected from `image_dataset_from_directory`
- ✅ Learning Curve + Confusion Matrix saved as PNG (dpi=300), each in separate cell
- ✅ `classification_report` per class with `target_names=CLASS_NAMES`
- ✅ Excel export with 2 sheets + embedded images
