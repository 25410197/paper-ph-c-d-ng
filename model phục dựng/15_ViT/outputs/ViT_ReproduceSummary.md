# ViT — Reproduce Summary

**Paper:** Plant disease detection using vision transformers  
**Authors:** Mhaned Ali, Mouatassim Salma, Mounia El Haji, Benhra Jamal  
**Journal:** IJECE Vol. 15 No. 2, April 2025, pp. 2334–2344  
**Framework:** PyTorch *(paper dùng Keras ngầm định qua Table 2 conventions — intentional deviation, accepted by user)*

---

## Architecture Decisions

| Param | Paper (Table 2) | Notebook | Match |
|-------|----------------|----------|-------|
| Input size | 256×256×3 | 256×256 | ✓ |
| Patch size | 16×16 | 16 | ✓ |
| Embedding dim | 512 | 512 | ✓ |
| TransformerEncoder layers | 4 | 4 | ✓ |
| Attention heads | 8 | 8 | ✓ |
| FFN dim | Not stated | 2048 | Assumed |
| Dense head | 256→2048→1024→512→128→64→32→55 | Same | ✓ |
| Dense activation | Not stated | ReLU | Assumed |
| Optimizer | Adam, LR=1e-4 | Adam, LR=1e-4 | ✓ |
| Loss | Sparse cat. cross-entropy | CrossEntropyLoss | ✓ |
| Batch size | 32 | 32 | ✓ |
| Epochs | 20 | 20 | ✓ |

## Assumptions Made

- **Framework:** PyTorch (paper implies Keras via `(None,...)` shapes, `SoftMax`, `sparse categorical cross entropy`)
- **FFN dim:** 2048 (standard ViT-Base ratio; paper not stated)
- **Dense layer activation:** ReLU (paper not stated)
- **Normalization:** ImageNet mean/std (paper not stated; model trained from scratch)

## Augmentation (Section 2.3)

| Technique | Paper | Notebook |
|-----------|-------|---------|
| Horizontal Flip 50% | ✓ | RandomHorizontalFlip(p=0.5) |
| Crop 0–10% | ✓ | RandomResizedCrop(scale=(0.90,1.0)) |
| Contrast [0.75, 1.5] | ✓ | ColorJitter(contrast=(0.75,1.5)) |
| Gaussian Noise σ=0.05×255 | ✓ | Lambda(x + 0.05×randn_like(x)) |
| Brightness [0.8, 1.2] | ✓ | ColorJitter(brightness=(0.8,1.2)) |
| Affine rot±5°, shear±16° | ✓ | RandomAffine(degrees=5, shear=16) |
