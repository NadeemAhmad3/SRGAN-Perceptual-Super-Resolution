# ESRGAN — Enhanced Super-Resolution Generative Adversarial Network

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-DIV2K-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scale_Factor-4×-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Paper-ECCV_2018-9B59B6?style=for-the-badge"/>
</p>

<p align="center">
  A from-scratch PyTorch implementation of <strong>ESRGAN</strong> (Wang et al., ECCV Workshop 2018),<br/>
  trained end-to-end on the DIV2K dataset inside a single 4-hour Kaggle GPU session.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Why ESRGAN Over SRGAN](#why-esrgan-over-srgan)
- [Architecture](#architecture)
  - [RRDB Generator](#rrdb-generator)
  - [Relativistic Discriminator](#relativistic-discriminator)
  - [Pre-Activation Perceptual Loss](#pre-activation-perceptual-loss)
- [Training Strategy](#training-strategy)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Concepts Implemented](#concepts-implemented)
- [References](#references)
- [Contact](#contact)

---

## Overview

Single-Image Super-Resolution (SISR) is the task of reconstructing a High-Resolution (HR) image from a single Low-Resolution (LR) input. It is a fundamentally **ill-posed inverse problem** — for any given LR image, infinitely many HR images could have produced it. The goal of a learned SR model is to find the most perceptually plausible mapping through that high-dimensional solution space.

This implementation goes beyond a vanilla SRGAN port. Every architectural and training decision in ESRGAN is a **targeted correction of a diagnosed failure mode** in its predecessor. This repository implements all five of those corrections from scratch, with detailed markdown commentary in the notebook explaining the *why* behind every design choice.

The entire pipeline — data degradation, generator, discriminator, perceptual loss, two-stage training, and inference — runs within a single Kaggle P100/T4 GPU session.

---

## Why ESRGAN Over SRGAN

| Dimension | SRGAN (Ledig et al., 2017) | ESRGAN (Wang et al., 2018) | Impact |
|---|---|---|---|
| **Generator block** | ResBlock + BatchNorm | RRDB — no BN | Eliminates ghosting/halo artefacts |
| **Discriminator loss** | Standard binary BCE | Relativistic Average (RaD) | Denser generator gradients; comparative realism |
| **Perceptual features** | Post-ReLU (sparse) | Pre-ReLU (dense) | Stronger supervision; sharper texture recovery |
| **Pixel loss** | MSE | L1 | Sharper edges; no regression-to-the-mean blurring |
| **Training strategy** | Single stage | PSNR pre-train → GAN fine-tune | Stable convergence; no cold-start collapse |
| **MOS score** | Baseline | **+0.37 improvement** | State-of-the-art perceptual quality |

---

## Architecture

### RRDB Generator

The generator is built around the **Residual-in-Residual Dense Block (RRDB)** — a two-level nested residual architecture with no Batch Normalisation anywhere in the network.

```
Input (LR)
    │
    ▼
Head Conv (3 → 64)
    │
    ▼
┌──────────────────────────────────────────┐
│            RRDB  ×  16                   │
│  ┌────────────────────────────────────┐  │
│  │      ResidualDenseBlock  ×  3      │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │       DenseLayer  ×  5       │  │  │
│  │  │  (concat all previous feats) │  │  │
│  │  └──────────────────────────────┘  │  │
│  │   output = RDB(x) * 0.2 + x       │  │
│  └────────────────────────────────────┘  │
│   output = RRDB(x) * 0.2 + x            │
└──────────────────────────────────────────┘
    │
    ▼
Body Conv + Global Residual
    │
    ▼
PixelShuffle ×2 → PixelShuffle ×2    (4× total upscale, no checkerboard artefacts)
    │
    ▼
Tail Conv → Tanh
    │
    ▼
Output (SR) in [-1, 1]
```

**Why no Batch Normalisation?**
BN was designed for classification — invariance to style. In image generation, exact pixel intensities matter. BN introduces ghosting artefacts by bleeding batch statistics across images and causes training instability by interacting destructively with discriminator gradients.

**Why dense connections?**
Every layer in an RDB receives the concatenated outputs of all previous layers. High-frequency spatial information from the input is never discarded — it remains accessible at every depth. SRGAN's simple skip connection `F(x) + x` does not have this property.

**Why PixelShuffle instead of transposed convolutions?**
Transposed convolutions produce checkerboard artefacts due to uneven kernel overlap. PixelShuffle rearranges `C × r²` channels into a `C`-channel image `r×` larger — mathematically artefact-free. Applied in two `2×` stages for the total `4×` upscale.

---

### Relativistic Discriminator

The standard SRGAN discriminator asks in isolation: *"Is this image real or fake?"* This is informationally asymmetric — the generator only learns to make fakes look real, with no incentive around comparative realism.

The **Relativistic Average Discriminator (RaD)** reframes this as a comparative question:

> *"Is the real image more realistic than the average fake? Is the fake less realistic than the average real?"*

$$D_{Ra}(x_r, x_f) = \sigma\bigl(C(x_r) - \mathbb{E}[C(x_f)]\bigr)$$

where `C(x)` is the **raw unactivated logit** — no sigmoid inside the discriminator.

This means every generator update uses information from **both real and fake batches simultaneously**, producing richer and more stable gradients. Empirically this leads to sharper textures and better-defined edges compared to SRGAN.

**Label smoothing** of `0.9` (instead of `1.0`) is applied to real targets to prevent discriminator overconfidence and sigmoid saturation, which would zero out generator gradients.

---

### Pre-Activation Perceptual Loss

Both SRGAN and ESRGAN use VGG-based perceptual loss. The critical difference is **where** in the VGG network features are extracted:

| | SRGAN | ESRGAN |
|---|---|---|
| Extraction point | After ReLU | Before ReLU (VGG19 layer 34) |
| Feature density | Sparse (~50% zeroed by ReLU) | Dense (all values preserved) |
| Gradient signal | Weaker | Stronger |
| Practical result | Blurry textures | Sharp, faithful fine detail |

**L1 vs MSE content loss:**
MSE penalises errors quadratically, pushing the generator toward the *mean* of all plausible solutions — producing over-smoothed, plasticky outputs. L1 penalises linearly, tolerating the high-frequency textures that GAN outputs produce naturally.

**Total generator loss:**

$$L_{\text{total}} = L_{\text{perceptual}} + \lambda \cdot L_{\text{adversarial}} + \eta \cdot L_1$$

| Term | Weight | Role |
|---|---|---|
| Perceptual | 1.0 | Dominant semantic alignment signal |
| Adversarial | λ = 5×10⁻³ | Texture hallucination — kept small to prevent semantic errors |
| L1 pixel | η = 1×10⁻² | Structural anchor — prevents colour drift and global distortion |

---

## Training Strategy

Training is split into two distinct stages — a deliberate curriculum that solves the cold-start problem endemic to deep GAN training.

```
┌──────────────────────────────────────────────────────────────────────┐
│   STAGE 1 — PSNR Pre-Training   (Epochs 1 – 40, ~1.5 hrs)          │
│                                                                       │
│   Generator only. Loss = L1 pixel loss.                              │
│   Optimiser : Adam + Cosine Annealing LR  (2e-4 → 1e-6)            │
│                                                                       │
│   Goal   : Learn correct spatial structure and colour distribution   │
│            before any adversarial confusion.                         │
│   Result : Blurry but structurally accurate outputs.                 │
│   Saves  : generator_pretrained.pth                                  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│   STAGE 2 — GAN Fine-Tuning   (Epochs 1 – 60, ~2.5 hrs)            │
│                                                                       │
│   Generator + Discriminator. Full composite loss.                    │
│   Optimiser : Adam + MultiStepLR  (halve at epoch 30 and 45)        │
│                                                                       │
│   Goal   : Hallucinate photo-realistic high-frequency textures.      │
│   Result : Sharp, perceptually competitive SR images.                │
│   Saves  : generator_best.pth  +  generator_final.pth               │
└──────────────────────────────────────────────────────────────────────┘
```

**Why two stages?**
A randomly-initialised 16-block RRDB generator (~16M params) immediately facing a discriminator will diverge. The discriminator trivially rejects all outputs, the generator receives uninformative gradients, and training collapses. Pre-training with L1 provides principled weight initialisation — vastly superior to random Xavier/He init for this specific task.

**Why MultiStepLR in Stage 2?**
The GAN loss landscape is non-stationary — every discriminator update shifts the surface the generator is optimising over. Cosine annealing continuously decays LR even during moments the generator needs large updates to respond to a suddenly stronger discriminator. MultiStepLR keeps LR flat between milestones (allowing both networks to reach equilibrium), then halves it at predefined checkpoints for progressive stability.

---

## Project Structure

```
esrgan-kaggle/
│
├── esrgan.ipynb                   # Main notebook — all phases with inline documentation
├── README.md                      # This file
│
└── outputs/  (generated at runtime inside /kaggle/working/)
    ├── generator_pretrained.pth   # Stage 1 checkpoint
    ├── generator_best.pth         # Best Stage 2 checkpoint (lowest generator loss)
    ├── generator_final.pth        # Final Stage 2 checkpoint
    ├── discriminator_final.pth    # Final discriminator checkpoint
    └── esrgan_results.png         # Inference visualisation — bicubic vs ESRGAN vs GT
```

---

## Dataset Setup

This notebook uses the **DIV2K** dataset (Diverse 2K resolution), containing 800 high-definition training images and 100 validation images spanning a wide variety of scenes — architecture, nature, people, text, and abstract patterns.

**Attach to your Kaggle notebook:**

1. Open your Kaggle notebook editor
2. Click **+ Add Data** (top right)
3. Search for `div2k-dataset` by `joe1995`
4. Direct link → [https://www.kaggle.com/datasets/joe1995/div2k-dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset)
5. Click **Add**

The notebook uses these exact mounted paths:

```python
DIV2K_TRAIN_PATH = '/kaggle/input/datasets/joe1995/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR'
DIV2K_VALID_PATH = '/kaggle/input/datasets/joe1995/div2k-dataset/DIV2K_valid_HR/DIV2K_valid_HR'
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| HR patch size | 128 × 128 | GPU memory efficiency + high texture variance per epoch |
| Scale factor | 4× | Standard SISR benchmark |
| Batch size | 16 | Stable GAN training on P100/T4 |
| Generator LR | 2×10⁻⁴ | Adam standard for deep generative models |
| Discriminator LR | 1×10⁻⁴ | Slower than generator to prevent discriminator dominance |
| λ adversarial | 5×10⁻³ | Paper-specified; adversarial as texture refinement, not dominant objective |
| η L1 | 1×10⁻² | Paper-specified; pixel anchor to prevent structural drift |
| Pre-train epochs | 40 | ~1.5 hrs on P100 — sufficient for structural convergence |
| GAN epochs | 60 | ~2.5 hrs on P100 — sufficient for texture fine-tuning |
| Residual scaling | 0.2 | Prevents gradient explosion in 16-block deep RRDB stack |
| Label smoothing | 0.9 | Prevents discriminator sigmoid saturation |
| RRDB blocks | 16 | Paper recommendation; yields ~16M generator parameters |
| Growth rate (RDB) | 32 | Controls dense block channel expansion per layer |

---

## Results

After training, the inference cell produces a side-by-side comparison across 4 validation images:

| Column | Description |
|---|---|
| **Bicubic Upscale** | Classical baseline — smooth, blurry, no texture recovery |
| **ESRGAN Output** | Network's hallucinated high-frequency reconstruction |
| **Ground Truth HR** | Original 4× image — the perceptual upper bound |

**A note on PSNR vs perceptual quality:**
ESRGAN outputs will typically have a *lower* PSNR than bicubic interpolation. This is expected and by design — hallucinated textures are not pixel-accurate to ground truth (that information was destroyed by downsampling), but they are perceptually superior. The field evaluates SR models using **LPIPS** (Learned Perceptual Image Patch Similarity) and **MOS** (Mean Opinion Score), both of which ESRGAN directly optimises for.

---

## Concepts Implemented

- [x] Custom PyTorch `Dataset` with stochastic Gaussian blur degradation (σ ∈ [0.2, 1.5])
- [x] Random HR patch extraction with horizontal/vertical flip augmentation
- [x] `DenseLayer` → `ResidualDenseBlock` → `RRDB` module hierarchy
- [x] BN-free deep generator with residual scaling (β = 0.2) at both RDB and RRDB level
- [x] Two-stage PixelShuffle upsampling (2× → 2× = 4×, no checkerboard artefacts)
- [x] VGG-style discriminator outputting raw logits — no internal sigmoid
- [x] Relativistic Average Discriminator loss with label smoothing (0.9)
- [x] Pre-activation VGG19 feature extractor truncated at layer 34
- [x] ImageNet-normalised VGG input pipeline with buffer-registered mean/std
- [x] L1 perceptual content loss
- [x] Composite generator loss — perceptual + λ·adversarial + η·L1
- [x] Stage 1: L1-only pre-training with cosine annealing LR schedule
- [x] Stage 2: Full GAN training with MultiStepLR milestone schedule
- [x] Gradient clipping (norm = 1.0) on both generator and discriminator
- [x] Best-checkpoint saving based on lowest generator loss across all GAN epochs
- [x] Inference visualisation — bicubic vs ESRGAN vs ground truth

---

## References

```bibtex
@inproceedings{wang2018esrgan,
  title     = {ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
  author    = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao
               and Dong, Chao and Qiao, Yu and Loy, Chen Change},
  booktitle = {European Conference on Computer Vision Workshops},
  year      = {2018}
}

@inproceedings{ledig2017srgan,
  title     = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
  author    = {Ledig, Christian and Theis, Lucas and Huszar, Ferenc and Caballero, Jose
               and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan
               and Totz, Johannes and Wang, Zehan and Shi, Wenzhe},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2017}
}

@article{jolicoeur2018relativistic,
  title   = {The Relativistic Discriminator: a Key Element Missing from Standard GAN},
  author  = {Jolicoeur-Martineau, Alexia},
  journal = {arXiv preprint arXiv:1807.00734},
  year    = {2018}
}

@inproceedings{agustsson2017div2k,
  title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  author    = {Agustsson, Eirikur and Timofte, Radu},
  booktitle = {CVPR Workshops},
  year      = {2017}
}
```

---

## Contact

**Nadeem Ahmad**

<p>
  <a href="mailto:engrnadeem26@gmail.com">
    <img src="https://img.shields.io/badge/Email-engrnadeem26@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/nadeem-ahmad3/">
    <img src="https://img.shields.io/badge/LinkedIn-Nadeem_Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
</p>

---

<p align="center">
  Built with PyTorch &nbsp;·&nbsp; Trained on Kaggle &nbsp;·&nbsp; Based on Wang et al., ECCV Workshop 2018
</p>
