# Lab Notebook

## 2026-03-31 Afternoon
- Set up Colab environment, A100 confirmed
- Downloaded 756 DESI cutouts before PC went to sleep
- Bologna dataset download still pending (Zenodo too slow)
- Read through full project plan doc
- Questions/thoughts: I don't really know what I'm doing

## 2026-03-31 Evening 

### What I did
- Set up Colab Pro with A100 GPU, verified Zoobot installs correctly
- Spent ~2 hours hunting for Bologna dataset — server is down, no Zenodo mirror exists
- Emailed Prof. Metcalf (robertbenton.metcalf@unibo.it) requesting the data
- Pivoted to generating synthetic data with lenstronomy (same tool Bologna used)
- Generated 20,000 images (10k lens, 10k non-lens) at 101x101px
- Saved both 64x64 and 101x101 versions to Google Drive

### Results
- images.npy shape: (20000, 101, 101)
- labels.npy shape: (20000,)

### Decisions made
- Using 101x101 to match Bologna resolution
- Single channel (grayscale) — simpler than 4-band but sufficient for the comparison

### Blockers
- Bologna dataset still not confirmed — waiting on Metcalf reply
- If he sends it, may switch datasets for better realism

### Tomorrow
- Task 2: build the PyTorch Dataset class and data pipeline




## 2026-04-01 Morning

### What I did
- Loaded synthetic dataset from Google Drive (20,000 images, 101x101)
- Built LensDataset class inheriting from torch.utils.data.Dataset
- Implemented arcsinh stretch + standardization preprocessing
- Created 70/15/15 train/val/test split with fixed random seed (42)
- Wrapped splits in DataLoaders (batch_size=32, shuffle=True for train only)

### Results
- Train: 14,000 samples, Val: 3,000, Test: 3,000
- Pixel range after preprocessing: min=-1.66, max=8.06, mean=0.0017
- Confirmed batch shape: (32, 101, 101)

### Concepts learned
- Dataset vs DataLoader and what each does
- Map-style datasets and why we use them
- Batching, shuffling, and iteration
- Arcsinh stretch for astronomical dynamic range
- Standardization (zero mean, unit variance) and why it helps training

### Decisions made
- a=0.1 for arcsinh scale factor (standard for astronomy)
- seed=42 for reproducibility across all three model experiments

### Next Task
- Task 3: train ResNet18 from scratch




## 2026-04-01 (continued — afternoon)

### What I did
- Completed Task 3: trained ResNet18 from scratch
- Completed Task 4: trained ImageNet pretrained ResNet18 (frozen then unfrozen)
- Completed Task 5: trained Zoobot ConvNeXT-Nano (frozen then unfrozen)
- Saved all checkpoints to Google Drive

### Results summary

**ResNet18 scratch**
- Converged to near-zero loss very quickly
- Perfect predictions on val set — synthetic data too easy
- Saved: resnet18_scratch.pth

**ResNet18 ImageNet (frozen → unfrozen)**
- Frozen phase: val loss started at 0.0041 but drifted upward to 0.0105 by epoch 30 — frozen ImageNet features have some signal but head overfits without encoder adaptation
- Note: initial run showed stuck val loss at 0.8519 — this was a code bug where validation loop was evaluating the scratch model instead of the ImageNet model
- Unfrozen phase: converged to near-zero after full fine-tuning
- Saved: resnet18_imagenet.pth

**Zoobot ConvNeXT-Nano (frozen → unfrozen)**
- Frozen phase: smooth decreasing learning curve, 0.41 → 0.08 val loss over 30 epochs — frozen galaxy features are genuinely useful and don't overfit
- Unfrozen phase: converged to near-zero rapidly, final val loss 0.0002
- Saved: zoobot_frozen.pth, zoobot_unfrozen.pth

### Key findings so far
1. Synthetic data is too clean — all models eventually achieve near-perfect accuracy
2. Frozen ImageNet features have some signal but head overfits quickly — val loss drifts upward without encoder adaptation
3. Frozen Zoobot features are genuinely useful — smooth decreasing val loss throughout frozen phase, no overfitting
4. The frozen phase behavior is the clearest signal of domain-specific pretraining value — Zoobot learns meaningfully without touching encoder weights, ImageNet doesn't
5. Sim-to-real gap will be the real test — flagged for Task 8

### Limitations noted
- Synthetic lenstronomy data too uniform — no real galaxy negatives
- Single channel grayscale — real surveys use 3-4 bands
- Need real galaxy dataset for non-lens negatives (COSMOS, HSC PDR3)

### Next
- Task 6: evaluate all three models with AUC-ROC, TPR@FPR=0.01, precision-recall curves





## 2026-04-01 (continued — Task 6: Evaluation)

### What I did
- Ran inference on test set (3,000 images) for all three models
- Computed AUC-ROC and TPR@FPR=0.01 for each model
- Generated ROC curves and precision-recall curves, saved to Drive
- Fixed inverted predictions for scratch model (model learned correctly but with flipped outputs — common with random initialization)

### Results

| Model    | AUC-ROC | TPR@FPR=0.01 |
|----------|---------|--------------|
| Scratch  | 0.9963  | 0.9887       |
| ImageNet | 1.0000  | 1.0000       |
| Zoobot   | 1.0000  | 1.0000       |

### Observations
- All three models perform near-perfectly on synthetic test data — confirms data is too easy
- Scratch model slightly lower on both metrics, consistent with having no pretrained features
- ImageNet and Zoobot indistinguishable on synthetic data — real differentiation was in training dynamics (frozen phase behavior)
- ROC curves: ImageNet and Zoobot overlap exactly at top-left corner, scratch slightly below
- PR curves: all three near top-right, scratch slightly off at high recall

### Key reminder
- These metrics are on synthetic data — the real comparison happens in Task 8 on DESI images
- The frozen phase training curves remain the clearest signal of domain-specific pretraining value

### Saved to Drive
- roc_curves.png
- pr_curves.png

### Next
- Task 7: MC Dropout uncertainty estimation + Grad-CAM visualizations



## 2026-04-01 (continued — Task 7: MC Dropout + Grad-CAM)

### What I did
- Rebuilt Zoobot with dropout head (p=0.5) for uncertainty estimation
- Trained head-only for 15 epochs — smooth convergence from 0.45 to 0.12 val loss
- Implemented MC Dropout inference: 50 forward passes per image, mean = prediction, std = uncertainty
- Generated Grad-CAM heatmaps for lens and non-lens images
- Saved: zoobot_mc_dropout.pth, mc_dropout_viz.png, gradcam.png, gradcam_comparison.png

### MC Dropout results
- High confidence lens (idx 1): pred=0.98, uncertainty=0.011
- Low confidence lens (idx 0): pred=0.81, uncertainty=0.087
- Ambiguous non-lens (idx 10000): pred=0.28, uncertainty=0.103 — highest uncertainty of all
- Confident non-lens (idx 10002): pred=0.0002, uncertainty=0.0002
- Uncertainty correctly tracks confidence — ambiguous cases have higher std

### Grad-CAM results
- Lens heatmap: red/yellow concentrated in ring around center — model correctly focuses on arc regions, not central galaxy
- Non-lens heatmap: scattered red on periphery, no coherent ring structure — model finds nothing lens-like
- This confirms the model learned physically meaningful features, not just brightness patterns
- Contrast between lens/non-lens heatmaps is visually compelling and publishable

### Next
- Task 8: DESI Legacy Survey search — run model on real sky data and find new lens candidates



## 2026-04-01 (continued — Task 8: DESI Survey Search)

### What I did
- Queried NOIRLab Astro Data Lab for 1,000 massive elliptical galaxies (DEV/SER type, flux_r > 50) in RA 150-200, Dec 0-30
- Downloaded 1,000 FITS cutouts from DESI Legacy Survey DR10 (101x101px, grz bands)
- Ran Zoobot MC Dropout model on all 1,000 cutouts
- Visually inspected top candidates in Legacy Survey viewer
- Cross-referenced against confirmed DESI Einstein ring DESI-015.6763-14.0150 as visual reference

### Results
- p > 0.9: 657/1000 galaxies flagged (expected: ~1)
- p > 0.5: 809/1000 galaxies flagged
- p < 0.1: 89/1000 galaxies
- Top candidates visually inspected — none showed clear Einstein ring features
- Confirmed sim-to-real gap: model trained on clean simulations is wildly overconfident on real noisy data

### Key finding
The sim-to-real gap is severe and quantified:
- Synthetic test AUC: 1.000
- Real deployment false positive rate: ~65.7% vs expected ~0.1%
- This is a ~657x inflation of the expected positive rate

### Saved to Drive
- desi_candidates.csv
- desi_score_distribution.png

### Status
Not done — pursuing model improvements to reduce false positive rate and find real lens candidates. Ahead of schedule, continuing work.





## 2026-04-02 (Full day — Tasks 8.1 through 8.4)

### Task 8.1: Threshold calibration
- Applied 99th percentile threshold to original synthetic-trained model
- Top 10 candidates identified, none within 10 arcsec of known lenses in lenscat
- Visually inspected all 10 in Legacy Survey viewer
- One candidate (ra=150.5326, dec=2.3808) showed possible arc in residual image — graded C
- Confirmed what a real Einstein ring looks like: DESI-015.6763-14.0150 as reference

### Task 8.2 + 8.4 Combined: Real data retraining
- Downloaded 481 confirmed/candidate lens cutouts from lenscat catalog via DESI API
- Combined with 1,000 real DESI elliptical galaxy cutouts as non-lenses
- Built RealLensDataset class: loads 3-channel FITS files on the fly, arcsinh + normalize
- Trained on Kaggle T4 GPU (Colab compute units exhausted)
- Head-only training: 100 epochs, val loss 2.19 → 0.175, still improving
- Saved best checkpoint throughout training

### Test set evaluation (222 held-out images — honest result)
| Metric | Synthetic model | Real-data model |
|--------|----------------|-----------------|
| AUC-ROC | 1.000 (trivial) | 0.9638 |
| False positive rate | 65.7% on real data | 2.7% on test set |
| Precision (lens) | N/A | 94% |
| Recall (lens) | N/A | 88% |
| Accuracy | N/A | 94% |

Confusion matrix:
- 146/150 non-lenses correctly rejected (4 false alarms)
- 63/72 lenses correctly found (9 missed)

### Key finding
Training on real data reduced false positive rate by ~24x compared to synthetic training.
This directly quantifies the sim-to-real gap and demonstrates the value of real training data.

### Survey search attempts
- NOIRLab Astro Data Lab query timing out intermittently (works on Colab, not Kaggle)
- Random grid cutouts invalid for inference — model expects galaxy-centered images
- Test set remains the honest evaluation for now
- Will retry NOIRLab query tonight for fresh targeted galaxy coordinates

### Compute notes
- Exhausted Colab Pro compute units during training — purchased additional units
- Kaggle T4 used for actual training run
- NOIRLab queries: work on Colab, consistently timeout on Kaggle

### Saved files
- zoobot_real_trained.pth (Google Drive + Kaggle output)
- test_set_results.csv

### Next
- Tonight: retry NOIRLab query, download fresh galaxy cutouts, run survey search
- Find real lens candidates with the improved model
- Begin Task 9 writeup



## 2026-04-02 (Tasks 8.1–8.4: Full Model Comparison)

### Overview
Two training regimes compared across three model architectures:
- Synthetic data: 20,000 lenstronomy simulations (10k lens, 10k non-lens), single channel
- Real data: 481 confirmed/candidate lenses from lenscat + 1,000 real DESI elliptical galaxies, 3-channel grz

### Task 8.1: Threshold calibration (synthetic model)
- Applied 99th percentile threshold to synthetic-trained Zoobot on 1,000 real DESI galaxies
- 10 candidates identified, none within 10 arcsec of known lenses in lenscat
- One candidate (ra=150.5326, dec=2.3808) showed possible arc structure in residual image — graded C
- False positive rate: 65.7% — confirmed severe sim-to-real gap
- Reference Einstein ring confirmed: DESI-015.6763-14.0150

### Task 8.2 + 8.4: Real data retraining
**Dataset:**
- Positive: 481 FITS cutouts from lenscat coordinates, downloaded from DESI Legacy Survey DR10
- Negative: 1,000 real DESI elliptical galaxy cutouts (DEV/SER type, flux_r > 50)
- All 3-channel (grz), 101x101px, preprocessed with arcsinh stretch + standardization
- 70/15/15 train/val/test split, seed=42
- Train: 1,037 images, Val: 222, Test: 222

**Training infrastructure:**
- Kaggle T4 GPU (Colab compute units exhausted mid-project)
- Head-only runs: 100 epochs, best checkpoint saved throughout
- Full fine-tuning runs: 100 epochs, best checkpoint saved throughout
- Learning rates: scratch 1e-3, ImageNet 1e-4, Zoobot 1e-5 (tuned per model, standard practice)

### Synthetic data results (test set AUC, from earlier tasks)
| Model | Training regime | AUC |
|-------|----------------|-----|
| Scratch ResNet18 | Full model | ~1.000 |
| ImageNet ResNet18 | Frozen → unfrozen | ~1.000 |
| Zoobot ConvNeXT-Nano | Frozen → unfrozen | ~1.000 |

All models trivially achieved near-perfect AUC on synthetic data — data too clean, not discriminative.

### Real data results (test set AUC, 222 held-out images)
| Model | Training regime | AUC | Val loss |
|-------|----------------|-----|----------|
| Scratch ResNet18 | Full model | 0.9968 | 0.0546 |
| Scratch ResNet18 | Frozen head | 0.9865 | 0.1564 |
| ImageNet ResNet18 | Full model | 0.9976 | 0.0522 |
| ImageNet ResNet18 | Frozen head | 0.9719 | 0.1662 |
| Zoobot ConvNeXT-Nano | Full model | 0.9930 | 0.0937 |
| Zoobot ConvNeXT-Nano | Frozen head | 0.9638 | 0.1751 |

### Best model detailed evaluation (ImageNet full, AUC 0.9976)
To be run — confusion matrix and precision/recall pending.

### Key findings
1. Real data training dramatically reduces false positive rate: 65.7% → ~2.7% on test set
2. Full fine-tuning consistently outperforms frozen head across all architectures
3. Zoobot underperforms both ImageNet and scratch in both regimes — galaxy-pretrained features do not provide expected advantage for lens detection
4. Scratch frozen head (0.9865) beats ImageNet frozen head (0.9719) — random features + linear head is surprisingly competitive
5. All full fine-tuning runs showed severe overfitting after ~5-20 epochs due to small dataset (1,037 training images) — best checkpoints saved early
6. Sim-to-real gap confirmed and quantified: AUC 1.000 on synthetic → 0.96-0.99 on real data

### Overfitting analysis
Full fine-tuning on 1,037 images leads to rapid memorization:
- ImageNet full: best val loss at epoch 3, overfitting thereafter
- Zoobot full: best val loss at epoch 7, monotonically increasing thereafter
- Scratch full: unstable val loss throughout, high variance

This suggests dataset size is the primary bottleneck. With 5,000+ real lens examples, full fine-tuning would likely generalize significantly better.

### Saved models (Kaggle output)
- zoobot_real_trained.pth (frozen head)
- resnet18_imagenet_real.pth (frozen head)
- resnet18_scratch_real.pth (full model)
- resnet18_scratch_frozen.pth (frozen head)
- zoobot_full_real.pth (full model)
- resnet18_imagenet_full.pth (full model)

### Survey search status
- NOIRLab Astro Data Lab queries timing out on Kaggle (works on Colab only)
- Random grid cutouts invalid for inference — not galaxy-centered
- Test set (222 images) remains the honest evaluation
- Pending: fresh galaxy-targeted cutouts via NOIRLab on Colab for real survey search

### Next
- Download all 6 model checkpoints from Kaggle to Drive
- Run confusion matrix on best model (ImageNet full)
- Retry NOIRLab query on Colab tonight for survey search
- Begin Task 9 writeup


## 2026-04-03 (Task 8 continued — Hard Negative Mining + Survey Search)

### NOIRLab query fix
- Discovered that `brick_primary = 1` is required for fast queries — without it the database scans 3 billion rows and times out
- Successfully queried RA 100-150, Dec 0-30 for 1,000 fresh elliptical galaxies (never seen during training)
- Downloaded 956 DESI cutouts (101x101px, grz) to desi_fresh_cutouts/

### Initial inference on fresh galaxies (pre-hard-negatives)
| Model | p > 0.9 | p > 0.5 | p < 0.1 |
|-------|---------|---------|---------|
| ImageNet full | 312 | 573 | 149 |
| Zoobot frozen | 22 | 235 | 430 |

- Visual inspection of top candidates: all false positives
- Main contaminant: DESI pipeline image artifacts (rectangular border patterns)
- Cross-matched against lenscat: 0 known lenses in this sky region
- Confirmed: false positive rate still too high for reliable visual inspection

### Hard negative mining (Task 8.1 improvement)
- Took 312 highest-scoring false positives from ImageNet model as hard negatives
- Added to training set: 481 lenses + 1,000 non-lenses + 312 hard negatives = 1,793 total
- Retrained ImageNet full model on Kaggle T4 GPU, 100 epochs
- Best val loss: 0.1443 at epoch 3
- Test AUC: 0.9802 (slight drop from 0.9976 — expected, test set now harder)

### Inference after hard negative mining
| Model | p > 0.9 | p > 0.5 | p < 0.1 |
|-------|---------|---------|---------|
| ImageNet + hard negs | 4 | 26 | 826 |

- False positive rate dropped from 60% to 2.7% — dramatic improvement
- Only 4 candidates above 0.9 threshold
- Visual inspection of top 4: none show clear Einstein ring features
- Expected yield in 956 galaxies: ~0.1 real lenses — need larger search area

### Saved files
- resnet18_imagenet_hardneg.pth (Kaggle output + Drive)
- fresh_survey_results_imagenet.csv
- fresh_survey_results_zoobot.csv
- hard_negatives.csv (312 coordinates)

### Pending
- [ ] Retrain Zoobot with hard negatives (same 312 hard negatives, same procedure)
- [ ] Download 10,000 galaxy cutouts overnight on PC (large_survey_coords.csv saved to Drive)
- [ ] Run inference on 10,000 galaxies with hard-neg-trained model
- [ ] Visual inspection of top candidates from large survey
- [ ] Ensemble scoring across all models

### Next session
- Start Zoobot hard negative retraining on Kaggle
- Start overnight PC download
- When download completes: upload to Kaggle, run inference, inspect top candidates


## 2026-04-03 (Task 8 continued — Large Survey Search)

### Large survey setup
- Queried NOIRLab for 10,000 massive elliptical galaxies, RA 100-150, Dec 0-30
- Downloaded 9,999 DESI cutouts (101x101px, grz) to desi_large_survey/ on Drive
- Key fix: brick_primary = 1 required for fast NOIRLab queries

### Inference results on 9,999 galaxies

| Model | p > 0.9 | p > 0.5 | p < 0.1 |
|-------|---------|---------|---------|
| ImageNet + hard negatives | 82 | 406 | 7,993 |
| Zoobot frozen + hard negatives | 7 | 318 | 5,863 |
| Ensemble (both > 0.9) | 4 | — | — |

### Known lens recovery
- ImageNet model correctly identified DESI-113.7119+17.2044 (lenscat grade: probable)
- Separation: 0.10 arcsec — essentially exact match
- Model score: p=0.9105
- This validates the model is detecting real lensing signatures

### New ensemble candidates (both models p > 0.9, not in lenscat)
| RA | Dec | ImageNet | Zoobot | Ensemble |
|----|-----|----------|--------|----------|
| 115.7005 | 9.3649 | 0.9931 | 0.9618 | 0.9775 |
| 110.2707 | 23.2891 | 1.0000 | 0.9405 | 0.9702 |
| 115.8200 | 22.5305 | 1.0000 | 0.9203 | 0.9602 |
| 116.3199 | 8.8465 | 0.9565 | 0.9023 | 0.9294 |

### Visual inspection grades
- ra=115.7005, dec=9.3649 — Grade C. Pink smudge offset left, disappears in residual
- ra=110.2707, dec=23.2891 — Grade C. Faint smudge in cluster environment, residual clean
- ra=115.8200, dec=22.5305 — Grade D. No arc features, galaxy group
- ra=116.3199, dec=8.8465 — Grade C. Slight color asymmetry, no residual ring

### Summary
- No definitive new Einstein ring identified on visual inspection
- Known lens successfully recovered — confirms model is functional
- False positive rate at ensemble level: 0.04% (4/9,999) — near the expected real lens rate
- Main remaining challenge: visual confirmation at DESI ground-based resolution is difficult for subtle lenses

### Saved files
- large_survey_imagenet_hardneg.csv
- large_survey_zoobot_hardneg.csv  
- large_survey_ensemble.csv
- new_lens_candidates.csv

### Next
- Begin Task 9: writeup
- Consider submitting Grade C candidates to a citizen science platform for second opinions
- Future work: search larger sky area, add more hard negatives, higher resolution follow-up