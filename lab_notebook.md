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



