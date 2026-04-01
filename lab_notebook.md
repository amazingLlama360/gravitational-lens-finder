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

