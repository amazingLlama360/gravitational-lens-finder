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