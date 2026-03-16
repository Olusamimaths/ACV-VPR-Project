"""
Generate report figures from the GardensPoint dataset.
Run from the VPR_Tutorial root directory:
    python generate_figures.py
Outputs:
    images/dataset_samples.png   – 2-row grid of day/night pairs
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.load_dataset import GardensPointDataset

# ── Config ────────────────────────────────────────────────────────────────────
# Frame indices to display (0-based, out of 50 frames)
FRAME_INDICES = [9, 24, 39]   # ≈ frame 10, 25, 40

OUTPUT_DIR = 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
imgs_db, imgs_q, _, _ = GardensPointDataset().load()

# ── Figure: dataset_samples.png ───────────────────────────────────────────────
n_cols = len(FRAME_INDICES)
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.2, 4.2))
fig.subplots_adjust(hspace=0.05, wspace=0.04)

col_labels = [f'Frame {i + 1}' for i in FRAME_INDICES]

for col, idx in enumerate(FRAME_INDICES):
    # Day (top row)
    axes[0, col].imshow(imgs_db[idx])
    axes[0, col].axis('off')
    axes[0, col].set_title(col_labels[col], fontsize=9, pad=3)

    # Night (bottom row)
    axes[1, col].imshow(imgs_q[idx])
    axes[1, col].axis('off')

# Row labels on the left
axes[0, 0].annotate('Day', xy=(0, 0.5), xytext=(-8, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='center', fontsize=9, rotation=90)
axes[1, 0].annotate('Night', xy=(0, 0.5), xytext=(-8, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='center', fontsize=9, rotation=90)

out_path = os.path.join(OUTPUT_DIR, 'dataset_samples.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')
