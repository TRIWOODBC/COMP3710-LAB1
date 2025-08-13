# cantor_layers_colored.py
# Classic layered rendering of the Cantor set (bars per iteration).
# Vectorised with PyTorch; each level gets a different color; saves an image.

import os
import time
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DEPTH = 6    # number of iterations the deepest level have 2^6 lines
LINEWIDTH = 12
CMAP_NAME = "tab20"
SAVE_PATH = "outputs/cantor_layers_colored.png"

init_starts = torch.tensor([0.0], dtype=torch.float32, device=device) # start points
init_lengths = torch.tensor([1.0], dtype=torch.float32, device=device) # initial lengths
intervals_per_level = []

# --------------------
# Build intervals (vectorised on PyTorch)
# --------------------
starts = init_starts
lengths = init_lengths
intervals_per_level.append((starts.clone(), lengths.clone()))

for d in range(1, DEPTH + 1):
    left_starts  = starts
    right_starts = starts + 2.0 * lengths / 3.0
    starts  = torch.cat([left_starts, right_starts], dim=0)
    lengths = torch.cat([lengths / 3.0, lengths / 3.0], dim=0)
    intervals_per_level.append((starts.clone(), lengths.clone()))
# --------------------
# Plot (each level in a different color)
# --------------------
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.figure(figsize=(8, 4))
plt.axis("off")

cmap = plt.get_cmap(CMAP_NAME)
n_levels = len(intervals_per_level)

for row, (starts_lvl, lengths_lvl) in enumerate(intervals_per_level):
    y_val = n_levels - 1 - row # reverse order: top level on top
    t = row / max(1, n_levels - 1)
    color = cmap(t)

    xs0 = starts_lvl #line left
    xs1 = starts_lvl + lengths_lvl #line right
    ys  = torch.full_like(starts_lvl, float(y_val))

    # 转为 numpy 供 matplotlib 使用
    plt.hlines(
        y=ys.detach().cpu().numpy(),
        xmin=xs0.detach().cpu().numpy(),
        xmax=xs1.detach().cpu().numpy(),
        linewidth=LINEWIDTH,
        color=color
    )

plt.xlim(-0.02, 1.02)
plt.ylim(-0.5, n_levels - 0.5)
plt.title("Cantor Set — layered construction (middle third removed each level)")

plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
print(f"Saved to: {SAVE_PATH}")
plt.show()


plt.xlim(-0.02, 1.02)
plt.ylim(-0.5, len(intervals_per_level) - 0.5)
plt.title("Cantor Set — layered construction (remove middle third each level)")
plt.show()
