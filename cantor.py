# cantor_layers.py
# Classic layered rendering of the Cantor set (bars per iteration).
# Vectorised with PyTorch; saves an image that matches the textbook look.

import os
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DEPTH = 6  
init_starts = torch.tensor([0.0], dtype=torch.float32, device=device)
init_lengths = torch.tensor([1.0], dtype=torch.float32, device=device)


intervals_per_level = []

starts = init_starts
lengths = init_lengths
intervals_per_level.append((starts.clone(), lengths.clone()))

for d in range(1, DEPTH + 1):
    left_starts = starts
    right_starts = starts + 2.0 * lengths / 3.0
    new_starts = torch.cat([left_starts, right_starts], dim=0)
    new_lengths = torch.cat([lengths / 3.0, lengths / 3.0], dim=0)

    starts, lengths = new_starts, new_lengths
    intervals_per_level.append((starts.clone(), lengths.clone()))


os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.axis('off')

for row, (starts, lengths) in enumerate(intervals_per_level):
    y = len(intervals_per_level) - 1 - row  
    for i in range(starts.numel()):
        x0 = starts[i].item()
        x1 = (starts[i] + lengths[i]).item()
        plt.hlines(y=y, xmin=x0, xmax=x1, linewidth=12)  

plt.xlim(-0.02, 1.02)
plt.ylim(-0.5, len(intervals_per_level) - 0.5)
plt.title("Cantor Set — layered construction (remove middle third each level)")
plt.show()