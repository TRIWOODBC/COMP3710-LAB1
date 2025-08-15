import torch
import matplotlib.pyplot as plt
from matplotlib import cm

def cantor_mask(level: int, width: int, device: str = "cpu") -> torch.Tensor:
    base = 3 ** level if level > 0 else 1
    x = torch.arange(width, device=device) * base // width
    mask = torch.ones(width, dtype=torch.bool, device=device)
    t = x.clone()
    for _ in range(level):
        mask &= (t.remainder(3) != 1)  # keep 0 and 2
        t = torch.div(t, 3, rounding_mode='floor') # integer division
    return mask

def cantor_ladder(n_levels=6, width=1200, bar_thickness=10, row_gap=8, device="cpu"):
    height = (n_levels + 1) * (bar_thickness + row_gap) + row_gap
    img = torch.ones((height, width, 3), dtype=torch.float32, device=device)  # white bg

    # colors on device
    cmap = cm.get_cmap("tab10", n_levels + 1)
    colors = torch.tensor([cmap(i)[:3] for i in range(n_levels + 1)],
                          dtype=torch.float32, device=device)

    y = row_gap
    for level in range(n_levels + 1):
        mask1d = cantor_mask(level, width, device=device)  # (W,)
        mask3d = mask1d.view(1, -1, 1).expand(bar_thickness, width, 3)
        color  = colors[level].view(1, 1, 3).expand(bar_thickness, width, 3)
        band = img[y:y+bar_thickness, :, :]
        img[y:y+bar_thickness, :, :] = torch.where(mask3d, color, band)
        y += bar_thickness + row_gap
    return img

# ---- run on GPU if available ----
device = "cuda" if torch.cuda.is_available() else "cpu"
img = cantor_ladder(n_levels=6, width=1400, bar_thickness=12, row_gap=10, device=device)

plt.figure(figsize=(10, 3))
plt.imshow(img.detach().cpu().numpy(), interpolation="nearest")
plt.axis("off")
plt.show()

