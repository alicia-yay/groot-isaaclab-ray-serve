"""Polish g1_groot_n16_g1pnp.gif: trim black frames, crop, brighten, slow."""
import os, sys
import numpy as np
import imageio.v2 as imageio

src = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/g1_groot_n16_g1pnp.gif"
dst = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/g1_groot_n16_polished.gif"

print(f"Loading {src}...")
frames = list(imageio.mimread(src, memtest=False))
print(f"Loaded {len(frames)} frames, shape={frames[0].shape}")

# 1. Drop black/near-black frames
non_black = [f for f in frames if np.asarray(f).mean() > 20]
print(f"After dropping black: {len(non_black)} frames")

# 2. Crop to focus on robot + table (tighter framing)
def crop(f):
    arr = np.asarray(f)
    h, w = arr.shape[:2]
    # crop top 15%, bottom 25%, left 25%, right 25% (focus on center)
    return arr[int(h*0.15):int(h*0.75), int(w*0.20):int(w*0.80)]

cropped = [crop(f) for f in non_black]
print(f"After crop: shape={cropped[0].shape}")

# 3. Brighten (multiplicative gain on RGB)
def brighten(arr, gain=1.35):
    out = arr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)

bright = [brighten(f) for f in cropped]

# 4. Save at slower fps for clearer motion
imageio.mimsave(dst, bright, fps=8, loop=0)
size_mb = os.path.getsize(dst) / 1e6
print(f"\nSAVED: {dst} ({len(bright)} frames, {size_mb:.1f}MB)")
