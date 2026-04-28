"""
Aggressive post-processing for the demo GIF.

Inputs:  g1_groot_n16_g1pnp.gif (or g1_groot_n16_polished.gif)
Outputs: g1_groot_n16_aggressive.gif (single, polished)

What it does:
  1. Drop near-black frames (Isaac Sim's first render frame is often black)
  2. Drop near-static segments (no perceptible motion - boring)
  3. Tight crop on robot+table (remove empty floor + sky)
  4. Brighten + slight gamma correction
  5. Overlay a clean text banner at the top with title
  6. Slow down to 6 fps for clearer motion
  7. Pad final frame for 0.5s "rest" so loop reads as a single demo

No external network or compute, just numpy + Pillow + imageio.
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.expanduser("~/default_cld_g54aiirwj1s8t9ktgzikqur41k")
SRC = os.path.join(ROOT, "g1_groot_n16_g1pnp.gif")
DST = os.path.join(ROOT, "g1_groot_n16_aggressive.gif")

TITLE = "GR00T-N1.6-G1 fine-tune  +  Isaac Lab  +  Ray Serve on Anyscale"
SUBTITLE = "Unitree G1 pick-place, zero-shot inference"

print(f"Loading {SRC}...")
frames = list(imageio.mimread(SRC, memtest=False))
print(f"Loaded {len(frames)} frames, shape={frames[0].shape}")

# 1. Drop near-black frames
def is_dark(arr, thr=20):
    return np.asarray(arr).mean() < thr

bright_frames = [f for f in frames if not is_dark(f)]
print(f"After dropping near-black: {len(bright_frames)} frames")

# 2. Drop near-static frames (consecutive frames with very small diff)
def frame_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())

motion_frames = [bright_frames[0]]
for f in bright_frames[1:]:
    if frame_diff(f, motion_frames[-1]) > 1.5:  # threshold tunable
        motion_frames.append(f)
print(f"After dropping static: {len(motion_frames)} frames")

# 3. Tight crop on robot + table region
def crop(arr):
    a = np.asarray(arr)
    h, w = a.shape[:2]
    # Empirically robot+table is roughly center-vertical, slightly above-mid horizontal
    y0 = int(h * 0.18)
    y1 = int(h * 0.78)
    x0 = int(w * 0.22)
    x1 = int(w * 0.78)
    return a[y0:y1, x0:x1]

cropped = [crop(f) for f in motion_frames]
print(f"After crop: shape={cropped[0].shape}")

# 4. Brighten with gamma
def brighten(arr, gain=1.30, gamma=0.92):
    a = arr.astype(np.float32) / 255.0
    a = np.power(a, gamma)  # gamma < 1 brightens midtones
    a = np.clip(a * gain, 0, 1.0)
    return (a * 255).astype(np.uint8)

bright = [brighten(f) for f in cropped]

# 5. Overlay title + subtitle banner
def add_banner(arr_np):
    img = Image.fromarray(arr_np).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = arr_np.shape[:2]
    
    # Try to use a real font, fall back to default
    font_title = font_sub = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            font_title = ImageFont.truetype(fp, max(14, w // 32))
            font_sub = ImageFont.truetype(fp.replace("-Bold", ""), max(11, w // 44))
            break
        except Exception:
            continue
    if font_title is None:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    # Banner background (semi-transparent black)
    banner_h = max(50, h // 8)
    draw.rectangle([(0, 0), (w, banner_h)], fill=(0, 0, 0, 180))
    
    # Title
    pad = 12
    draw.text((pad, pad - 2), TITLE, font=font_title, fill=(255, 255, 255, 255))
    draw.text((pad, pad + max(18, w // 32)), SUBTITLE, font=font_sub, fill=(200, 220, 255, 255))
    
    return np.array(img)

with_banner = [add_banner(f) for f in bright]
print(f"Banner added")

# 6. Save at slower fps
# 7. Pad with last frame for end-rest
last = with_banner[-1]
final_frames = with_banner + [last] * 6   # ~1 sec rest at 6 fps

imageio.mimsave(DST, final_frames, fps=6, loop=0)
size_mb = os.path.getsize(DST) / 1e6
print(f"\nSAVED: {DST}")
print(f"  {len(final_frames)} frames, {size_mb:.1f}MB at 6 fps")
