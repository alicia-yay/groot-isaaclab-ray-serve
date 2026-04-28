"""
Side-by-side comparison GIF: N1.7 zero-shot (left) vs N1.6 fine-tune (right).

Both source GIFs already exist. We:
  1. Load both
  2. Trim near-black frames from both
  3. Resize to same height
  4. Concatenate horizontally
  5. Add a comparison banner: "Zero-shot baseline | Task fine-tune"
  6. Loop the shorter one to match the longer
  7. Save at 8 fps
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.expanduser("~/default_cld_g54aiirwj1s8t9ktgzikqur41k")
LEFT_SRC = os.path.join(ROOT, "g1_groot_n17_zeroshot.gif")
RIGHT_SRC = os.path.join(ROOT, "g1_groot_n16_g1pnp.gif")
DST = os.path.join(ROOT, "g1_groot_comparison.gif")

LEFT_LABEL = "GR00T-N1.7-3B base  (REAL_G1)"
RIGHT_LABEL = "GR00T-N1.6-G1 fine-tune  (UNITREE_G1)"
HEADER = "Same Isaac Lab pick-place task, two models, both via Ray Serve on Anyscale"

print(f"Loading {LEFT_SRC} ...")
left_frames = list(imageio.mimread(LEFT_SRC, memtest=False))
print(f"Loaded {len(left_frames)} frames, shape={left_frames[0].shape}")

print(f"Loading {RIGHT_SRC} ...")
right_frames = list(imageio.mimread(RIGHT_SRC, memtest=False))
print(f"Loaded {len(right_frames)} frames, shape={right_frames[0].shape}")

# Drop near-black frames
def trim_black(frames, thr=20):
    return [f for f in frames if np.asarray(f).mean() > thr]

left_frames = trim_black(left_frames)
right_frames = trim_black(right_frames)
print(f"After trim: left={len(left_frames)}, right={len(right_frames)}")

# Crop both to focus on robot
def crop(arr):
    a = np.asarray(arr)
    h, w = a.shape[:2]
    return a[int(h*0.18):int(h*0.78), int(w*0.22):int(w*0.78)]

left_frames = [crop(f) for f in left_frames]
right_frames = [crop(f) for f in right_frames]

# Resize both panes to same height
TARGET_H = 360  # pane height

def resize_to_h(arr, target_h):
    img = Image.fromarray(arr).convert("RGB")
    h, w = arr.shape[:2]
    new_w = int(w * target_h / h)
    img = img.resize((new_w, target_h), Image.BILINEAR)
    return np.array(img)

left_frames = [resize_to_h(f, TARGET_H) for f in left_frames]
right_frames = [resize_to_h(f, TARGET_H) for f in right_frames]
print(f"After resize: pane shape L={left_frames[0].shape} R={right_frames[0].shape}")

# Brighten both
def brighten(arr, gain=1.25, gamma=0.95):
    a = arr.astype(np.float32) / 255.0
    a = np.power(a, gamma)
    a = np.clip(a * gain, 0, 1.0)
    return (a * 255).astype(np.uint8)

left_frames = [brighten(f) for f in left_frames]
right_frames = [brighten(f) for f in right_frames]

# Pad the shorter one (loop it) so they align
N = max(len(left_frames), len(right_frames))
def pad_loop(frames, n):
    if len(frames) >= n:
        return frames[:n]
    out = []
    while len(out) < n:
        out += frames
    return out[:n]

left_frames = pad_loop(left_frames, N)
right_frames = pad_loop(right_frames, N)

# Compose: header banner + 2 panes + label strip
def font(size):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def compose(lf, rf, header, l_label, r_label):
    pane_h, pane_w = lf.shape[:2]
    assert rf.shape[0] == pane_h
    rpane_w = rf.shape[1]
    
    HEADER_H = 56
    LABEL_H = 38
    GAP = 4
    
    total_w = pane_w + GAP + rpane_w
    total_h = HEADER_H + pane_h + LABEL_H
    
    canvas = Image.new("RGB", (total_w, total_h), (15, 15, 22))
    
    # Header
    draw = ImageDraw.Draw(canvas)
    f_header = font(18)
    bbox = draw.textbbox((0, 0), header, font=f_header)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, (HEADER_H - (bbox[3] - bbox[1])) // 2 - 2),
              header, font=f_header, fill=(220, 230, 240))
    
    # Left pane
    canvas.paste(Image.fromarray(lf), (0, HEADER_H))
    # Right pane
    canvas.paste(Image.fromarray(rf), (pane_w + GAP, HEADER_H))
    
    # Labels
    f_label = font(15)
    draw = ImageDraw.Draw(canvas)
    
    bbox = draw.textbbox((0, 0), l_label, font=f_label)
    tw = bbox[2] - bbox[0]
    draw.text(((pane_w - tw) // 2, HEADER_H + pane_h + 8),
              l_label, font=f_label, fill=(180, 200, 220))
    
    bbox = draw.textbbox((0, 0), r_label, font=f_label)
    tw = bbox[2] - bbox[0]
    draw.text((pane_w + GAP + (rpane_w - tw) // 2, HEADER_H + pane_h + 8),
              r_label, font=f_label, fill=(180, 220, 200))
    
    return np.array(canvas)

print("Composing frames...")
composed = [compose(l, r, HEADER, LEFT_LABEL, RIGHT_LABEL)
            for l, r in zip(left_frames, right_frames)]

# Save
imageio.mimsave(DST, composed, fps=8, loop=0)
size_mb = os.path.getsize(DST) / 1e6
print(f"\nSAVED: {DST}")
print(f"  {len(composed)} frames, {size_mb:.1f}MB at 8 fps")
print(f"  pane size: {composed[0].shape}")
