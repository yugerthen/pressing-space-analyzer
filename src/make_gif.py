import imageio.v2 as imageio
import glob
from PIL import Image
import numpy as np

frame_paths = sorted(glob.glob("../data/gif_frames/frame_*.jpg"))[::2]

frames = []
for p in frame_paths:
    img = Image.open(p)
    w, h = img.size
    new_w = 640
    new_h = int(h * (new_w / w))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=96)
    frames.append(np.array(img.convert("RGB")))

imageio.mimsave("../data/demo.gif", frames, fps=6, loop=0)
print(f"GIF créé avec {len(frames)} frames")