# binarize_mask.py
from PIL import Image
import numpy as np

src = "door_knob_left_2001.jpg"   # image with white object on black
dst = "door_knob_left_2001.png"

img = Image.open(src).convert("L")
m = (np.array(img) > 127).astype(np.uint8) * 255   # strictly 0/255
Image.fromarray(m).save(dst)
print("Saved:", dst)
