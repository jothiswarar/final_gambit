import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# CONFIG
IMAGE_SIZE = 224

JPEG_QUALITIES = {
    "jpeg": 75,
    "jpeg_q30": 30,
    "jpeg_q50": 50,
    "jpeg_q70": 70
}

BLUR_KERNEL = (5, 5)
NOISE_STD = 10
CROP_RATIO = 0.8

ATTACKS = [
    "jpeg",
    "jpeg_q30",
    "jpeg_q50",
    "jpeg_q70",
    "resize",
    "crop",
    "blur",
    "noise"
]

# UTILS
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ATTACKS
def jpeg(img, q):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def resize(img):
    h, w = img.shape[:2]
    return cv2.resize(cv2.resize(img, (w//2, h//2)), (w, h))

def crop(img):
    h, w = img.shape[:2]
    ch, cw = int(h*0.8), int(w*0.8)
    y = random.randint(0, h-ch)
    x = random.randint(0, w-cw)
    return cv2.resize(img[y:y+ch, x:x+cw], (w, h))

def blur(img):
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)

def noise(img):
    n = np.random.normal(0, NOISE_STD, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)

# PROCESS
def process_class(in_dir, out_dir):

    paths = [in_dir/f for f in os.listdir(in_dir)
             if f.lower().endswith((".jpg",".png",".jpeg",".bmp"))]

    random.shuffle(paths)

    split = len(paths)//len(ATTACKS)

    for i, attack in enumerate(ATTACKS):

        subset = paths[i*split:] if i==len(ATTACKS)-1 else paths[i*split:(i+1)*split]

        for p in tqdm(subset, desc=f"{in_dir.name}-{attack}"):

            img = load_image(p)
            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            if attack in JPEG_QUALITIES:
                img = jpeg(img, JPEG_QUALITIES[attack])
            elif attack=="resize":
                img = resize(img)
            elif attack=="crop":
                img = crop(img)
            elif attack=="blur":
                img = blur(img)
            elif attack=="noise":
                img = noise(img)

            save_image(img, out_dir/p.name)

# MAIN
def main():

    root = Path(__file__).parent

    input_base = root/"data"/"test"/"clean"
    output_base = root/"data"/"test"/"processed"

    for cls in ["real","fake"]:

        in_dir = input_base/cls
        out_dir = output_base/cls

        if in_dir.exists():
            process_class(in_dir, out_dir)

    print("Done")

if __name__ == "__main__":
    main()