# ===============================
# process_train_dataset.py
# ===============================

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import shutil

# ===============================
# CONFIG
# ===============================

IMAGE_SIZE = 224
NUM_SAMPLES = 5000  # images to convert per class

BLUR_KERNEL = (5, 5)
NOISE_STD = 10

ATTACKS = [
    "jpeg",
    "resize",
    "crop",
    "blur",
    "noise",
    "sharpen"
]

# ===============================
# UTILS
# ===============================

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ===============================
# ATTACK FUNCTIONS (SAME AS TEST)
# ===============================

def jpeg(img):
    q = random.randint(20, 60)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def resize_attack(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.4, 0.8)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img = cv2.resize(img, (w, h))
    return img

def crop(img):
    h, w = img.shape[:2]
    ratio = random.uniform(0.6, 0.85)
    ch, cw = int(h * ratio), int(w * ratio)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    return cv2.resize(img[y:y+ch, x:x+cw], (w, h))

def blur(img):
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)

def noise(img):
    noise = np.random.normal(0, NOISE_STD, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# ===============================
# STRONG PROCESSING (IDENTICAL)
# ===============================

def apply_random_attacks(img):

    num_attacks = random.randint(2, 4)
    chosen = random.sample(ATTACKS, num_attacks)

    for attack in chosen:

        if attack == "jpeg":
            img = jpeg(img)

        elif attack == "resize":
            img = resize_attack(img)

        elif attack == "crop":
            img = crop(img)

        elif attack == "blur":
            img = blur(img)

        elif attack == "noise":
            img = noise(img)

        elif attack == "sharpen":
            img = sharpen(img)

    return img

# ===============================
# PROCESS + MOVE
# ===============================

def process_and_move(input_dir, clean_out, processed_out):

    all_paths = [
        input_dir / f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ]

    if len(all_paths) == 0:
        print(f"❌ No images in {input_dir}")
        return

    # 🔥 RANDOM SELECTION
    selected = random.sample(all_paths, min(NUM_SAMPLES, len(all_paths)))

    for p in tqdm(all_paths, desc=f"{input_dir.name}"):

        if p in selected:
            # 🔥 PROCESS + MOVE
            img = load_image(p)
            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = apply_random_attacks(img)

            save_image(img, processed_out / p.name)

        else:
            # 🔥 MOVE CLEAN IMAGE
            clean_out.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(clean_out / p.name))

    print(f"✔ Done {input_dir.name}")

# ===============================
# MAIN
# ===============================

def main():

    root = Path(__file__).parent

    input_base = root / "data" / "train"

    output_base = root / "data" / "train_split"

    clean_base = output_base / "clean"
    processed_base = output_base / "processed"

    for cls in ["real", "fake"]:

        in_dir = input_base / cls

        clean_out = clean_base / cls
        processed_out = processed_base / cls

        if not in_dir.exists():
            print(f"❌ Missing {in_dir}")
            continue

        process_and_move(in_dir, clean_out, processed_out)

    print("\n✅ TRAIN DATASET SPLIT CREATED SUCCESSFULLY")

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    main()