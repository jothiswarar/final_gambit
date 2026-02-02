import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
INPUT_DIR = "Dataset\Real"
OUTPUT_DIR = "processed_dataset_Real"
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

random.seed(42)
np.random.seed(42)

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
# ---------------------------------------- #

# ---------- UTILS ---------- #
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ---------- ATTACK FUNCTIONS ---------- #
def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def resize_attack(img):
    h, w, _ = img.shape
    down = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

def crop_attack(img):
    h, w, _ = img.shape
    ch, cw = int(h * CROP_RATIO), int(w * CROP_RATIO)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    crop = img[y:y + ch, x:x + cw]
    return cv2.resize(crop, (w, h))

def blur_attack(img):
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)

def noise_attack(img):
    noise = np.random.normal(0, NOISE_STD, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ---------- MAIN PIPELINE ---------- #
def main():
    image_paths = [
        Path(INPUT_DIR) / f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    random.shuffle(image_paths)

    total_images = len(image_paths)
    split_size = total_images // len(ATTACKS)

    print(f"Total images      : {total_images}")
    print(f"Images per attack : {split_size}")

    for i, attack in enumerate(ATTACKS):
        subset = image_paths[i * split_size:(i + 1) * split_size]
        print(f"\nApplying {attack.upper()} to {len(subset)} images")

        for img_path in tqdm(subset):
            img = load_image(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            if attack in JPEG_QUALITIES:
                processed = jpeg_compress(img, JPEG_QUALITIES[attack])
            elif attack == "resize":
                processed = resize_attack(img)
            elif attack == "crop":
                processed = crop_attack(img)
            elif attack == "blur":
                processed = blur_attack(img)
            elif attack == "noise":
                processed = noise_attack(img)
            else:
                processed = img

            out_path = Path(OUTPUT_DIR) / img_path.name
            save_image(processed, out_path)

    print("\n✅ Processing complete. Flat dataset preserved.")

if __name__ == "__main__":
    main()