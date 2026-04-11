import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# CONFIG (optimized for your setup)
# ==========================================================

TRAIN_PER_CLASS = 10000
TEST_PER_CLASS = 5000

SEED = 42
random.seed(SEED)

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# ==========================================================
# UTILS
# ==========================================================

def get_images(folder):
    return [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXT]

def copy_images(files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc=f"Copying -> {dest_dir}"):
        shutil.copy2(f, dest_dir / f.name)

# ==========================================================
# MAIN
# ==========================================================

def main():

    project_root = Path(__file__).parent
    base_data = project_root / "data"

    real_dir = base_data / "real"
    fake_dir = base_data / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        print("❌ real/ or fake/ folder missing inside data/")
        return

    # ------------------------------------------------------
    # Load images
    # ------------------------------------------------------

    real_images = get_images(real_dir)
    fake_images = get_images(fake_dir)

    print(f"Found {len(real_images)} real images")
    print(f"Found {len(fake_images)} fake images")

    # Shuffle
    random.shuffle(real_images)
    random.shuffle(fake_images)

    # ------------------------------------------------------
    # Select balanced subset
    # ------------------------------------------------------

    total_needed = TRAIN_PER_CLASS + TEST_PER_CLASS

    real_selected = real_images[:total_needed]
    fake_selected = fake_images[:total_needed]

    # Split
    real_train = real_selected[:TRAIN_PER_CLASS]
    real_test  = real_selected[TRAIN_PER_CLASS:]

    fake_train = fake_selected[:TRAIN_PER_CLASS]
    fake_test  = fake_selected[TRAIN_PER_CLASS:]

    # ------------------------------------------------------
    # Create new structure
    # ------------------------------------------------------

    new_root = project_root / "data_new"

    train_real = new_root / "train" / "real"
    train_fake = new_root / "train" / "fake"

    test_real = new_root / "test" / "clean" / "real"
    test_fake = new_root / "test" / "clean" / "fake"

    print("\nCreating dataset structure...")

    # ------------------------------------------------------
    # Copy files
    # ------------------------------------------------------

    copy_images(real_train, train_real)
    copy_images(fake_train, train_fake)

    copy_images(real_test, test_real)
    copy_images(fake_test, test_fake)

    print("\n✅ Dataset prepared successfully!")

    print("\nFinal structure:")
    print(new_root)

    print("\nSummary:")
    print(f"Train -> {len(real_train)} real, {len(fake_train)} fake")
    print(f"Test  -> {len(real_test)} real, {len(fake_test)} fake")


if __name__ == "__main__":
    main()