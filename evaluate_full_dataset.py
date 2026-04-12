# ===============================
# evaluate_full_dataset.py (FINAL FIXED VERSION)
# ===============================

import os
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import cv2

logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from densenet_classifier import DenseNetClassifier


# ===================== WAVELET =====================

def haar_wavelet(image):
    image = cv2.resize(image, (224,224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    LL = (gray[0::2,0::2] + gray[1::2,0::2] +
          gray[0::2,1::2] + gray[1::2,1::2]) / 4

    LH = (gray[0::2,0::2] + gray[1::2,0::2] -
          gray[0::2,1::2] - gray[1::2,1::2]) / 4

    HL = (gray[0::2,0::2] - gray[1::2,0::2] +
          gray[0::2,1::2] - gray[1::2,1::2]) / 4

    HH = (gray[0::2,0::2] - gray[1::2,0::2] -
          gray[0::2,1::2] + gray[1::2,1::2]) / 4

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return np.stack([norm(LL), norm(LH), norm(HL), norm(HH)], axis=0)


# ===================== DATASET =====================

class TestDataset(Dataset):

    def __init__(self, root):
        self.samples = []

        for cls in ["real","fake"]:
            for f in os.listdir(Path(root)/cls):
                self.samples.append((Path(root)/cls/f, 0 if cls=="real" else 1))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        return torch.tensor(haar_wavelet(img), dtype=torch.float32), label

    def __len__(self):
        return len(self.samples)


# ===================== EVALUATION =====================

def evaluate_folder(folder):

    loader = DataLoader(TestDataset(folder), batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ FIXED: correct argument passing
    model = DenseNetClassifier(device=device)

    # ✅ FIXED: correct loading
    model.load_model("densenet_best.pth")

    # ✅ FIXED: set eval mode properly
    model.features.eval()
    model.adapter.eval()
    model.se.eval()
    model.classifier.eval()

    acc, preds, labels, auc, ap, eer, loss = model.evaluate(loader)

    logger.info(f"{folder.name} ACC: {acc:.2f} | AUC: {auc:.2f} | EER: {eer:.2f}")

    # ================= SAVE METRICS =================

    results_file = folder/"evaluation_results.txt"

    with open(results_file, "w") as f:
        f.write("="*70 + "\n")
        f.write(f"{folder.name.upper()} RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"AUC: {auc:.2f}%\n")
        f.write(f"AP: {ap:.2f}%\n")
        f.write(f"EER: {eer:.2f}%\n")
        f.write(f"Loss: {loss:.4f}\n")

    logger.info(f"Results saved to {results_file}")

    # ================= CONFUSION MATRIX =================

    cm = confusion_matrix(labels, preds)

    sns.heatmap(cm/np.sum(cm,axis=1,keepdims=True), annot=True, fmt=".2%")
    plt.title(f"{folder.name} Confusion Matrix")
    plt.savefig(folder/"confusion.png")
    plt.close()


# ===================== MAIN =====================

def main():

    root = Path("data/test")

    logger.info("Evaluating CLEAN dataset...")
    evaluate_folder(root/"clean")

    logger.info("Evaluating PROCESSED dataset...")
    evaluate_folder(root/"processed")


if __name__ == "__main__":
    main()