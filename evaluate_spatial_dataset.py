# ===============================
# evaluate_spatial_dataset.py (FIXED)
# ===============================

import os
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_folder(folder_path):

    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(folder_path, transform=transform)

    if len(dataset) == 0:
        logger.error("Empty dataset!")
        return

    real_idx, fake_idx = [], []

    for i, (_, label) in enumerate(dataset.samples):
        if label==0 and len(real_idx)<1000:
            real_idx.append(i)
        elif label==1 and len(fake_idx)<1000:
            fake_idx.append(i)

        if len(real_idx)==1000 and len(fake_idx)==1000:
            break

    subset = Subset(dataset, real_idx+fake_idx)

    loader = DataLoader(subset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.densenet121()
    model.classifier = torch.nn.Linear(model.classifier.in_features,2)

    # 🔥 FIXED LOADING
    model.load_state_dict(torch.load("spatial_densenet_best.pth", map_location=device))

    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"Evaluating {folder_path.name}"):

            imgs = imgs.to(device)
            out = model(imgs)

            p = out.argmax(1)

            preds.extend(p.cpu().numpy())
            labels.extend(lbls.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    acc = 100*np.mean(preds==labels)

    real_mask = labels == 0
    fake_mask = labels == 1

    real_acc = 100*np.mean(preds[real_mask]==labels[real_mask])
    fake_acc = 100*np.mean(preds[fake_mask]==labels[fake_mask])

    logger.info(f"{folder_path.name} Accuracy: {acc:.2f}%")
    logger.info(f"Real Acc: {real_acc:.2f}% | Fake Acc: {fake_acc:.2f}%")

    # ================= SAVE RESULTS =================
    results_file = folder_path/"spatial_results.txt"

    with open(results_file, "w") as f:
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"Real Accuracy: {real_acc:.2f}%\n")
        f.write(f"Fake Accuracy: {fake_acc:.2f}%\n")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float)/cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues")
    plt.savefig(folder_path/"spatial_confusion_matrix.png")
    plt.close()


def main():

    root = Path(__file__).parent / "data" / "test"

    logger.info("Evaluating CLEAN (Spatial)...")
    evaluate_folder(root/"clean")

    logger.info("Evaluating PROCESSED (Spatial)...")
    evaluate_folder(root/"processed")


if __name__ == "__main__":
    main()