# ===============================
# densenet_classifier.py (FINAL COMPLETE)
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split


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

    return LL, LH, HL, HH


# ===================== DATASET =====================

class WaveletDataset(Dataset):

    def __init__(self, image_paths, labels, train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.train = train

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])

        # 🔥 Augmentation
        if self.train:
            if np.random.rand() < 0.3:
                img = cv2.GaussianBlur(img, (5,5), 0)
            if np.random.rand() < 0.3:
                img = cv2.resize(img, (112,112))
                img = cv2.resize(img, (224,224))

        LL, LH, HL, HH = haar_wavelet(img)

        stacked = np.stack([
            self.normalize(LL),
            self.normalize(LH),
            self.normalize(HL),
            self.normalize(HH)
        ], axis=0)

        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.image_paths)


# ===================== MODEL =====================

class DenseNetClassifier:

    def __init__(self, device='cuda'):

        self.device = device

        self.model = models.densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)

        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, 2)

        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, loader):

        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for x,y in tqdm(loader, desc="Training"):
            x,y = x.to(self.device), y.to(self.device)

            out = self.forward(x)
            loss = self.criterion(out,y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

        return total_loss/len(loader), 100*correct/total

    def evaluate(self, loader):

        self.model.eval()

        preds, labels, probs = [], [], []
        total_loss, correct, total = 0,0,0

        with torch.no_grad():
            for x,y in tqdm(loader, desc="Evaluating"):
                x,y = x.to(self.device), y.to(self.device)

                out = self.forward(x)
                loss = self.criterion(out,y)

                p = torch.softmax(out,1)[:,1]

                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
                probs.extend(p.cpu().numpy())

                total_loss += loss.item()
                correct += (out.argmax(1)==y).sum().item()
                total += y.size(0)

        preds = np.array(preds)
        labels = np.array(labels)
        probs = np.array(probs)

        acc = 100*np.mean(preds==labels)
        auc = roc_auc_score(labels, probs)*100
        ap = average_precision_score(labels, probs)*100

        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))] * 100

        val_loss = total_loss / len(loader)

        return acc, preds, labels, auc, ap, eer, val_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


# ===================== TRAIN =====================

def load_image_paths(data_path):

    real = [str((data_path/"real")/f) for f in os.listdir(data_path/"real")]
    fake = [str((data_path/"fake")/f) for f in os.listdir(data_path/"fake")]

    return real+fake, [0]*len(real)+[1]*len(fake)


def main():

    root = Path(__file__).parent
    data_path = root/"data"/"train"

    paths, labels = load_image_paths(data_path)

    train_p, test_p, train_l, test_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(WaveletDataset(train_p, train_l, True), batch_size=32, shuffle=True)
    test_loader = DataLoader(WaveletDataset(test_p, test_l, False), batch_size=32)

    model = DenseNetClassifier()

    best_auc = 0
    best_metrics = {}

    for epoch in range(10):

        logger.info(f"Epoch {epoch+1}")

        train_loss, train_acc = model.train_epoch(train_loader)
        acc, preds, labels, auc, ap, eer, val_loss = model.evaluate(test_loader)

        logger.info(f"Train: {train_acc:.2f} | Test: {acc:.2f} | AUC: {auc:.2f}")

        if auc > best_auc or epoch == 0:
            best_auc = auc

            best_metrics = {
                "epoch": epoch+1,
                "train_acc": train_acc,
                "test_acc": acc,
                "auc": auc,
                "ap": ap,
                "eer": eer,
                "val_loss": val_loss
            }

            model.save_model(root/"densenet_best.pth")

    # ================= SAVE REPORT =================

    results_file = root/"training_results.txt"

    with open(results_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("Enhanced DenseNet121 Deepfake Detection\n")
        f.write("="*70 + "\n\n")

        f.write(f"Best Epoch: {best_metrics['epoch']}\n")
        f.write(f"Train Accuracy: {best_metrics['train_acc']:.2f}%\n")
        f.write(f"Test Accuracy: {best_metrics['test_acc']:.2f}%\n")
        f.write(f"AUC: {best_metrics['auc']:.2f}%\n")
        f.write(f"AP: {best_metrics['ap']:.2f}%\n")
        f.write(f"EER: {best_metrics['eer']:.2f}%\n")
        f.write(f"Validation Loss: {best_metrics['val_loss']:.4f}\n")

        f.write("\nArchitecture:\n")
        f.write("- Haar Wavelet Input (4 channels)\n")
        f.write("- DenseNet121 Backbone\n")
        f.write("- Data Augmentation\n")

        f.write("\n" + "="*70 + "\n")

    logger.info(f"Training results saved to {results_file}")
    logger.info("Training complete")


if __name__ == "__main__":
    main()