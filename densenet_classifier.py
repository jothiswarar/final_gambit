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

    def frequency_perturb(self, LH, HL, HH):
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.02, LH.shape)
            LH += noise
            HL += noise
            HH += noise

        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            LH *= scale
            HL *= scale
            HH *= scale

        return LH, HL, HH

    def frequency_threshold(self, LH, HL, HH):
        threshold = 0.05
        LH = np.where(np.abs(LH) < threshold, 0, LH)
        HL = np.where(np.abs(HL) < threshold, 0, HL)
        HH = np.where(np.abs(HH) < threshold, 0, HH)
        return LH, HL, HH

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])

        if self.train:
            if np.random.rand() < 0.3:
                img = cv2.GaussianBlur(img, (5,5), 0)
            if np.random.rand() < 0.3:
                img = cv2.resize(img, (112,112))
                img = cv2.resize(img, (224,224))

        LL, LH, HL, HH = haar_wavelet(img)

        if self.train:
            LH, HL, HH = self.frequency_perturb(LH, HL, HH)

        LH, HL, HH = self.frequency_threshold(LH, HL, HH)

        stacked = np.stack([
            self.normalize(LL),
            self.normalize(LH),
            self.normalize(HL),
            self.normalize(HH)
        ], axis=0)

        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.image_paths)


# ===================== SE BLOCK =====================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ===================== MODEL =====================

class DenseNetClassifier:

    def __init__(self, class_weights=None, device='cuda'):

        self.device = device

        base = models.densenet121(weights=None)
        base.features.conv0 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)

        self.features = base.features
        self.se = SEBlock(1024)

        self.adapter = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 2)

        self.to(device)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def parameters(self):
        return list(self.features.parameters()) + \
               list(self.adapter.parameters()) + \
               list(self.se.parameters()) + \
               list(self.classifier.parameters())

    def to(self, device):
        self.features.to(device)
        self.adapter.to(device)
        self.se.to(device)
        self.classifier.to(device)
        self.dropout.to(device)

    def forward(self, x):
        x = self.adapter(x)
        x = self.features(x)
        x = self.se(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def train_epoch(self, loader):

        self.features.train()
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

        self.features.eval()

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

    # ✅ FIXED SAVE FUNCTION
    def save_model(self, path):
        torch.save({
            "features": self.features.state_dict(),
            "adapter": self.adapter.state_dict(),
            "se": self.se.state_dict(),
            "classifier": self.classifier.state_dict()
        }, path)

    # ✅ LOAD FUNCTION
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.features.load_state_dict(checkpoint["features"])
        self.adapter.load_state_dict(checkpoint["adapter"])
        self.se.load_state_dict(checkpoint["se"])
        self.classifier.load_state_dict(checkpoint["classifier"])


# ===================== TRAIN =====================

def load_image_paths(data_path):

    all_paths = []
    all_labels = []

    for domain in ["clean", "processed"]:
        for cls in ["real", "fake"]:

            class_dir = data_path / domain / cls

            if not class_dir.exists():
                continue

            files = [str(class_dir / f) for f in os.listdir(class_dir)]

            all_paths.extend(files)

            if cls == "real":
                all_labels.extend([0] * len(files))
            else:
                all_labels.extend([1] * len(files))

    return all_paths, all_labels

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def main():

    root = Path(__file__).parent
    data_path = root/"data"/"train"

    paths, labels = load_image_paths(data_path)
    class_weights = compute_class_weights(labels)

    train_p, test_p, train_l, test_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(WaveletDataset(train_p, train_l, True), batch_size=32, shuffle=True)
    test_loader = DataLoader(WaveletDataset(test_p, test_l, False), batch_size=32)

    model = DenseNetClassifier(class_weights=class_weights)

    best_auc = 0

    for epoch in range(10):

        logger.info(f"Epoch {epoch+1}")

        train_loss, train_acc = model.train_epoch(train_loader)
        acc, preds, labels, auc, ap, eer, val_loss = model.evaluate(test_loader)

        logger.info(f"Train: {train_acc:.2f} | Test: {acc:.2f} | AUC: {auc:.2f}")

        if auc > best_auc:
            best_auc = auc
            model.save_model(root/"densenet_best.pth")

    logger.info("Training complete")


if __name__ == "__main__":
    main()