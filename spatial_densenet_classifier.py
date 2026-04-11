import os
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve

# ===============================
# SETUP
# ===============================

logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)


# ===============================
# MODEL CLASS
# ===============================

class SpatialDenseNet:

    def __init__(self, device="cuda"):

        self.device = device

        self.model = models.densenet121(weights="IMAGENET1K_V1")

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, 2)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=2, factor=0.5
        )

    def train_epoch(self, loader):

        self.model.train()

        loss_sum, correct, total = 0, 0, 0

        for x, y in tqdm(loader, desc="Training"):

            x, y = x.to(self.device), y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

        return loss_sum/len(loader), 100*correct/total

    def evaluate(self, loader):

        self.model.eval()

        probs, labels = [], []
        correct, total = 0, 0

        with torch.no_grad():

            for x,y in tqdm(loader, desc="Evaluating"):

                x,y = x.to(self.device), y.to(self.device)

                out = self.model(x)

                p = torch.softmax(out,1)[:,1]

                probs.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())

                correct += (out.argmax(1)==y).sum().item()
                total += y.size(0)

        acc = 100*correct/total
        auc = roc_auc_score(labels, probs)*100
        ap = average_precision_score(labels, probs)*100

        fpr,tpr,_ = roc_curve(labels, probs)
        fnr = 1-tpr
        eer = fpr[np.nanargmin(np.abs(fnr-fpr))]*100

        return acc, auc, ap, eer

    def save(self, path):
        torch.save(self.model.state_dict(), path)


# ===============================
# MAIN TRAINING
# ===============================

def main():

    root = Path(__file__).parent
    data_path = root/"data"/"train"

    if not data_path.exists():
        logger.error("Training dataset not found!")
        return

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)

    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return

    # ===============================
    # LIMIT DATASET
    # ===============================

    real_idx, fake_idx = [], []

    for i,(_,label) in enumerate(dataset.samples):

        if label==0 and len(real_idx)<10000:
            real_idx.append(i)
        elif label==1 and len(fake_idx)<10000:
            fake_idx.append(i)

        if len(real_idx)==10000 and len(fake_idx)==10000:
            break

    subset = Subset(dataset, real_idx+fake_idx)

    # ===============================
    # TRAIN / VAL SPLIT
    # ===============================

    train_size = int(0.8*len(subset))
    val_size = len(subset)-train_size

    train_set, val_set = torch.utils.data.random_split(
        subset, [train_size,val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpatialDenseNet(device)

    best_auc = 0
    best_metrics = {}

    # ===============================
    # TRAIN LOOP
    # ===============================

    for epoch in range(20):

        logger.info(f"\nEpoch {epoch+1}")

        train_loss, train_acc = model.train_epoch(train_loader)
        val_acc, auc, ap, eer = model.evaluate(val_loader)

        logger.info(f"Train: {train_acc:.2f} | Val: {val_acc:.2f} | AUC: {auc:.2f}")

        model.scheduler.step(auc)

        # 🔥 ALWAYS SAVE FIRST EPOCH + IMPROVEMENTS
        if auc > best_auc or epoch == 0:
            best_auc = auc

            best_metrics = {
                "epoch": epoch+1,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "auc": auc,
                "ap": ap,
                "eer": eer
            }

            model.save(root/"spatial_densenet_best.pth")

    # ===============================
    # SAVE RESULTS
    # ===============================

    results_file = root/"spatial_training_results.txt"

    with open(results_file, "w") as f:

        f.write("="*60 + "\n")
        f.write("Spatial DenseNet Training Results\n")
        f.write("="*60 + "\n\n")

        for k,v in best_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n" + "="*60 + "\n")

    logger.info(f"Results saved to {results_file}")
    logger.info("Training complete")


if __name__ == "__main__":
    main()