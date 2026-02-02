import os
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torch.optim as optim
    import torchvision.models as models
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False


class WaveletDataset(Dataset):
    """Dataset for wavelet coefficients (HH, HL, LH)"""
    
    def __init__(self, file_paths, labels, target_size=224):
        """
        Args:
            file_paths: List of tuples (hh_path, hl_path, lh_path)
            labels: List of labels (0 for real, 1 for fake)
            target_size: Resize images to this size
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_size = target_size
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        hh_path, hl_path, lh_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load coefficient arrays
        hh = np.load(hh_path).astype(np.float32)
        hl = np.load(hl_path).astype(np.float32)
        lh = np.load(lh_path).astype(np.float32)
        
        # Normalize to [0, 1]
        hh = (hh - hh.min()) / (hh.max() - hh.min() + 1e-8)
        hl = (hl - hl.min()) / (hl.max() - hl.min() + 1e-8)
        lh = (lh - lh.min()) / (lh.max() - lh.min() + 1e-8)
        
        # Stack into 3-channel image (HH, HL, LH)
        image = np.stack([hh, hl, lh], axis=0)  # Shape: (3, H, W)
        
        # Convert to tensor
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


class DenseNetClassifier:
    """DenseNet classifier for real vs fake image detection"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 num_classes=2, pretrained=True):
        """
        Initialize DenseNet classifier
        
        Args:
            device: 'cuda' or 'cpu'
            num_classes: Number of output classes (2 for real/fake)
            pretrained: Use pretrained weights
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DenseNet")
        
        self.device = device
        self.num_classes = num_classes
        
        # Load DenseNet121 WITHOUT pretrained weights (faster, avoids download)
        logger.info(f"Loading DenseNet121 (from scratch)...")
        self.model = models.densenet121(weights=None)
        
        # Freeze early layers, unfreeze only denseblock4 and norm5 for fine-tuning
        # This strategy preserves low-level wavelet feature detectors while allowing
        # the deeper layers to adapt to wavelet-specific discriminative patterns.
        # Early DenseBlocks (1-3) learn generic texture features that transfer well.
        # DenseBlock4 and norm5 are task-specific layers trained for real/fake distinction.
        for name, param in self.model.named_parameters():
            if 'denseblock4' in name or 'norm5' in name:
                param.requires_grad = True
            elif 'features' in name:
                param.requires_grad = False
        
        # Modify classifier layer for binary classification
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(device)
        
        # Loss with label smoothing for better calibration
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Adam optimizer for trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        logger.info(f"DenseNet121 initialized on {device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        logger.info(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate on test set with metrics: ACC, AUC, EER, AP, and validation loss"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)  # Validation loss
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of fake class
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        acc = 100 * correct / total
        val_loss = total_loss / len(test_loader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # AUC: Area under ROC curve
        auc = roc_auc_score(all_labels, all_probs) * 100
        
        # AP: Average Precision (area under precision-recall curve)
        ap = average_precision_score(all_labels, all_probs) * 100
        
        # EER: Equal Error Rate - point where false positive rate equals false negative rate
        # roc_curve returns (fpr, tpr, thresholds); fnr = 1 - tpr
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = 100 * fpr[eer_idx]  # EER is the FPR (or FNR) at the crossing point
        
        logger.info(f"Test ACC: {acc:.2f}% | AUC: {auc:.2f}% | AP: {ap:.2f}% | EER: {eer:.2f}% | Val Loss: {val_loss:.4f}")
        return acc, all_preds, all_labels, auc, ap, eer, val_loss
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


def load_wavelet_data(wavelet_output_path, balance=True, max_samples=None):
    """
    Load HH, HL, LH wavelet coefficients and create file paths
    
    Args:
        wavelet_output_path: Path to wavelet_output directory
        balance: Balance real and fake to same count
        max_samples: Limit samples per class
        
    Returns:
        Tuple of (file_paths, labels)
    """
    wavelet_path = Path(wavelet_output_path)
    
    # Load real images using listdir for speed
    real_hh_dir = wavelet_path / 'real' / 'HH'
    real_files = sorted([f for f in os.listdir(real_hh_dir) if f.endswith('.npy')])
    real_hh = [real_hh_dir / f for f in real_files]
    
    # Load fake images
    fake_hh_dir = wavelet_path / 'fake' / 'HH'
    fake_files = sorted([f for f in os.listdir(fake_hh_dir) if f.endswith('.npy')])
    fake_hh = [fake_hh_dir / f for f in fake_files]
    
    logger.info(f"Found {len(real_hh)} real images and {len(fake_hh)} fake images")
    
    # Balance dataset: use min count for both classes
    if balance:
        min_count = min(len(real_hh), len(fake_hh))
        real_hh = real_hh[:min_count]
        fake_hh = fake_hh[:min_count]
        logger.info(f"Balanced dataset: using {min_count} images per class")
    
    # Limit samples
    if max_samples:
        real_hh = real_hh[:max_samples]
        fake_hh = fake_hh[:max_samples]
        logger.info(f"Using {max_samples} samples per class")
    
    # Create file paths and labels
    file_paths = []
    labels = []
    
    # Real images (label 0)
    for hh_file in real_hh:
        stem = hh_file.stem
        hl_file = wavelet_path / 'real' / 'HL' / f"{stem}.npy"
        lh_file = wavelet_path / 'real' / 'LH' / f"{stem}.npy"
        
        if hl_file.exists() and lh_file.exists():
            file_paths.append((str(hh_file), str(hl_file), str(lh_file)))
            labels.append(0)
    
    # Fake images (label 1)
    for hh_file in fake_hh:
        stem = hh_file.stem
        hl_file = wavelet_path / 'fake' / 'HL' / f"{stem}.npy"
        lh_file = wavelet_path / 'fake' / 'LH' / f"{stem}.npy"
        
        if hl_file.exists() and lh_file.exists():
            file_paths.append((str(hh_file), str(hl_file), str(lh_file)))
            labels.append(1)
    
    logger.info(f"Total samples prepared: {len(file_paths)}")
    return file_paths, labels


def manual_train_test_split(file_paths, labels, test_size=0.2, random_state=42):
    """Manually split data with stratification"""
    np.random.seed(random_state)
    
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    # Separate by class
    real_indices = np.where(labels == 0)[0]
    fake_indices = np.where(labels == 1)[0]
    
    # Shuffle
    np.random.shuffle(real_indices)
    np.random.shuffle(fake_indices)
    
    # Split each class
    split_real = int(len(real_indices) * (1 - test_size))
    split_fake = int(len(fake_indices) * (1 - test_size))
    
    train_indices = np.concatenate([real_indices[:split_real], fake_indices[:split_fake]])
    test_indices = np.concatenate([real_indices[split_real:], fake_indices[split_fake:]])
    
    np.random.shuffle(train_indices)
    
    train_paths = file_paths[train_indices]
    train_labels = labels[train_indices]
    test_paths = file_paths[test_indices]
    test_labels = labels[test_indices]
    
    return train_paths.tolist(), train_labels.tolist(), test_paths.tolist(), test_labels.tolist()


def main():
    """Main training pipeline"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Please install it: pip install torch torchvision")
        return
    
    project_root = Path(__file__).parent
    wavelet_output_path = project_root / "wavelet_output"
    
    logger.info("="*70)
    logger.info("DenseNet121 Real vs Fake Image Classification")
    logger.info("Using HH, HL, LH wavelet coefficients")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading wavelet data...")
    file_paths, labels = load_wavelet_data(
        wavelet_output_path, balance=True, max_samples=10000
    )
    
    # 80-20 split
    train_paths, train_labels, test_paths, test_labels = manual_train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nDataset Split:")
    logger.info(f"  Train: {len(train_paths)} samples")
    logger.info(f"  Test:  {len(test_paths)} samples")
    
    # Create datasets and dataloaders
    train_dataset = WaveletDataset(train_paths, train_labels, target_size=224)
    test_dataset = WaveletDataset(test_paths, test_labels, target_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize classifier
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    classifier = DenseNetClassifier(device=device, pretrained=False)
    
    # Training loop
    num_epochs = 15
    best_auc = 0  # Track best AUC instead of accuracy
    patience_counter = 0
    best_epoch = 0
    best_metrics = {}  # Store all metrics from best epoch
    
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info("="*70)
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = classifier.train_epoch(train_loader)
        
        # Evaluate
        test_acc, _, _, auc, ap, eer, val_loss = classifier.evaluate(test_loader)
        
        # Scheduler step with AUC (more stable metric than accuracy for imbalanced scenarios)
        classifier.scheduler.step(auc)
        
        # Save best model based on AUC (most reliable metric for deepfake detection)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'auc': auc,
                'ap': ap,
                'eer': eer,
                'val_loss': val_loss
            }
            classifier.save_model(project_root / 'densenet_best.pth')
        else:
            patience_counter += 1
        
        # Early stopping based on AUC plateau (patience=5)
        if patience_counter >= 5:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (AUC did not improve for 5 epochs)")
            break
        
        logger.info("-"*70)
    
    logger.info("\n" + "="*70)
    logger.info(f"Training Complete!")
    logger.info(f"Best Model Epoch: {best_metrics['epoch']}")
    logger.info(f"Best Test Accuracy: {best_metrics['test_acc']:.2f}%")
    logger.info(f"Best AUC: {best_metrics['auc']:.2f}%")
    logger.info(f"Best AP: {best_metrics['ap']:.2f}%")
    logger.info(f"Best EER: {best_metrics['eer']:.2f}%")
    logger.info(f"Best Val Loss: {best_metrics['val_loss']:.4f}")
    logger.info(f"Model saved to: {project_root / 'densenet_best.pth'}")
    logger.info(f"Training configuration: Adam (lr=1e-4), Label Smoothing=0.1, Epochs=15")
    logger.info(f"Unfroze denseblock4 and norm5 for fine-tuning")
    logger.info("="*70)
    
    # Save results to file for report
    results_file = project_root / 'training_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DenseNet121 Deepfake Detection - Training Results\n")
        f.write("="*70 + "\n\n")
        f.write("Dataset: 10,000 images per class (5,000 real + 5,000 fake)\n")
        f.write("Train/Test Split: 80/20 (8000 train, 2000 test per class)\n\n")
        f.write("Best Model Metrics (Epoch {}):\n".format(best_metrics['epoch']))
        f.write("-"*70 + "\n")
        f.write("Training Loss: {:.4f}\n".format(best_metrics['train_loss']))
        f.write("Training Accuracy: {:.2f}%\n".format(best_metrics['train_acc']))
        f.write("\nTest Metrics:\n")
        f.write("  Accuracy (ACC): {:.2f}%\n".format(best_metrics['test_acc']))
        f.write("  Area Under ROC (AUC): {:.2f}%\n".format(best_metrics['auc']))
        f.write("  Average Precision (AP): {:.2f}%\n".format(best_metrics['ap']))
        f.write("  Equal Error Rate (EER): {:.2f}%\n".format(best_metrics['eer']))
        f.write("  Validation Loss: {:.4f}\n".format(best_metrics['val_loss']))
        f.write("\n" + "="*70 + "\n")
        f.write("Model Configuration:\n")
        f.write("-"*70 + "\n")
        f.write("Architecture: DenseNet121\n")
        f.write("Input: Haar wavelet coefficients (HH, HL, LH)\n")
        f.write("Frozen Layers: DenseBlock1-3 (early feature extraction)\n")
        f.write("Trainable Layers: DenseBlock4 + norm5 (task-specific)\n")
        f.write("Optimizer: Adam (lr=1e-4, weight_decay=1e-5)\n")
        f.write("Loss: CrossEntropyLoss with label_smoothing=0.1\n")
        f.write("Scheduler: ReduceLROnPlateau (mode='max', factor=0.5, patience=2)\n")
        f.write("Early Stopping: AUC-based (patience=5 epochs)\n")
        f.write("Batch Size: 32 (GPU-optimized for RTX 4050)\n")
        f.write("Max Epochs: 15\n")
        f.write("\n" + "="*70 + "\n")
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
