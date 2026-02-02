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
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    PLOTTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Plotting libraries not available: {e}")
    PLOTTING_AVAILABLE = False

# Import from the training script
from densenet_classifier import WaveletDataset, DenseNetClassifier, load_wavelet_data


def load_unseen_data(wavelet_output_path, exclude_indices=10000):
    """
    Load unseen wavelet data (data NOT used during training).
    
    Training used: first 10000 real + first 10000 fake images.
    This function loads: remaining real + remaining fake images.
    
    Args:
        wavelet_output_path: Path to wavelet_output directory
        exclude_indices: Number of samples per class to exclude (used in training)
        
    Returns:
        Tuple of (file_paths, labels)
    """
    wavelet_path = Path(wavelet_output_path)
    
    # Load ALL real images
    real_hh_dir = wavelet_path / 'real' / 'HH'
    real_files = sorted([f for f in os.listdir(real_hh_dir) if f.endswith('.npy')])
    real_hh = [real_hh_dir / f for f in real_files]
    
    # Load ALL fake images
    fake_hh_dir = wavelet_path / 'fake' / 'HH'
    fake_files = sorted([f for f in os.listdir(fake_hh_dir) if f.endswith('.npy')])
    fake_hh = [fake_hh_dir / f for f in fake_files]
    
    logger.info(f"Found {len(real_hh)} real images and {len(fake_hh)} fake images")
    logger.info(f"Excluding first {exclude_indices} samples per class (used in training)")
    
    # Exclude training/test samples
    real_hh_unseen = real_hh[exclude_indices:]
    fake_hh_unseen = fake_hh[exclude_indices:]
    
    logger.info(f"Unseen real images: {len(real_hh_unseen)}")
    logger.info(f"Unseen fake images: {len(fake_hh_unseen)}")
    
    # Create file paths and labels
    file_paths = []
    labels = []
    
    # Real images (label 0)
    for hh_file in real_hh_unseen:
        stem = hh_file.stem
        hl_file = wavelet_path / 'real' / 'HL' / f"{stem}.npy"
        lh_file = wavelet_path / 'real' / 'LH' / f"{stem}.npy"
        
        if hl_file.exists() and lh_file.exists():
            file_paths.append((str(hh_file), str(hl_file), str(lh_file)))
            labels.append(0)
    
    # Fake images (label 1)
    for hh_file in fake_hh_unseen:
        stem = hh_file.stem
        hl_file = wavelet_path / 'fake' / 'HL' / f"{stem}.npy"
        lh_file = wavelet_path / 'fake' / 'LH' / f"{stem}.npy"
        
        if hl_file.exists() and lh_file.exists():
            file_paths.append((str(hh_file), str(hl_file), str(lh_file)))
            labels.append(1)
    
    logger.info(f"Total unseen samples prepared: {len(file_paths)}")
    return file_paths, labels


def main():
    """Evaluate trained model on unseen data for robustness analysis"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Please install it: pip install torch torchvision")
        return
    
    project_root = Path(__file__).parent
    wavelet_output_path = project_root / "wavelet_output"
    model_path = project_root / "densenet_best.pth"
    
    logger.info("="*70)
    logger.info("DenseNet121 Unseen Data Evaluation - Robustness Analysis")
    logger.info("Using HH, HL, LH wavelet coefficients")
    logger.info("="*70)
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run densenet_classifier.py first to train the model.")
        return
    
    # Load unseen data
    logger.info("\nLoading unseen wavelet data...")
    file_paths, labels = load_unseen_data(wavelet_output_path, exclude_indices=10000)
    
    if len(file_paths) == 0:
        logger.error("No unseen data found!")
        return
    
    # Create dataset and dataloader
    logger.info("\nPreparing DataLoader...")
    unseen_dataset = WaveletDataset(file_paths, labels, target_size=224)
    unseen_loader = DataLoader(
        unseen_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Initialize classifier and load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info("Loading trained model...")
    classifier = DenseNetClassifier(device=device, pretrained=False)
    classifier.load_model(model_path)
    
    # Evaluate on unseen data
    logger.info("\n" + "="*70)
    logger.info("UNSEEN DATA EVALUATION (Robustness Analysis)")
    logger.info("="*70)
    
    test_acc, all_preds, all_labels, auc, ap, eer, val_loss = classifier.evaluate(unseen_loader)
    
    # Compute class-wise metrics
    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    
    real_acc = 100 * np.mean(all_preds[real_mask] == all_labels[real_mask])
    fake_acc = 100 * np.mean(all_preds[fake_mask] == all_labels[fake_mask])
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY METRICS")
    logger.info("="*70)
    logger.info(f"Overall Accuracy: {test_acc:.2f}%")
    logger.info(f"Real Class Accuracy: {real_acc:.2f}%")
    logger.info(f"Fake Class Accuracy: {fake_acc:.2f}%")
    logger.info(f"AUC (ROC): {auc:.2f}%")
    logger.info(f"Average Precision: {ap:.2f}%")
    logger.info(f"Equal Error Rate: {eer:.2f}%")
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info("="*70)
    
    # Compute and visualize confusion matrix
    if PLOTTING_AVAILABLE:
        logger.info("\nGenerating confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)
        
        # Normalize by true labels (row-wise) for better interpretation
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Proportion'},
            square=True,
            linewidths=2,
            linecolor='black'
        )
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - Unseen Data (Normalized by True Label)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        cm_path = project_root / 'unseen_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Log confusion matrix values
        logger.info("\nConfusion Matrix (Normalized by True Label):")
        logger.info("-"*70)
        logger.info(f"True Real -> Predicted Real: {cm_normalized[0, 0]:.2%}")
        logger.info(f"True Real -> Predicted Fake: {cm_normalized[0, 1]:.2%}")
        logger.info(f"True Fake -> Predicted Real: {cm_normalized[1, 0]:.2%}")
        logger.info(f"True Fake -> Predicted Fake: {cm_normalized[1, 1]:.2%}")
        logger.info("-"*70)
    else:
        logger.warning("Plotting libraries not available. Skipping confusion matrix visualization.")
    
    # Save results to file
    results_file = project_root / 'unseen_data_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DenseNet121 Unseen Data Evaluation - Robustness Analysis\n")
        f.write("="*70 + "\n\n")
        f.write("Dataset: All unseen wavelet data (NOT used in training)\n")
        f.write(f"Real samples: {np.sum(real_mask)}\n")
        f.write(f"Fake samples: {np.sum(fake_mask)}\n")
        f.write(f"Total samples: {len(all_labels)}\n\n")
        f.write("="*70 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Overall Accuracy (ACC): {test_acc:.2f}%\n")
        f.write(f"Real Class Accuracy: {real_acc:.2f}%\n")
        f.write(f"Fake Class Accuracy: {fake_acc:.2f}%\n")
        f.write(f"Area Under ROC (AUC): {auc:.2f}%\n")
        f.write(f"Average Precision (AP): {ap:.2f}%\n")
        f.write(f"Equal Error Rate (EER): {eer:.2f}%\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        
        # Add confusion matrix if available
        if PLOTTING_AVAILABLE:
            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            f.write(f"\n" + "="*70 + "\n")
            f.write("CONFUSION MATRIX (Normalized by True Label)\n")
            f.write("="*70 + "\n")
            f.write(f"True Real -> Predicted Real: {cm_normalized[0, 0]:.2%}\n")
            f.write(f"True Real -> Predicted Fake: {cm_normalized[0, 1]:.2%}\n")
            f.write(f"True Fake -> Predicted Real: {cm_normalized[1, 0]:.2%}\n")
            f.write(f"True Fake -> Predicted Fake: {cm_normalized[1, 1]:.2%}\n")
            f.write(f"\nConfusion matrix visualization saved to: unseen_confusion_matrix.png\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("MODEL & ARCHITECTURE\n")
        f.write("="*70 + "\n")
        f.write("Model: DenseNet121\n")
        f.write("Input: Haar wavelet coefficients (HH, HL, LH)\n")
        f.write("Frozen Layers: DenseBlock1-3 (early feature extraction)\n")
        f.write("Trainable Layers: DenseBlock4 + norm5 (task-specific)\n")
        f.write("Optimizer: Adam (lr=1e-4, weight_decay=1e-5)\n")
        f.write("Loss: CrossEntropyLoss with label_smoothing=0.1\n")
        f.write("Batch Size: 32 (GPU-optimized for RTX 4050)\n")
        f.write("\n" + "="*70 + "\n")
        f.write("NOTES\n")
        f.write("="*70 + "\n")
        f.write("This evaluation tests the model on completely unseen data.\n")
        f.write("Training used 10,000 images per class (first 10k real, first 10k fake).\n")
        f.write("Unseen data consists of remaining images from the full dataset.\n")
        f.write("Results indicate model generalization and robustness.\n")
        f.write("="*70 + "\n")
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
