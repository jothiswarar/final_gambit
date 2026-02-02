import os
import pywt
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaveletDecomposer:
    """Apply wavelet decomposition to images and organize output"""
    
    def __init__(self, dataset_path, output_path, wavelet='haar'):
        """
        Initialize the wavelet decomposer
        
        Args:
            dataset_path: Path to dataset folder containing 'Real' and 'Fake' subfolders
            output_path: Path where to save decomposed images
            wavelet: Wavelet type (default: 'haar' - Haar wavelet)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.wavelet = wavelet
        
        # Create output directory structure
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directory structure"""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        for category in ['Real', 'Fake']:
            category_path = self.output_path / category.lower()
            category_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for each coefficient type
            for coef in ['LL', 'LH', 'HL', 'HH']:
                (category_path / coef).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory structure at {self.output_path}")
    
    def decompose_image(self, image_path):
        """
        Apply 2D wavelet decomposition to an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (LL, LH, HL, HH) coefficient arrays, or None if error
        """
        try:
            # Load image and convert to grayscale
            img = Image.open(image_path).convert('L')
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32)
            
            # Apply 2D discrete wavelet transform (single level)
            cA, (cH, cV, cD) = pywt.dwt2(img_array, self.wavelet)
            
            # Map to standard notation: LL, LH, HL, HH
            return cA, cH, cV, cD  # LL, LH, HL, HH
            
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")
            return None
    
    def _normalize_coefficients(self, coef_array):
        """Normalize coefficient array to 0-255 range for visualization"""
        coef_min = np.min(coef_array)
        coef_max = np.max(coef_array)
        
        if coef_max - coef_min == 0:
            return np.zeros_like(coef_array, dtype=np.uint8)
        
        normalized = ((coef_array - coef_min) / (coef_max - coef_min) * 255).astype(np.uint8)
        return normalized
    
    def save_coefficients(self, coefficients, image_name, category):
        """
        Save decomposed coefficients as images and numpy arrays
        
        Args:
            coefficients: Tuple of (LL, LH, HL, HH)
            image_name: Original image filename (without extension)
            category: 'Real' or 'Fake'
        """
        if coefficients is None:
            return False
        
        LL, LH, HL, HH = coefficients
        category_lower = category.lower()
        base_path = self.output_path / category_lower
        
        coef_dict = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
        
        try:
            for coef_name, coef_array in coef_dict.items():
                # Save as PNG image (normalized)
                normalized = self._normalize_coefficients(coef_array)
                img_output = Image.fromarray(normalized)
                img_path = base_path / coef_name / f"{image_name}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_output.save(img_path)
                
                # Save as numpy array
                npy_path = base_path / coef_name / f"{image_name}.npy"
                np.save(npy_path, coef_array)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving coefficients for {image_name}: {e}")
            return False
    
    def process_category(self, category):
        """
        Process all images in a category (Real or Fake)
        
        Args:
            category: 'Real' or 'Fake'
            
        Returns:
            Tuple of (processed_count, failed_count)
        """
        category_path = self.dataset_path / category
        
        if not category_path.exists():
            logger.error(f"Category path does not exist: {category_path}")
            return 0, 0
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([f for f in category_path.iterdir() 
                      if f.suffix.lower() in image_extensions])
        
        logger.info(f"Processing {len(image_files)} images from {category} folder...")
        
        processed = 0
        failed = 0
        
        # Use tqdm for progress bar
        for image_path in tqdm(image_files, desc=f"Processing {category}", unit="images"):
            # Decompose image
            coefficients = self.decompose_image(image_path)
            
            if coefficients is not None:
                # Save coefficients
                image_name = image_path.stem  # Filename without extension
                if self.save_coefficients(coefficients, image_name, category):
                    processed += 1
                else:
                    failed += 1
            else:
                failed += 1
        
        logger.info(f"Completed {category}: {processed} successful, {failed} failed")
        return processed, failed
    
    def process_all(self):
        """Process all images from Real and Fake folders"""
        logger.info("Starting wavelet decomposition...")
        logger.info(f"Wavelet: {self.wavelet}")
        
        all_processed = 0
        all_failed = 0
        
        for category in ['Real', 'Fake']:
            processed, failed = self.process_category(category)
            all_processed += processed
            all_failed += failed
        
        logger.info(f"\n{'='*50}")
        logger.info(f"DECOMPOSITION COMPLETE")
        logger.info(f"Total Processed: {all_processed}")
        logger.info(f"Total Failed: {all_failed}")
        logger.info(f"Output saved to: {self.output_path}")
        logger.info(f"{'='*50}")


def main():
    """Main entry point"""
    # Define paths
    project_root = Path(__file__).parent
    dataset_path = project_root / "Dataset"
    output_path = project_root / "wavelet_output"
    
    # Create decomposer and process
    decomposer = WaveletDecomposer(dataset_path, output_path, wavelet='haar')
    decomposer.process_all()


if __name__ == "__main__":
    main()
