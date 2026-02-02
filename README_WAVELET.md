# Wavelet Decomposition Project Documentation

## Overview
This project applies 2D Discrete Wavelet Transform (DWT) to a dataset of real and fake images, decomposing each image into wavelet coefficients (cA, cH, cV, cD) using the Daubechies-4 (db4) wavelet.

## Dataset Structure
```
Dataset/
├── Real/        (30,000 images - real video frames)
└── Fake/        (35,000+ images - deepfake/synthetic video frames)
```

### Dataset Statistics
- **Total Images**: ~65,000+
- **Real Images**: 30,000
- **Fake Images**: 35,000+
- **Image Format**: JPG
- **Real Image Naming**: Numeric IDs (e.g., 24016.jpg, 24017.jpg, ..., 29999.jpg)
- **Fake Image Naming**: fake_* prefix (e.g., fake_62.jpg, fake_6200.jpg, ..., fake_65399.jpg)

## Wavelet Decomposition

### What is Wavelet Decomposition?
Wavelet decomposition breaks down an image into different frequency components:
- **cA (Approximation)**: Low-frequency content (smooth areas, overall structure)
- **cH (Horizontal Details)**: High-frequency content in horizontal direction (vertical edges)
- **cV (Vertical Details)**: High-frequency content in vertical direction (horizontal edges)
- **cD (Diagonal Details)**: Diagonal edge information

### Why db4?
The Daubechies-4 (db4) wavelet is chosen because:
1. Good balance between time and frequency localization
2. Effective for feature extraction in image analysis
3. Widely used in deepfake detection research
4. Captures both high and low-frequency components effectively

### Single-Level Decomposition
Each image (size H×W) is decomposed into 4 coefficient matrices:
- cA: (H/2 × W/2) - approximation coefficients
- cH: (H/2 × W/2) - horizontal detail coefficients
- cV: (H/2 × W/2) - vertical detail coefficients  
- cD: (H/2 × W/2) - diagonal detail coefficients

## Output Structure
```
wavelet_output/
├── real/                 (Real image decompositions)
│   ├── cA/              (30,000 approximation coefficients)
│   ├── cH/              (30,000 horizontal details)
│   ├── cV/              (30,000 vertical details)
│   └── cD/              (30,000 diagonal details)
└── fake/                (Fake image decompositions)
    ├── cA/              (35,000+ approximation coefficients)
    ├── cH/              (35,000+ horizontal details)
    ├── cV/              (35,000+ vertical details)
    └── cD/              (35,000+ diagonal details)
```

Each coefficient is saved in two formats:
1. **PNG Image**: Normalized to 0-255 range for visualization
2. **NumPy Array (.npy)**: Raw coefficient values for numerical analysis

## Files in This Project

### Main Script: `wavelet_decomposition.py`
- **Purpose**: Apply wavelet decomposition to all images
- **Class**: `WaveletDecomposer`
- **Key Methods**:
  - `decompose_image()`: Apply 2D DWT to single image
  - `save_coefficients()`: Save as PNG and NumPy format
  - `process_category()`: Process all images in a category
  - `process_all()`: Main execution method

### Analysis Script: `analyze_wavelet.py`
- **Purpose**: Analyze and visualize decomposition results
- **Features**:
  - Count processed images per coefficient type
  - Visualize sample decomposition
  - Generate comparison plots

## Processing Details

### Preprocessing
1. Load image from disk (JPEG)
2. Convert to grayscale (L mode - single channel)
3. Convert to float32 for numerical stability

### Decomposition
```python
import pywt
cA, (cH, cV, cD) = pywt.dwt2(image_array, 'db4')
```

### Coefficient Normalization
Raw coefficients contain negative values and large ranges. For visualization:
```
normalized = ((coef - min) / (max - min)) * 255
```

This preserves relative intensity information while making it viewable as standard images.

### Output Format Details

#### PNG Files
- 8-bit grayscale (0-255 range)
- Same filename as input image
- Located in coefficient subdirectories
- Suitable for visualization and inspection

#### NumPy Arrays (.npy)
- Float32 type (original precision)
- Full range of values preserved
- Suitable for machine learning models
- Can be loaded: `np.load('coeff.npy')`

## Performance Considerations

### Memory Usage
- Each decomposition loads 1 image at a time
- Saves 4 output files per input image
- Keeps only 1 image in memory during processing
- **Estimated disk usage**: ~200-300 GB for full dataset

### Processing Time
- ~1-2 seconds per image (approximate)
- **Total estimated time**: 20-30 hours for full dataset
- Can be parallelized for faster processing if needed

### Optimization Options
1. **Batch Processing**: Process multiple images in parallel
2. **Multi-threading**: Use Python multiprocessing
3. **Compression**: Store only essential coefficients
4. **Subsampling**: Process every nth image for testing

## Usage

### Basic Usage
```bash
python wavelet_decomposition.py
```

### With Custom Parameters
```python
from wavelet_decomposition import WaveletDecomposer

decomposer = WaveletDecomposer(
    dataset_path="Dataset",
    output_path="wavelet_output",
    wavelet='db4'
)
decomposer.process_all()
```

### Analyzing Results
```bash
python analyze_wavelet.py
```

## Applications

This wavelet decomposition can be used for:
1. **Deepfake Detection**: Train ML models on decomposed coefficients
2. **Feature Analysis**: Study differences between real and fake images
3. **Artifact Detection**: Identify compression/manipulation artifacts in detail coefficients
4. **Frequency Analysis**: Analyze specific frequency bands separately

## Future Enhancements

1. **Multi-level Decomposition**: Apply decomposition multiple times recursively
2. **Parallel Processing**: Use multiprocessing for faster execution
3. **Compression**: Implement wavelet coefficient quantization
4. **Feature Extraction**: Automatically compute statistics on coefficients
5. **Visualization Dashboard**: Interactive exploration of results

## References

- PyWavelets Documentation: https://pywavelets.readthedocs.io/
- Wavelet Transforms in Image Processing: https://en.wikipedia.org/wiki/Wavelet_transform
- Daubechies Wavelets: https://en.wikipedia.org/wiki/Daubechies_wavelet

## Troubleshooting

### Issue: Out of Memory
- **Solution**: Process images in smaller batches or use multi-threading with queue

### Issue: Slow Processing
- **Solution**: Use SSD instead of HDD, reduce image quality, or parallelize

### Issue: Corrupted Images
- **Solution**: Script skips corrupted images and logs them, continue processing

## Project Status

- ✅ Data Exploration: Dataset structure analyzed (30K real, 35K+ fake images)
- ✅ Code Development: Main decomposition script created
- 🔄 Processing: Running full dataset decomposition
- ⏳ Analysis: Post-processing and visualization pending
- ⏳ ML Model Training: Ready after decomposition complete

---

**Last Updated**: January 27, 2026
**Processing Status**: In Progress (Real folder ~23% complete)
