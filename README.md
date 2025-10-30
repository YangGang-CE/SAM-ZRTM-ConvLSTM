# 3D-point cloud-based spatiotemporal series model for tunnel rock mass discontinuities prediction

This project implements ConvLSTM and Self-Attention ConvLSTM models for spatiotemporal sequence prediction.

The complete code is still under preparation at present. Coming soon!

## Author
- **Gang YANG**
- **Contact**: yanggang_ce@163.com

## Models

### 1. ConvLSTM
- Standard Convolutional LSTM for spatiotemporal modeling
- Uses convolutions instead of fully connected layers
- Maintains spatial structure while processing temporal sequences

### 2. Self-Attention ConvLSTM (SA-ConvLSTM)
- Enhanced ConvLSTM with self-attention mechanisms
- Includes Self-Attention Memory (SAM) module
- Uses zigzag recurrent transition mechanism
- Better captures long-range spatial dependencies

## Features

- **Teacher Forcing**: Configurable teacher forcing rate during training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **TensorBoard Logging**: Comprehensive training visualization
- **Multiple Metrics**: MSE and SSIM evaluation
- **Frame-wise Analysis**: Detailed per-frame performance metrics

### Prediction
```bash
# Predict from a directory containing 10 sequential images (00.png to 09.png)
python predictor.py --model_path ./checkpoint.pt --input_dir ./data/ --output_dir ./predictions --visualize
```

## Configuration

Key hyperparameters can be modified in the main functions:

- `BATCH_SIZE`: Training batch size (default: 8)
- `IMG_SIZE`: Image dimensions (default: 256x256)
- `INPUT_WINDOW_SIZE`: Input sequence length (default: 10)
- `OUTPUT_WINDOW_SIZE`: Prediction sequence length (default: 10)
- `EPOCHS`: Number of training epochs (default: 500)
- `LEARNING_RATE`: Learning rate (default: 1e-3)
- `MODEL_TYPE`: Model type ('sa_convlstm' or 'convlstm')

## Dataset

The Moving MNIST dataset consists of:
- 2564x256 grayscale images (can be downsampled to 64x64)
- 20-frame sequences (10 input + 10 prediction)
- Moving handwritten digits
- Image-based directory structure for better visualization

### Data Structure

The dataset is organized in the following directory structure:
```
data/
├── train/
│   ├── 00001/
│   │   ├── 00.png
│   │   ├── 01.png
│   │   └── ... (up to 19.png)
│   ├── 00002/
│   └── ...
└── test/
    ├── 09001/
    └── ...
```


**Important**: The model supports various image sizes. For 256x256 images, they are encoded to 64x64 feature maps internally through 4 downsampling layers.

## Requirements

- PyTorch
- NumPy
- scikit-image
- matplotlib
- tensorboardX
- Pillow (PIL)
- tqdm (for conversion progress)

## License

This project is for academic and research purposes.


Copyright belongs to Gang YANG and use of this code for commercial applications or profit-driven ventures requires explicit permission from the author.





