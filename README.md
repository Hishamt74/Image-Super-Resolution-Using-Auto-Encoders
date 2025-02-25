# Image-Super-Resolution-Using-Auto-Encoders
## Introduction
This project implements an **Image Super-Resolution** model using **CNN-based Autoencoders**. The goal is to enhance the resolution of low-quality images by training an autoencoder that learns to reconstruct high-resolution images from their low-resolution counterparts.

## Features
- Uses **Convolutional Autoencoders (CAE)** for upscaling images.
- Custom dataset preprocessing pipeline.
- Trained using **Mean Squared Error (MSE)** loss for image reconstruction.
- Supports multiple datasets for training and testing.
- Implements **OpenCV & Matplotlib** for visualization.
## Dependencies
Ensure you have the following installed:
- numpy
- matplotlib
- tensorflow
- keras
- opencv-python
- scikit-image
## Model Architecture
The model consists of:
1. **Encoder:** Extracts important features from the low-resolution image.
2. **Bottleneck Layer:** Compresses the features while retaining essential details.
3. **Decoder:** Reconstructs the high-resolution image from the learned features.
