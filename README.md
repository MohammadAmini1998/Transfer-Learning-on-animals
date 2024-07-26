# Animal Image Classification using Transformers, ResNet18, and VGG16

## Overview

This project aims to classify images of various animals using state-of-the-art deep learning models, specifically Transformers, ResNet18, and VGG16. The models are trained and evaluated on a comprehensive animal image dataset available on Kaggle.

## Transformers

Transformers, introduced in the paper "Attention is All You Need" by Vaswani et al., have revolutionized the field of natural language processing and have been successfully adapted for computer vision tasks. Unlike traditional convolutional neural networks (CNNs), Transformers leverage self-attention mechanisms to model global dependencies in the data. This allows them to capture intricate patterns and relationships within images, leading to superior performance in many vision tasks.

## Dataset

The dataset used in this project is the [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals), which contains images of 90 different animals. It is a rich collection that provides a diverse set of challenges for image classification models. The dataset includes:

- **Number of classes**: 90
- **Total images**: 29,206
- **Image format**: JPEG
- **Image size**: Various

This dataset is ideal for training and evaluating deep learning models due to its variety and complexity.

## VGG16

VGG16 is a deep convolutional neural network architecture that was introduced by the Visual Geometry Group (VGG) at the University of Oxford. It consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. The key features of VGG16 are:

- **Simplicity**: The architecture uses small (3x3) convolution filters, making it straightforward yet powerful.
- **Depth**: With 16 layers, VGG16 is deeper than many earlier models, allowing it to capture more complex features.
- **Performance**: VGG16 has been proven to perform well on a variety of image classification tasks, including the ImageNet challenge.

## ResNet18

ResNet18 is part of the ResNet (Residual Networks) family, which introduced the concept of residual learning. This architecture addresses the problem of vanishing gradients in deep networks by introducing skip connections, allowing gradients to flow more easily through the network. Key features of ResNet18 include:

- **Residual Blocks**: These blocks enable the training of very deep networks by allowing gradients to bypass one or more layers.
- **Depth**: ResNet18 has 18 layers, making it less deep compared to VGG16 but highly efficient due to its residual connections.
- **Efficiency**: Despite being shallower, ResNet18 often achieves comparable or even better performance than deeper networks without residual connections.

## Project Structure

- `data/`: Contains the animal image dataset.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model training.
- `models/`: Pre-trained and fine-tuned models.
- `scripts/`: Python scripts for training and evaluating the models.
- `results/`: Evaluation metrics and model predictions.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/animal-image-classification.git
   cd animal-image-classification
