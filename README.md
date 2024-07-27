# 🐾 Animal Image Classification using Transformers, ResNet18, and VGG16 🐾

## 📌 Overview

This project aims to classify images of various animals using state-of-the-art deep learning models, specifically Transformers, ResNet18, and VGG16. The models are trained and evaluated on a comprehensive animal image dataset available on Kaggle.

## 🔄 Transfer Learning

Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. This project leverages pre-trained models like ResNet18 and VGG16, which were originally trained on large datasets such as ImageNet, and fine-tunes them on the animal image dataset.

### Benefits of Transfer Learning:
- **⏱ Reduced Training Time**: The model, already partially trained, requires less time to converge.
- **🚀 Improved Performance**: Pre-trained models have learned useful features from a large dataset, enhancing performance on the new task.
- **📉 Less Data Required**: Achieve good performance even with a smaller dataset.

## 📊 Dataset

The dataset used in this project is the [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals), which contains images of 90 different animals. It provides a rich collection of diverse challenges for image classification models.

### Dataset Details:
- **Number of classes**: 90
- **Total images**: 5400
- **Image format**: JPEG
- **Image size**: Various

## 🧠 Models

### VGG16

VGG16 is a deep convolutional neural network architecture introduced by the Visual Geometry Group (VGG) at the University of Oxford. It consists of 16 layers, including 13 convolutional layers and 3 fully connected layers.

#### Key Features:
- **Simplicity**: Uses small (3x3) convolution filters.
- **Depth**: With 16 layers, it captures more complex features.
- **Performance**: Proven to perform well on various image classification tasks, including the ImageNet challenge.

[Read the VGG16 Paper](https://arxiv.org/abs/1409.1556)

### ResNet18

ResNet18 is part of the ResNet (Residual Networks) family, which introduced the concept of residual learning. This architecture addresses the problem of vanishing gradients in deep networks by introducing skip connections, allowing gradients to flow more easily.

#### Key Features:
- **Residual Blocks**: Enable training of very deep networks.
- **Depth**: Has 18 layers, making it less deep but highly efficient due to its residual connections.
- **Efficiency**: Achieves comparable or even better performance than deeper networks without residual connections.

[Read the ResNet Paper](https://arxiv.org/abs/1512.03385)

## 📈 Results

![Results](https://github.com/user-attachments/assets/e23028c4-2e2c-4a39-89d1-b80c2d8e74a1)

## 💻 Usage

To get started, clone the repository and run the Jupyter Notebook file:

```bash
git clone https://github.com/yourusername/animal-image-classification.git
cd animal-image-classification
jupyter notebook
