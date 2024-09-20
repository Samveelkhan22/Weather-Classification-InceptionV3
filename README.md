# Weather Classification using Transfer Learning with InceptionV3

This repository contains the implementation of weather condition classification using transfer learning with the pre-trained **InceptionV3** model. The dataset consists of images of different weather conditions such as rain, snow, fog, and others, which are classified into 11 categories. Transfer learning is employed to fine-tune the model for multi-class image classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## Overview
This project leverages **InceptionV3** pre-trained on ImageNet and fine-tunes it to classify images of various weather conditions. The model was trained using **transfer learning**, where the base InceptionV3 layers were frozen, and new layers were trained for weather classification. After that, the model was fine-tuned by unfreezing the InceptionV3 layers to achieve better accuracy.

### Key Features:
- Uses **InceptionV3** pre-trained on **ImageNet**.
- Achieves **82% accuracy** or higher on the weather condition classification task.
- Handles 11 different weather conditions.

## Dataset
The dataset used in this project contains images of 11 different weather conditions, such as:
- Rain
- Snow
- Fog
- Hail
- Frost
- Dew
- Lightning
- Rainbow
- Rime
- Sandstorm
- Glaze

The dataset was split into training and testing subsets, with separate folders for each class.

## Model Architecture
The model is based on **InceptionV3** pre-trained on ImageNet. The final layers added to the model are:
- A **GlobalAveragePooling2D** layer.
- A **Dense** layer with 1024 units and **ReLU** activation.
- A **Dense** layer for classification with **softmax** activation, mapping to the 11 weather categories.

The model is trained in two phases:
1. **Train new layers** added on top of the frozen InceptionV3 base.
2. **Fine-tune the entire model** by unfreezing the InceptionV3 layers.
