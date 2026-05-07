# Tomato Leaf Disease Prediction

A deep learning model that classifies tomato plant diseases from leaf images, built with PyTorch.

## Introduction

More than 25% of total tomato crop production is lost annually due to plant diseases, which significantly reduce yields and degrade crop quality. Human identification of these diseases can be slow and error-prone. This project trains a neural network to identify tomato leaf diseases from a single image, enabling faster and more consistent diagnosis.

## Data

The model is trained on [this dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) from Kaggle. Rather than using the default train/validation split the dataset ships with, our code re-splits the data into training, validation, and test sets for more robust evaluation.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/ConnorFair36/Tomato-Leaf-Disease-Prediction
cd Tomato-Leaf-Disease-Prediction
```

### 2. Set up your Python environment

This example uses [uv](https://github.com/astral-sh/uv), but any Python environment manager will work.

```bash
uv init
uv add -r requirements.txt
```

### 3. Download the dataset

Get the dataset from [here](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) and place the train and validation folders into the data file. The python script that comes with this dataset can be thrown away because we will be using pytorch instead of tensorflow.

```
Tomato-Leaf-Disease-Prediction/
└── data/
    ├── train/
    │   ├── Tomato_Bacterial_spot/
    │   ├── Tomato_Early_blight/
    │   ├── Tomato_healthy/
    │   ├── Tomato_Late_blight/
    │   ├── Tomato_Leaf_Mold/
    │   ├── Tomato_Septoria_leaf_spot/
    │   ├── Tomato_Spider_mites_Two-spotted_spider_mite/
    │   ├── Tomato__Target_Spot/
    │   ├── Tomato__Tomato_mosaic_virus/
    │   └── Tomato__Tomato_YellowLeaf__Curl_Virus/
    └── val/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_Early_blight/
        ├── Tomato_healthy/
        ├── Tomato_Late_blight/
        ├── Tomato_Leaf_Mold/
        ├── Tomato_Septoria_leaf_spot/
        ├── Tomato_Spider_mites_Two-spotted_spider_mite/
        ├── Tomato__Target_Spot/
        ├── Tomato__Tomato_mosaic_virus/
        └── Tomato__Tomato_YellowLeaf__Curl_Virus/
```

### Training

To train this model, change `MODE: "train"` in config.yaml and change the parameters however you want. The model weights are automaticly stored in `src/weights` within their own folder after training.

```{bash}
uv run main.py config.yaml
```

### Inference

To run inference on an existing model, set `MODE: "inference"` in cofig.yaml and update `MODEL_WEIGHTS` to the file directory thak contains the .pt file for the model. 

```{bash}
uv run main.py config.yaml
```

## Resources

Resources used to learn PyTorch, organize the project, and collect data.

| Topic | Link |
|---|---|
| PyTorch fundamentals | [learnpytorch.io](https://www.learnpytorch.io) |
| Introduction to neural networks (Chapter 1) | [The LM Book](https://thelmbook.com) |
| PyTorch project organization | [Medium — Good Practices for Deep Learning](https://medium.com/@zhangx9411/the-guide-to-pytorch-good-practices-for-deep-learning-bd9e90bf8c0e) |
| Dataset | [Tomato Leaf Dataset — Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) |

## References

Trivedi, N. K., Gautam, V., Anand, A., Aljahdali, H. M., Villar, S. G., Anand, D., Goyal, N., & Kadry, S. (2021). Early Detection and Classification of Tomato Leaf Disease Using High-Performance Deep Neural Network. *Sensors*, 21(23), 7987. https://doi.org/10.3390/s21237987

Kantor, L., & Blazejczyk, A. (2021). *Food Availability and Consumption*. USDA Economic Research Service. https://www.ers.usda.gov/data-products/ag-and-food-statistics-charting-the-essentials/food-availability-and-consumption
