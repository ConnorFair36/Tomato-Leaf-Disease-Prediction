import os
from yacs.config import CfgNode as CN
import argparse
from src.configs import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.read_data import get_dataloaders
from src.data.preprocessors import DATA_TRANSFORM
from src.models.CNN import CNN
from src.engine.train_loop import train_model
from src.engine.eval import evaluate_model


def get_file_arg():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="The main script for running tests on our CNN model.")
    # 2. Add arguments
    parser.add_argument("cfg_file", help="The path to the config.yaml file.")
    # 3. Parse arguments
    args = parser.parse_args()
    # 4. Use the values
    return str(args.cfg_file)


def get_optim(model):
    name_to_optim = {
        "SGD": optim.SGD,
        "SGDmomentum": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW
    }
    optimizer = name_to_optim[config.TRAINING.OPTIMIZER]
    if config.TRAINING.OPTIMIZER == "SGDmomentum":
        optimizer = optimizer(model.parameters(), lr=config.TRAINING.LEARNING_RATE, momentum=0.9)
    else:
        optimizer = optimizer(model.parameters(), lr=config.TRAINING.LEARNING_RATE)
    return optimizer


def create_training_folder():
    folder_name = f"src/weights/{config.TRAINING.MODEL_NAME}"
    counter = 1
    temp_name = folder_name

    while os.path.exists(temp_name):
        temp_name = f"{folder_name}_{counter}"
        counter += 1

    os.makedirs(temp_name)
    return temp_name


def training_mode():
    train_data, val_data, _ = get_dataloaders(DATA_TRANSFORM, n_workers=config.DATASET.WORKERS)
    model = CNN()
    model = torch.compile(model)
    loss_function = nn.CrossEntropyLoss()
    optimizer = get_optim(model)
    epochs_list = sorted(config.TRAINING.EPOCHS)
    if len(epochs_list) > 1:
        epochs_list_to_run = [last - first for first, last in zip(epochs_list[:-1], epochs_list[1:])]
        epochs_list_to_run.insert(0, epochs_list[0])
    else:
        epochs_list_to_run = epochs_list

    folder_name = create_training_folder()

    all_results = [[],[]]
    for total_epochs, epochs in zip(epochs_list, epochs_list_to_run):
        print(f"---{total_epochs}---")
        results = train_model(
            model=model,
            train_loader=train_data,
            val_loader=val_data,
            epochs=epochs,
            loss_fn=loss_function,
            optimizer=optimizer
        )
        checkpoint = {
            'model_state_dict': model._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': total_epochs
        }
        torch.save(checkpoint, f'{folder_name}/{config.TRAINING.MODEL_NAME}_{total_epochs}.pt')
        all_results[0].extend(results[0])
        all_results[1].extend(results[1])
    df = pd.DataFrame({"train": results[0], "validation": results[1]})
    df.to_csv(f"{folder_name}/{config.TRAINING.MODEL_NAME}.csv")


def print_metric(metric: str, results: dict):
    print(f"{metric}:")
    for name, value in results.items():
        if metric in name:
            print(f"  {name[len(metric)+1:]}: {value}")


def inference_mode():
    assert config.INFERENCE.DATASET in ("validation", "test")
    class_names = ["Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Spider mitesTwo",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "Healthy"]
    model = CNN()
    state = torch.load(config.INFERENCE.MODEL_WEIGHTS)
    model.load_state_dict(state["model_state_dict"])
    _, val_data, test_data = get_dataloaders(DATA_TRANSFORM, n_workers=config.DATASET.WORKERS)
    if config.INFERENCE.DATASET == "validation":
        results = evaluate_model(model, val_data)
    else:
        results = evaluate_model(model, test_data)
    print(f"Accuracy: {results['accuracy']}")
    print_metric('precision', results)
    print_metric('recall', results)
    print_metric('mcc', results)
    cm = results['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 8})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {config.INFERENCE.DATASET}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # read in the configuration file from the cli
    cfg_file = get_file_arg()
    config.merge_from_file(cfg_file)
    config.freeze()
    # run or train the model using the set configurations
    if config.MODE == "train":
        training_mode()
    elif config.MODE == "inference":
        inference_mode()
        
