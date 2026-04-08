import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    # Walk up one level from notebooks/ to reach the project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.data.read_data import get_dataloaders
    from src.data.preprocessors import DATA_TRANSFORM
    from src.models.CNN import CNN
    from src.engine.train_loop import train_model, sucessive_halving

    return DATA_TRANSFORM, nn, optim, sucessive_halving


@app.cell
def _(nn):
    # a stupid model because I'm impatient, lol
    class idiots_mlp(nn.Module):
        def __init__(self, n_hidden: int = 1000):
            super().__init__()
            self.model_weights = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*256*1, n_hidden),
                nn.Linear(n_hidden, 10)
            )

        def forward(self, x):
            return self.model_weights(x)

    return (idiots_mlp,)


@app.cell
def _(DATA_TRANSFORM, nn, optim):
    # the algorithm needs a dictonary of lists of parameters to try
    params = {
        "model_n_hidden": [1000, 500],          # parameters that start with "model_" will be imputed directly into the model
        "optimizer": [optim.AdamW, optim.SGD],  # optimizers should be given not instanciated
        "optim_lr": [0.0001, 0.001],            # parameters that start with "optim_" will be imputed directly into the optimizer
        "loss": [nn.CrossEntropyLoss],                   # losses should be given not instanciated
        "preprocessor": [DATA_TRANSFORM]        # data transformations are given through preprocessor
    }
    # The algorithm will start by running all possible number of combinations of these hyperparameters
    #  in this case it will run 2*2*2*1 combinations of these on the first round, then every iteration will divide
    #  the number of used parameters in half after each round until there is one winner
    return (params,)


@app.cell
def _(idiots_mlp, params, sucessive_halving):
    sucessive_halving(
        model_class =  idiots_mlp,                    # takes the model type as input
        epochs = 50,                                  # total number of epochs that could be run
        hyper_parameters = params,                    # the dictonary of parameters to test
        checkpoint_folder = "./src/weights/test_run/" # the folder where every model being tested will be checkpointed
    )
    # While this algorithm runs, each round will double the number of epochs per run until the final model trained on all the epochs
    #  in this case with our 8 possible combinations and 50 epochs:
    #  - 1st round will run all 8 model for 7 epochs 
    #  - 2nd round will run 4 models for an aditional 6 epochs (13 epochs total for these models)
    #  - 3nd round will run 2 models for an aditional 12 epochs (25 epochs total for these models)
    #  - 4nd round will run the winning model for an aditional 25 epochs (50 epochs total for the winner)

    # all of the final model weights will be stored in the checkpoint folder. This doesn't store any of the intermetiate weights, but I can change that if it's needed :)
    return


if __name__ == "__main__":
    app.run()
