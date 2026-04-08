import torch
from torch import nn, optim, utils
from torch.utils.data import DataLoader
import itertools
import math
from ..data.read_data import get_dataloaders

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer) -> list[list, list]: 
    """Trains a model and returns a list of the training and validation losses over training."""
    # Decide whether to use GPU or CPU
    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.mps.is_available():
    #    device = "mps"
    else:
        device = "cpu"

    losses = [[],[]]
    # Training loop
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        model.train()  # put model in training mode
        total_train_loss = 0

        # Training step 
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()          # reset gradients
            outputs = model(images)        # forward pass
            loss = loss_fn(outputs, labels) # compute loss
            loss.backward()                # compute gradients
            optimizer.step()               # update weights

            total_train_loss += loss.item()

        losses[0].append(total_train_loss / len(train_loader))

        # Validation step --> testing 
        model.eval()  # put model in evaluation mode
        total_val_loss = 0

        with torch.no_grad():  # no gradients needed for validation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

        losses[1].append(total_val_loss / len(val_loader))
        print(f"Training loss: {losses[0][-1]:.4f}")
        print(f"Validation loss: {losses[1][-1]:.4f}")
        
    return losses
        
# Sucessive halving

def _get_params(parameters: dict) -> tuple[dict, dict, dict]:
    """Gets the hyperparameters for the model and optimizer"""
    model_params = dict()
    optim_params = dict()
    other = dict()
    for key, value in parameters.items():
        if key.startswith("model_"):
            model_params[key[6:]] = value
        elif key.startswith("optim_"):
            optim_params[key[6:]] = value
        else:
            other[key] = value
    return (model_params, optim_params, other)

def _train_model(model, training_dataset, budget, loss_fun, optimizer):
    device = str(next(model.parameters()).device)
    for epoch in range(budget):
        model.train()  # put model in training mode

        # Training step 
        for images, labels in training_dataset:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()          # reset gradients
            outputs = model(images)        # forward pass
            loss = loss_fun(outputs, labels) # compute loss
            loss.backward()                # compute gradients
            optimizer.step()               # update weights

def _val_model(model, validation_dataset, loss_fun) -> float:
    device = str(next(model.parameters()).device)
    model.eval()  # put model in evaluation mode
    total_val_loss = 0

    with torch.no_grad():  # no gradients needed for validation
        for images, labels in validation_dataset:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            loss = loss_fun(outputs, labels)
            total_val_loss += loss.item()
    return total_val_loss

# Source: https://arxiv.org/abs/1502.07943
def sucessive_halving(model_class: nn.Module, 
                      epochs: int, 
                      hyper_parameters: dict,
                      checkpoint_folder: str):
    """Applies the sucessive halving algorithm to find the best hyper parameters efficiently."""
    # gets all of the possible combinations of hyper parameters
    combinations = list(itertools.product(*list(hyper_parameters.values())))
    n = len(combinations)
    if not (n > 0 and n.bit_count() == 1):
        print("the total number of combinations must be a power of 2")
        return None
    results = [[] for _ in range(n)]
    in_race = [True for _ in range(n)] # which parameter combinations are still active
    budgets =  [math.ceil(epochs/(2**(i))) for i in range(math.ceil(math.log2(n)), -1, -1)]
    model_checkpoints = [f"{checkpoint_folder}/model_{idx}.pth" for idx in range(n)]
    # loop through all of the filtering rounds
    for k in range(int(math.log2(n)) + 1):
        budget = budgets[k] if k == 0 else budgets[k] - budgets[k-1]
        # loop through all of the combinations that are still being tested
        for model_idx in range(n):
            if not in_race[model_idx]:
                # sentinel value -1 assumes loss function is always non-negative
                results[model_idx].append(-1)
                continue
            # load in the model if it needs to continue training
            combo = dict(zip(hyper_parameters.keys(), combinations[model_idx]))
            model_params, optim_params, other = _get_params(combo)
            model = model_class(**model_params)
            optimizer = other["optimizer"](model.parameters(), **optim_params)
            loss = other["loss"]()
            training_dataset, validation_dataset, _ = get_dataloaders(other["preprocessor"])
            # load in saved model and optimizer states
            if k > 0:
                checkpoint = torch.load(model_checkpoints[model_idx])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # compile the model to boost performance
            model = torch.compile(model)
            # continue training
            print(f"---training model_{model_idx} on {budget} epocks---")
            _train_model(model, training_dataset, budget, loss, optimizer) 
            # save model performance after training
            results[model_idx].append(_val_model(model, validation_dataset, loss))
            # update checkpoint
            checkpoint = {
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_checkpoints[model_idx])
            print(f"---Finished Model {model_idx} on {k} Current loss: {results[model_idx][-1]}---")
        if k == int(math.log2(n)):
            continue
        # pick which models will not continue
        n_eliminated = sum(in_race) // 2
        current_results = [i[-1] for i in results]
        for _ in range(n_eliminated):
            worst = current_results.index(max(current_results))
            current_results[worst] = -1
            in_race[worst] = False
    
    final_results = [i[-1] for i in results]
    best_model = final_results.index(max(final_results))
    print(f"Best Model: model_{best_model}\n  Error:{final_results[best_model]}")
    


