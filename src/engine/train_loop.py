# Need to use pytorch
# indent needed 
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

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
        


