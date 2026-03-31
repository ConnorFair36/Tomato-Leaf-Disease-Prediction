# Need to use pytorch
# indent needed 
import torch
from torch.utils.data import DataLoader

def train_model(model,
                train_data,
                val_data,
                epochs,
                learning_rate,
                loss_fn,
                optimizer_class,
                batch_size=32): # check batch sixe before use 

    # Decide whether to use GPU or CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    # Turn datasets into DataLoader objects
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Optimizer for the learning rate 
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

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

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step --> testing 
        model.eval()  # put model in evaluation mode
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # no gradients needed for validation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total

    return model

