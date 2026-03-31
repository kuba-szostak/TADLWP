import torch
import torch.nn as nn
from torch.nn.modules import linear
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset, dataloader
from helpers.training_utils import (
    set_random_seed, 
    divide_data_to_train_val_test,
    load_coffee_dataset
)
from helpers.experiment_logger import Experiment

def create_model(sequence_length: int = 100, num_filters: int = 3, num_classes: int = 5) -> nn.Sequential:
    pooled_length = sequence_length // 2
    linear_input_size = num_filters * pooled_length

    model = nn.Sequential(
      nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3, padding=1),      
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(in_features=linear_input_size, out_features=num_classes)
    )
    return model

def evaluate(model, loader, criterion, device, flatten_input=False):
    """
    TODO: Set model to eval mode and use torch.no_grad(). Iterate through loader, compute outputs and loss,
    accumulate metrics. Return average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
      for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)
            
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0) 
        
        predictions = outputs.argmax(1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
            
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return average_loss, accuracy

def train_model_with_history(model, train_loader, val_loader, criterion, device,
                             learning_rate=0.001, momentum=0.9, epochs=10,
                             flatten_input=False):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if flatten_input:
                x = x.view(x.size(0), -1)
                
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            predictions = outputs.argmax(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        epoch_val_loss, epoch_val_acc = evaluate(
            model, val_loader, criterion, device, flatten_input
        )
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
              
    return history


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device
    
def create_data_loaders(batch_size: int = 64, sequence_length: int = None, num_samples: int = None):
    # Load data
    X_train, y_train, X_test, y_test, sequence_length = load_coffee_dataset("../dataset/Coffee")

    # print(f"X train shape {X_train.shape}")

    # Train / validation split (helper requires DataFrame)
    df = pd.DataFrame(X_train.reshape(len(X_train), -1))
    df["class"] = y_train

    X_train, X_val, _, y_train, y_val, _ = divide_data_to_train_val_test(
        df, test_size=0.2, val_size=0.2
    )

    # Restore Conv1d shape
    X_train  = X_train.reshape(-1, 1, sequence_length)
    X_val = X_val.reshape(-1, 1, sequence_length)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    class_names = ['Robusta', 'Arabica']

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_names

def main():
    """Main function to run Part 1 training."""
    print("\n=== Lab 4 - Part 1: 1D Convolutional Neural Network ===")
    
    # Set random seed for reproducibility
    set_random_seed(42)

    # Create data loaders (will determine sequence_length and num_classes from dataset)
    train_loader, val_loader, test_loader, class_names = create_data_loaders(batch_size=8)
    
    # Get sequence_length from a sample
    sample_batch = next(iter(train_loader))
    sequence_length = sample_batch[0].shape[2]
    num_classes = len(class_names)

    print(f"\nDataset info:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Classes: {class_names}")
    
    device = get_device()
    # Create model
    model = create_model(
        sequence_length=sequence_length,
        num_filters=32,
        num_classes=num_classes
    )
    print(f"\nModel architecture:\n{model}")
    
    # Train the model
    model.to(device)

    print("\n--- Training Conv1d Model on 1D Time Series ---")
    model = train_model_with_history(model, 
                                train_loader, 
                                val_loader,
                                criterion=nn.CrossEntropyLoss(),
                                device=device,
                                learning_rate=0.001, 
                                momentum=0.9, 
                                epochs=15,
                                flatten_input=False)

    print("\nPart 1 completed!")


if __name__ == "__main__":
    main()
