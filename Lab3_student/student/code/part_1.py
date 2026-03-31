import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Assuming helpers.training_utils is available in your environment
from helpers.training_utils import (
    load_unbalanced_mnist,
    divide_data_to_train_val_test,
    evaluate,
    setup_epoch_logging,
    log_batch_step,
    finalize_epoch_logging,
    log_epoch_metrics,
    create_experiment_with_config,
    finalize_training,
)

def create_model(input_size: int, num_classes: int) -> nn.Module:
    """
    Creates a feedforward neural network using PyTorch's Sequential module.
    We use 3 hidden layers with ReLU activations.
    """
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )
    return model


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """
    Converts numpy arrays to PyTorch tensors and wraps them in DataLoaders.
    """
    # 1. Convert numpy arrays to PyTorch tensors
    # X needs to be float32 for model weights, y needs to be long for CrossEntropyLoss
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 2. Wrap tensors in TensorDataset
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # 3. Create DataLoaders (only the training set needs to be shuffled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model_with_history(
    experiment: Any,
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    learning_rate: float = 0.02,
    momentum: float = 0.0,
    epochs: int = 100,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    model = model.to(device)
    
    for epoch in range(1, epochs + 1):
        verbose_epoch = epoch % 10 == 0 or epoch == epochs
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Ensure the model is in training mode (enables dropout/batchnorm if added later)
        model.train() 
        
        if verbose_epoch:
            hook_handles, parameters_from_epoch_start, old_parameters = setup_epoch_logging(model, experiment)
            
        for inputs, targets in train_loader:
            # Move data to the appropriate device (CPU or GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Zero out the gradients from the previous step
            optimizer.zero_grad()
            
            # 2. Forward pass: compute predictions
            outputs = model(inputs)
            
            # 3. Calculate the loss
            loss = criterion(outputs, targets)
            
            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # 5. Optimizer step: update the parameters
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            if verbose_epoch:
                old_parameters = log_batch_step(old_parameters, model, experiment)
                
        # Calculate epoch train loss and train accuracy
        train_loss = running_loss / total
        train_acc = correct / total
        
        if verbose_epoch:
            finalize_epoch_logging(hook_handles, parameters_from_epoch_start, old_parameters, experiment)
            
        # Validation evaluation is presumably handled correctly in your imported 'evaluate' function
        # Ensure 'evaluate' uses model.eval() and torch.no_grad() internally
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if verbose_epoch:
            print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            log_epoch_metrics(experiment, train_loss, val_loss, val_acc, model)
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
    return model, history


def create_basic_criterion(_, device=None):
    return nn.CrossEntropyLoss()

def empty_data_processor(X_train, y_train):
    return X_train, y_train


def run_experiment(create_data_loaders, 
                   criterion, 
                   data_processor=None, 
                   weight_initialization=None,
                   learning_rate=0.02,
                   batch_size=32,
                   momentum=0.0,
                   verbose=True):
    # Load and prepare data
    df = load_unbalanced_mnist()
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    class_names = label_encoder.classes_ 
    X_train, X_val, X_test, y_train, y_val, y_test = divide_data_to_train_val_test(df, test_size=0.2, val_size=0.1)
    X_train, y_train = (data_processor or empty_data_processor)(X_train, y_train)

    # Detect device first so criterion weights can be on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion(y_train, device=device)

    # Create data loaders and model
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)
    model = create_model(X_train.shape[1], len(class_names))
    
    if weight_initialization is not None:
        weight_initialization(model)

    # Setup experiment
    experiment = create_experiment_with_config(learning_rate, momentum, batch_size, weight_initialization)

    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    
    # Move model to device
    model = model.to(device)
    
    # Train the model
    model, history = train_model_with_history(experiment, model, train_loader, val_loader, criterion, learning_rate=learning_rate, momentum=momentum, epochs=50, device=device)
    
    # Evaluate and visualize results
    result = finalize_training(model, history, test_loader, criterion, class_names, "Part 1 training history", verbose)
    
    return result


def main():
    run_experiment(
        create_data_loaders,
        criterion=create_basic_criterion,
        data_processor=empty_data_processor,
        verbose=True
    )


if __name__ == "__main__":
    main()