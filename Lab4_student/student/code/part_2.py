import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from part_1 import (
    train_model_with_history,
    get_device
)

def create_basic_model(num_classes=10):
    """
    TODO: Create a simple 2D CNN model with Conv2d, ReLU, MaxPool2d, Flatten, and Linear layers.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    input_linear_layer = 32 * 16 * 16

    model = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(in_features=input_linear_layer, out_features=num_classes)
    )
    return model
    
def create_lenet_model(num_classes=10):
    """
    TODO: Create LeNet architecture with two Conv2d layers, AvgPool2d, and three fully connected layers.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        
        nn.Flatten(),
        
        nn.Linear(in_features=400, out_features=120),
        nn.ReLU(),
        
        nn.Linear(in_features=120, out_features=84),
        nn.ReLU(),
        
        nn.Linear(in_features=84, out_features=num_classes)
    )
    return model


def create_alexnet_model(num_classes=10):
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),

        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),

        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),

        nn.Linear(in_features=4096, out_features=num_classes)
    )
    
    return model

def create_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])
    train_set = datasets.CIFAR10(
        root="../dataset",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.CIFAR10(
        root="../dataset",
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model):
    print("\n--- Training Conv2d Model on CIFAR-10 ---")
    train_loader, test_loader = create_data_loaders(batch_size=64)

    device = get_device()
    model=model.to(device)

    model = train_model_with_history(
        model,
        train_loader,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=0.001,
        momentum=0.9,
        epochs=10,
        flatten_input=False
    )

def main():
    model = create_basic_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)

    model = create_lenet_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)

    model = create_alexnet_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)


if __name__ == "__main__":
  main()
   