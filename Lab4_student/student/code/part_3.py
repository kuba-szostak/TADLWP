import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from part_1 import train_model_with_history, get_device

class EdgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create a Conv2d layer with single input and output channel, and store it as self.conv.
        """
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        TODO: Apply the convolutional layer to the input and return the result.
        """
        return self.conv(x)

class EdgeDataset(Dataset):
    def __init__(self, cifar_subset, kernel):
        """
        TODO: Store the cifar_subset and kernel as instance variables (self.dataset and self.kernel).
        """
        self.dataset = cifar_subset
        self.kernel = kernel

    def __len__(self):
        """
        TODO: Return the length of the underlying dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        TODO: Get the input image x from the dataset. Compute target y by applying conv2d with the kernel to x (use padding=1).
        Return the tuple (x, y).
        """
        x, _ = self.dataset[idx]
        x_batched = x.unsqueeze(0)

        with torch.no_grad():
          y_batched = F.conv2d(x_batched, self.kernel, padding = 1)

        y = y_batched.squeeze(0)

        return x, y

def vertical_edge_kernel():
    return torch.tensor([
        [-1.0,  0.0,  1.0],
        [-1.0,  0.0,  1.0],
        [-1.0,  0.0,  1.0],
    ]).view(1, 1, 3, 3)
    

def create_edge_dataloader(batch_size=32, subset_size=512):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    cifar = datasets.CIFAR10(
        root="../dataset",
        train=True,
        download=True,
        transform=transform
    )

    subset_train = Subset(cifar, range(subset_size))
    subset_val = Subset(cifar, range(subset_size, 2*subset_size))
    kernel = vertical_edge_kernel()

    dataset_train = EdgeDataset(subset_train, kernel)
    dataset_val = EdgeDataset(subset_val, kernel)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def show_kernel(kernel, title):
    k = kernel.detach().cpu().numpy()[0, 0]
    plt.imshow(k, cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.show()


def show_edges(model, dataloader, num_images=5):
    device = next(model.parameters()).device
    model.eval()

    x, _ = next(iter(dataloader))
    x = x[:num_images].to(device)

    with torch.no_grad():
        edges = model(x)

    x = x.cpu()
    edges = edges.cpu()

    fig, axes = plt.subplots(num_images, 2, figsize=(5, 2 * num_images))

    for i in range(num_images):
        axes[i, 0].imshow(x[i, 0], cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(edges[i, 0], cmap="gray")
        axes[i, 1].set_title("Detected Edges")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def train_model(model):
    train_loader, val_loader = create_edge_dataloader(batch_size=64)
    
    device = get_device()
    model=model.to(device)

    history = train_model_with_history(
        model,
        train_loader,
        val_loader=val_loader,
        criterion=nn.MSELoss(),
        device=device,
        learning_rate=0.01,
        momentum=0.9,
        epochs=30,
        flatten_input=False
    )

    print("\nLearned kernel:")
    show_kernel(model.conv.weight, "Learned Kernel")

    print("\nExpected kernel:")
    show_kernel(vertical_edge_kernel(), "Target Kernel")

    print("\nExample edge detection results:")
    show_edges(model, train_loader)

    return model

def main():
    model = EdgeDetector()
    print(f"\nModel architecture:\n{model}")
    train_model(model)


if __name__ == "__main__":
    main()