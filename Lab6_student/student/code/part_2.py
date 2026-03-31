# part_2.py
import torch
import torch.nn as nn
from helpers.training_utils import get_device, evaluate, batch_metrics, visualize_experiment, get_dataloaders
from part_1 import train_model, ConvBlock, ClassifierHead
from helpers.draw_architecture_helper import print_and_draw_model_structure

"""
Part 2 — ResNeXt
"""

# -------------------------
# ResNeXt (grouped convolutions)
# -------------------------

class ResNeXtBlock(nn.Module):

    def __init__(self, channels, cardinality=4, bottleneck_width=16):
        """
        TODO: Create a ResNeXt block with a parallel convolution blocks (cardinality number of them)
        and a skip connection. ModuleList can be used to store the convolution blocks.
        """
        super().__init__()
        self.cardinality = cardinality
        self.paths = nn.ModuleList()
        for _ in range(cardinality):
          path = nn.Sequential(
            nn.Conv2d(channels, bottleneck_width, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(bottleneck_width, bottleneck_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(bottleneck_width, channels, kernel_size=1),
          )
          self.paths.append(path)


    def forward(self, x):
        identity = x
        out = sum([path(x) for path in self.paths])
        return torch.relu(out + identity)

class SimpleResNeXt(nn.Module):
    def __init__(self, num_classes=10, cardinality=4, bottleneck_width=16):
        """
        TODO: Create a SimpleResNeXt network with a convBlock at the start, two ResNeXt blocks and a classifier head.
        """
        super().__init__()
        raise NotImplementedError("SimpleResNeXt is not implemented")

    def forward(self, x):
        """
        TODO: Pass input through the SimpleResNeXt.
        """
        raise NotImplementedError("SimpleResNeXt is not implemented")

def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    histories = {"SimpleResNeXt": {"model": [], "history": []}}

    model = SimpleResNeXt()
    criterion = nn.CrossEntropyLoss()

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        lr=0.01,
        criterion=criterion,
        momentum=0.9,
        weight_decay=1e-4,
        epochs=8,
    )

    histories["SimpleResNeXt"]["model"].append(trained_model)
    histories["SimpleResNeXt"]["history"].append(history)

    print_and_draw_model_structure(
        trained_model,
        output_file=f"SimpleResNeXt_fx_graph",
        fmt="svg")

    


if __name__ == "__main__":
    main()