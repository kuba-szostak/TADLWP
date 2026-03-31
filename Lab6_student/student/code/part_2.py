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

    def __init__(self, channels, cardinality=4):
        super().__init__()
        self.paths = nn.ModuleList([
            ConvBlock(channels, channels)
            for _ in range(cardinality)
        ])

    def forward(self, x):
        identity = x
        out = sum([path(x) for path in self.paths])
        return torch.relu(out + identity)

class SimpleResNeXt(nn.Module):
    def __init__(self, num_classes=10, cardinality=4):
        super().__init__()
        self.net = nn.Sequential(
             ConvBlock(3, 64),
             ResNeXtBlock(64, cardinality),
             ResNeXtBlock(64, cardinality),
             ClassifierHead(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

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