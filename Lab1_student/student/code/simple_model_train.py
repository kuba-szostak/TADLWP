import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from experiment_logger import Experiment


def train_model(model, dataloader, num_training_iterations, is_classification=True, lr=0.1):
    experiment = Experiment()

    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for training_iteration in range(num_training_iterations):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        if training_iteration == 0:
            train_X = []
            train_y = []
        for X, y in dataloader:
            if training_iteration == 0:
                train_X.append(X)
                train_y.append(y)

            optimizer.zero_grad()
            outputs = model(X)
            if is_classification:
                loss = criterion(outputs, y)
            else:
                loss = criterion(outputs.squeeze(), y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        if training_iteration == 0:
            train_X = torch.cat(train_X)
            train_y = torch.cat(train_y)
            experiment.save_npy_array('train_X', train_X.reshape(-1, train_X.shape[-1]).numpy())
            if is_classification:
                experiment.save_npy_array('train_y', F.one_hot(train_y.flatten()).numpy())
            else:
                experiment.save_npy_array('train_y', train_y.reshape(-1, 1) if len(train_y.shape)==1 else y)

        train_loss /= len(dataloader)
        train_acc = train_correct / train_total

        experiment.save_metadata_entry('train_loss', train_loss)
        if is_classification:
            experiment.save_metadata_entry('train_acc', train_acc)
        if is_classification:
            if isinstance(model, nn.Sequential):
                model_for_visualization = list(model.children())
            else:
                model_for_visualization = [model]
            model_for_visualization.append(nn.Softmax())
            model_for_visualization = nn.Sequential(*model_for_visualization)
        else:
            model_for_visualization = model
        experiment.save_torch_model_sequential('model', model_for_visualization)
        experiment.next_step()

        if is_classification:
            print(
                f"Step [{training_iteration+1}/{num_training_iterations}] "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f} | "
            )
        else:
            print(
                f"Step [{training_iteration+1}/{num_training_iterations}] "
                f"Train Loss: {train_loss:.4f} | "
            )


def evaluate(experiment, dataloader, model):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    criterion = nn.CrossEntropyLoss()

    for X, y in dataloader:
        experiment.save_npy_array('train_X', X.reshape(-1, X.shape[-1]))
        experiment.save_npy_array('train_y', F.one_hot(y.flatten()).numpy())
        outputs = model(X)
        loss = criterion(outputs, y)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()

    train_loss /= len(dataloader)
    train_acc = train_correct / train_total

    experiment.save_metadata_entry('train_loss', train_loss)
    experiment.save_metadata_entry('train_acc', train_acc)
    model_for_visualization = list(model.children())
    model_for_visualization.append(nn.Softmax())
    model_for_visualization = nn.Sequential(*model_for_visualization)

    experiment.save_torch_model_sequential('model', model_for_visualization)
    experiment.next_step()

    print(
        f"Step [{experiment.step}] "
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f} | "
    )
