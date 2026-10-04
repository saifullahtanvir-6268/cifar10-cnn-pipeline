from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


def load_params():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    return params["train"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CIFAR10CNN(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                num_filters,
                num_filters * 2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                num_filters * 2,
                num_filters * 4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    params = load_params()

    num_filters = params["num_filters"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    seed = params["seed"]

    set_seed(seed)

    processed_dir = Path("data/processed")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets...")

    train_data = torch.load(
        processed_dir / "train.pt",
        weights_only=False
    )

    val_data = torch.load(
        processed_dir / "val.pt",
        weights_only=False
    )

    train_dataset = TensorDataset(
        train_data["images"],
        train_data["labels"]
    )

    val_dataset = TensorDataset(
        val_data["images"],
        val_data["labels"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = CIFAR10CNN(
        num_filters=num_filters,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    model_path = model_dir / "model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_filters": num_filters,
            "dropout_rate": dropout_rate
        },
        model_path
    )

    history_df = pd.DataFrame(history)

    history_path = model_dir / "history.csv"
    history_df.to_csv(history_path, index=False)

    print()
    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()