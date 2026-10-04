from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset


class CIFAR10CNN(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                num_filters,
                num_filters * 2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

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


def load_params():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    return params["train"]


def main():
    params = load_params()

    batch_size = params["batch_size"]

    processed_dir = Path("data/processed")
    model_dir = Path("models")

    print("Loading processed test set...")

    test_data = torch.load(
        processed_dir / "test.pt",
        weights_only=False
    )

    test_dataset = TensorDataset(
        test_data["images"],
        test_data["labels"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print("Loading trained model...")

    checkpoint = torch.load(
        model_dir / "model.pth",
        map_location=device,
        weights_only=False
    )

    model = CIFAR10CNN(
        num_filters=checkpoint["num_filters"],
        dropout_rate=checkpoint["dropout_rate"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

    test_loss = total_loss / total
    test_accuracy = correct / total

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False
    )

    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = model_dir / "confusion_matrix.png"

    plt.savefig(
        confusion_matrix_path,
        dpi=150
    )

    plt.close()

    print()
    print("Evaluation complete.")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Metrics saved to: metrics.json")
    print(
        f"Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()