from pathlib import Path

import torch
from torchvision.datasets import CIFAR10


def main():
    # Project directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading CIFAR-10 dataset...")

    # Load CIFAR-10 training and test sets
    train_dataset = CIFAR10(
        root=raw_dir,
        train=True,
        download=True
    )

    test_dataset = CIFAR10(
        root=raw_dir,
        train=False,
        download=True
    )

    # Convert labels to tensors
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)

    # Store raw arrays and labels
    train_data = {
        "images": train_dataset.data,
        "labels": train_labels
    }

    test_data = {
        "images": test_dataset.data,
        "labels": test_labels
    }

    # Save raw dataset files
    torch.save(train_data, raw_dir / "train.pt")
    torch.save(test_data, raw_dir / "test.pt")

    print("CIFAR-10 preparation complete.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Saved training data to: {raw_dir / 'train.pt'}")
    print(f"Saved test data to: {raw_dir / 'test.pt'}")


if __name__ == "__main__":
    main()