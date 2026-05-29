import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset import (
    CervicalDataset,
    train_transforms,
    test_transforms
)
from src.model import SwinDANN


HERLEV_PATH = r"C:\Users\KIIT\.cache\kagglehub\datasets\yuvrajsinhachowdhury\herlev-dataset\versions\1"


def main():
    print("Loading datasets for Deep Fine-Tuning...")

    # Dataset versions
    dataset_train_augmented = CervicalDataset(
        HERLEV_PATH,
        'herlev',
        transform=train_transforms
    )

    dataset_test_clean = CervicalDataset(
        HERLEV_PATH,
        'herlev',
        transform=test_transforms
    )

    # ==========================================================
    # FIX 1: Proper 80/20 split
    # ==========================================================
    dataset_size = len(dataset_train_augmented)
    indices = list(range(dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(np.floor(0.8 * dataset_size))

    train_indices = indices[:split]
    test_indices = indices[split:]

    print(f"\nDataset size: {dataset_size}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # DataLoaders
    train_loader = DataLoader(
        Subset(dataset_train_augmented, train_indices),
        batch_size=16,
        shuffle=True
    )

    test_loader = DataLoader(
        Subset(dataset_test_clean, test_indices),
        batch_size=16,
        shuffle=False
    )

    # ==========================================================
    # Model
    # ==========================================================
    model = SwinDANN()

    weights_path = 'weights/swin_dann_augmented_final.pth'

    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location='cpu',
                weights_only=True
            )
        )
        print("\nBase weights loaded successfully.")
    else:
        print(f"\nCould not find weights: {weights_path}")
        return

    # ==========================================================
    # FIX 2: Unfreeze backbone for deep fine-tuning
    # ==========================================================
    print("\nUNFREEZING Swin Backbone...")

    for param in model.feature_extractor.parameters():
        param.requires_grad = True

    # ==========================================================
    # FIX 3: Differential learning rates
    # ==========================================================
    optimizer = optim.Adam([
        {
            'params': model.feature_extractor.parameters(),
            'lr': 1e-5
        },
        {
            'params': model.class_classifier.parameters(),
            'lr': 1e-4
        }
    ])

    # ==========================================================
    # FIX 4: Compute REAL class weights (F3 fixed)
    # ==========================================================
    print("\nComputing class weights from training distribution...")

    train_labels = [
        dataset_train_augmented.labels[i]
        for i in train_indices
    ]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    print("Computed class weights:")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ==========================================================
    # Training
    # ==========================================================
    epochs = 35
    alpha = 0.1  # FIX 5: DANN enabled (F4 fixed)

    print(f"\nStarting Deep Fine-Tuning ({epochs} Epochs)...")
    print(f"DANN alpha = {alpha}")

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            # FIX 6: DANN active
            class_preds, _ = model(
                images,
                alpha=alpha
            )

            loss = criterion(
                class_preds,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

    # ==========================================================
    # Save model
    # ==========================================================
    save_path = 'weights/swin_dann_deep_finetuned.pth'

    torch.save(
        model.state_dict(),
        save_path
    )

    print("\nDeep Fine-tuning complete!")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()