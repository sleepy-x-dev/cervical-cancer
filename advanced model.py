import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.dataset import CervicalDataset, train_transforms, test_transforms
from src.model import SwinDANN

HERLEV_PATH = r"C:\Users\KIIT\.cache\kagglehub\datasets\yuvrajsinhachowdhury\herlev-dataset\versions\1"

def main():
    print("Loading datasets for Deep Fine-Tuning...")
    
    dataset_train_augmented = CervicalDataset(HERLEV_PATH, 'herlev', transform=train_transforms)
    dataset_test_clean = CervicalDataset(HERLEV_PATH, 'herlev', transform=test_transforms)
    
    dataset_size = len(dataset_train_augmented)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, test_indices = indices[:split], indices[split:]
    
    train_loader = DataLoader(Subset(dataset_train_augmented, train_indices), batch_size=16, shuffle=True)
    test_loader = DataLoader(Subset(dataset_test_clean, test_indices), batch_size=16, shuffle=False)

    model = SwinDANN()
    # Load the pure, augmented base weights to start deep fine-tuning
    weights_path = 'weights/swin_dann_augmented_final.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        print("Base weights loaded successfully.")
    else:
        print("Could not find weights.")
        return

    print("\nUNFREEZING the Swin Backbone for Deep Fine-Tuning...")
    for param in model.feature_extractor.parameters():
        param.requires_grad = True # UNLOCKED
        
    # THE FIX: Differential Learning Rates
    # Backbone learns 10x slower to protect Grad-CAM shapes. Head learns normally.
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-5}, 
        {'params': model.class_classifier.parameters(), 'lr': 1e-4}
    ])
    
    class_weights = torch.tensor([3.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Starting Deep Fine-Tuning (15 Epochs)...")
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            class_preds, _ = model(images, alpha=0.0) 
            loss = criterion(class_preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/15], Loss: {total_loss/len(train_loader):.4f}")

    print("\nDeep Fine-tuning complete!")
    torch.save(model.state_dict(), 'weights/swin_dann_deep_finetuned.pth')
    
if __name__ == "__main__":
    main()