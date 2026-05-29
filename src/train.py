import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import cycle
import numpy as np

def train_dann(model, source_loader, target_loader, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Training on device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # FIX 1: Class Weights. Force the model to pay 2x more attention to "Normal" (Class 0) 
    # to stop it from just guessing "Abnormal" all the time.
    class_weights = torch.tensor([2.0, 1.0]).to(device)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    
    criterion_domain = nn.CrossEntropyLoss()
    
    model.to(device)
    total_batches = len(source_loader) * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        target_iter = iter(cycle(target_loader))
        pbar = tqdm(source_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (source_images, source_labels) in enumerate(pbar):
            target_images, _ = next(target_iter) 
            
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            
            optimizer.zero_grad()
            
            #Cap the Alpha. 
            current_batch = epoch * len(source_loader) + i
            p = current_batch / total_batches
            alpha = (2. / (1. + np.exp(-10 * p)) - 1.) * 0.3 
            
            # 1. Train on Source (Classification + Domain)
            class_preds_src, domain_preds_src = model(source_images, alpha=alpha)
            loss_class = criterion_class(class_preds_src, source_labels)
            
            domain_labels_src = torch.zeros(source_images.size(0), dtype=torch.long).to(device)
            loss_domain_src = criterion_domain(domain_preds_src, domain_labels_src)
            
            # 2. Train on Target (Domain ONLY)
            _, domain_preds_tgt = model(target_images, alpha=alpha)
            domain_labels_tgt = torch.ones(target_images.size(0), dtype=torch.long).to(device)
            loss_domain_tgt = criterion_domain(domain_preds_tgt, domain_labels_tgt)
            
            # 3. Combine and Backpropagate
            loss_domain = loss_domain_src + loss_domain_tgt
            total_loss = loss_class + loss_domain 
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'Class': f"{loss_class.item():.3f}", 
                'Dom': f"{loss_domain.item():.3f}",
                'Alpha': f"{alpha:.3f}" 
            })
            
    print("Training Complete!")
    return model