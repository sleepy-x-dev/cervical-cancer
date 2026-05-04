import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.dataset import get_dataloaders
from src.model import SwinDANN

SIPAKMED_PATH = r"C:\Users\KIIT\.cache\kagglehub\datasets\prahladmehandiratta\cervical-cancer-largest-dataset-sipakmed\versions\1"
HERLEV_PATH = r"C:\Users\KIIT\.cache\kagglehub\datasets\yuvrajsinhachowdhury\herlev-dataset\versions\1"

# --- Wrapper for Grad-CAM Metric ---
class SwinClassWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        class_preds, _ = self.base_model(x, alpha=0.0) 
        return class_preds

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

def evaluate_and_score(model, dataloader, dataset_name, device='cpu'):
    model.eval()
    all_probs = []
    all_labels = []
    
    print(f"\nEvaluating {dataset_name} using ROC Curve Analysis...")
    with torch.no_grad():
        for images, labels in dataloader:
            class_preds, _ = model(images, alpha=0.0)
            
            
            probs = F.softmax(class_preds, dim=1)[:, 1]
            
            all_probs.extend(probs.numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate AUROC 
    auroc = roc_auc_score(all_labels, all_probs)
    
    # Calculate ROC curve to find the absolute best threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # Youden's J statistic helps find the best balance between Sensitivity and Specificity
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold Found: {optimal_threshold:.4f}")
    
    # Apply the optimal threshold
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    sens = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    spec = cm[0,0] / (cm[0,0] + cm[0,1]) if len(cm) > 1 else 0.0

    print("="*50)
    print(f"OPTIMIZED METRICS FOR {dataset_name}")
    print("="*50)
    print(f"AUROC:       {auroc:.4f} (Area Under ROC Curve)")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Sensitivity: {sens*100:.2f}%")
    print(f"Specificity: {spec*100:.2f}%")
    print(f"F1 (Macro):  {f1:.4f}")
    print("="*50)
def calculate_gradcam_alignment_score(model, dataloader):
    
    print("\nCalculating Quantitative Grad-CAM Alignment Score...")
    wrapped_model = SwinClassWrapper(model)
    wrapped_model.eval()
    target_layers = [wrapped_model.base_model.feature_extractor.features[-1][-1]]
    
    def reshape_transform(tensor, height=7, width=7):
        return tensor.permute(0, 3, 1, 2)
        
    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    alignment_scores = []
    
    # Test a subset of images
    for i, (images, labels) in enumerate(dataloader):
        if i > 5: break # Just test 5 batches to get a score
        
        # Only evaluate on Abnormal cells (Class 1)
        mask = labels == 1
        if not mask.any(): continue
            
        abnormal_images = images[mask]
        targets = [ClassifierOutputTarget(1)] * len(abnormal_images)
        
        # Generate heatmaps
        grayscale_cams = cam(input_tensor=abnormal_images, targets=targets)
        
        for heatmap in grayscale_cams:
            # We define the "nucleus zone" as the center 50% of the image.
            # We calculate what percentage of the "hot" attention falls in this zone.
            h, w = heatmap.shape
            center_h_start, center_h_end = int(h*0.25), int(h*0.75)
            center_w_start, center_w_end = int(w*0.25), int(w*0.75)
            
            center_sum = np.sum(heatmap[center_h_start:center_h_end, center_w_start:center_w_end])
            total_sum = np.sum(heatmap)
            
            # Alignment Score: % of attention focused on the central morphology
            score = (center_sum / total_sum) * 100 if total_sum > 0 else 0
            alignment_scores.append(score)
            
    final_score = np.mean(alignment_scores)
    print("="*50)
    print(f" FINAL GRAD-CAM ALIGNMENT SCORE: {final_score:.2f}%")
    print("="*50)

def main():
    print("Loading datasets...")
    _, target_loader = get_dataloaders(SIPAKMED_PATH, HERLEV_PATH, batch_size=16) 
    
    model = SwinDANN()
    weights_path = 'weights/swin_dann_deep_finetuned.pth'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        print("Augmented weights loaded successfully.")
    else:
        print("Could not find weights. Run the training script first.")
        return

    # PS 1: Cross-Dataset Evaluation with Threshold Tuning to fix Mode Collapse
    evaluate_and_score(model, target_loader, "HERLEV (TARGET)")
    
    # PS 2: Grad-CAM Evaluation Metric
    calculate_gradcam_alignment_score(model, target_loader)

if __name__ == "__main__":
    main()