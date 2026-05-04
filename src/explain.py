import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class SwinClassWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super(SwinClassWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # Return ONLY the cancer classification predictions
        class_preds, _ = self.base_model(x, alpha=0.0) 
        return class_preds

def generate_gradcam(model, image_tensor, original_image_path, save_path, target_class=1):
    wrapped_model = SwinClassWrapper(model)
    wrapped_model.eval()

    # Target the final Swin Transformer block for the heatmap
    target_layers = [wrapped_model.base_model.feature_extractor.features[-1][-1]]

    # Vision Transformers need their output reshaped to a 2D grid for Grad-CAM
    def reshape_transform(tensor, height=7, width=7):
        result = tensor.permute(0, 3, 1, 2)
        return result

    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Load and format the original image for the overlay
    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_float = np.float32(img) / 255

    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(cam_image)
    ax2.set_title(f"Grad-CAM (Target: Class {target_class})")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM saved successfully to: {save_path}")