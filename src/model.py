import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class GradientReversalFn(torch.autograd.Function):
    """
    The magic behind DANN. 
    Forward pass: Does nothing (acts as an identity function).
    Backward pass: Multiplies the gradient by -alpha, reversing the optimization direction.
    This is called Gradient Reversal Layer
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SwinDANN(nn.Module):
    def __init__(self):
        super(SwinDANN, self).__init__()
        
        # 1. Feature Extractor: Pre-trained Swin-Tiny
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.num_features = swin.head.in_features  # Usually 768 for swin_t
        
        # Remove the classification head, leaving just the feature extractor
        swin.head = nn.Identity()
        self.feature_extractor = swin
        
        # 2. Class Classifier (Predicts: Normal vs Abnormal)
        self.class_classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Output size 2 for CrossEntropyLoss
        )
        
        # 3. Domain Discriminator (Predicts: SIPaKMeD vs Herlev)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Output size 2 for CrossEntropyLoss
        )

    def forward(self, x, alpha=1.0):
        # Pass image through Swin Transformer
        features = self.feature_extractor(x)
        
        class_output = self.class_classifier(features)
        
        reverse_features = GradientReversalFn.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output