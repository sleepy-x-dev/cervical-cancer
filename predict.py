import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.dataset import test_transforms
from src.model import SwinDANN
from src.explain import generate_gradcam

def run_prediction(image_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SwinDANN().to(device)
    weights_path = 'weights/swin_dann_deep_finetuned.pth'
    
    if not os.path.exists(weights_path):
        print(f" Error: Could not find {weights_path}. Run advanced_model.py first!")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("Final Deep Fine-Tuned Model Loaded.")

    # 3. Preprocess Image
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(device) # Add batch dimension

    # 4. Predict
    with torch.no_grad():
        class_preds, _ = model(img_tensor, alpha=0.0)
        probs = F.softmax(class_preds, dim=1)
        conf, pred_class = torch.max(probs, 1)

    class_names = ["NORMAL", "ABNORMAL"]
    print(f"PREDICTION ")
    print(f"Result: {class_names[pred_class.item()]}")
    print(f"Confidence: {conf.item()*100:.2f}%")
    print(f"---------------------\n")

    
    os.makedirs('output', exist_ok=True)
    save_path = f"output/prediction_heatmap.png"
    
    # We ask Grad-CAM to explain the prediction it just made
    generate_gradcam(model, img_tensor, image_path, save_path, target_class=pred_class.item())
    print(f"Heatmap saved to: {save_path}")

if __name__ == "__main__":
    test_image = r"c:\Users\KIIT\Downloads\153354589-153354597-001.bmp" 
    run_prediction(test_image)