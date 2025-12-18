import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# ==========================================
# 1. MODEL DEFINITION (Must match Notebook)
# ==========================================

class CAB(nn.Module):
    """Channel Attention Block"""
    def __init__(self, in_channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GAB(nn.Module):
    """Global Attention Block"""
    def __init__(self, in_channels, reduction=16):
        super(GAB, self).__init__()
        # Global Attention typically captures spatial dependencies or global context
        # Here we implement a spatial attention mechanism often paired with channel attention
        # Reduction can be used if we want to squeeze channels first for efficiency
        
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [b, c, h, w]
        # attention: [b, 1, h, w]
        attn = self.conv_s(x)
        return x * attn

class CABNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CABNet, self).__init__()
        # Backbone: MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Feature extractor layers (features: [0]...[18])
        # Output channels of MobileNetV2 features is 1280
        self.features = self.backbone.features
        
        # Attention Modules
        self.cab = CAB(1280)
        self.gab = GAB(1280)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        
        # Apply CAB then GAB (or parallel, sequential is standard for CBAM-like)
        x = self.cab(x)
        x = self.gab(x)
        
        x = self.classifier(x)
        return x

# ==========================================
# 2. INFERENCE LOGIC
# ==========================================

def load_model(model_path, device):
    """Loads the trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    model = CABNet(num_classes=5)
    
    # Load state dict
    try:
        # Map location ensures it loads on CPU if GPU is missing
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Loads and preprocesses an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        sys.exit(1)
        
    return transform(image).unsqueeze(0) # Add batch dimension

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_model.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_path = "./models/ddr_cabnet_best.pth"
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Preprocess
    print("Processing image...")
    input_tensor = preprocess_image(image_path).to(device)
    
    # Infer
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    class_idx = predicted.item()
    confidence = probabilities[0][class_idx].item()
    
    print(f"\nPrediction Result:")
    print(f"==================")
    print(f"Class Index: {class_idx}")
    print(f"Confidence:  {confidence:.4f}")
    print(f"DR Level:    {['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][class_idx]} (DR{class_idx})")

if __name__ == "__main__":
    main()
