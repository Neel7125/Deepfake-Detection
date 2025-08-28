import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the DeepFakeDetector model class from training
class DeepFakeDetector(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        self.inceptionv3 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        num_ftrs = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.inceptionv3(x)

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model(checkpoint_path, device):
    model = DeepFakeDetector()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)

def predict(image_path, model, device, class_names):
    image = Image.open(image_path).convert("RGB")
    print(type(image))
    image = transform(image).unsqueeze(0).to(device)

    image_converted = image.squeeze(0).cpu()  # Shape: [3, 299, 299]
    image_converted = image_converted.permute(1, 2, 0)  # Shape: [299, 299, 3]
    image_converted = image_converted.numpy()
    
    image_converted = np.clip(image_converted, 0, 1)

    plt.imshow(image_converted)
    plt.savefig("Temp.png")

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        print(outputs)
        print(probs)

        _, predicted = torch.max(probs, 1)
        predicted_class = class_names[predicted.item()]
        confidence = probs[0][predicted.item()].item()
        return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake Prediction")
    parser.add_argument("image", help="Path to the image")
    parser.add_argument("--model", default="best_model.pth", help="Path to the .pth model file")
    parser.add_argument("--classes", nargs="+", default=["manipulated", "original"], help="Class names")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    
    pred_class, conf = predict(args.image, model, device, args.classes)
    print(f"Predicted: {pred_class} (Confidence: {conf:.2f})")
