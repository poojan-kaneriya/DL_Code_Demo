# inference/predict.py

import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img)

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  # add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()
