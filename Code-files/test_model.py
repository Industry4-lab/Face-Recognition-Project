
import torch
from torchvision import models
import torchvision.transforms as transforms 
import torch.nn as nn
from PIL import Image

num_classes = 22
transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_model.pth"))
model.to(device)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

def predict_image(image_path, classes):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]
