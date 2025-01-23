
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, random_split 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])

full_dataset = ImageFolder(root="C:/Users/ksiri/OneDrive/Desktop/data augmentation", transform=transform)

train_size = int(0.8 * len(full_dataset)) 
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()

num_epochs = 10

for epoch in range(num_epochs): 
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(outputs, labels) 
        running_loss += loss.item()
        running_accuracy += accuracy
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {running_accuracy/len(train_loader):.4f}')

torch.save(model.state_dict(), "resnet50_model.pth")
