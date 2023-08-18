import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the AlexNet model architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the pre-trained AlexNet model
alexnet_model = AlexNet()
alexnet_model.load_state_dict(torch.load('alexnet-model.pth'))
alexnet_model.eval()

# Load validation dataset from image_folder and val.txt
dataset = "/home/xiuchuan/dataset/miniILSVRC2012/"
data_dir = dataset
val_txt_path = dataset + "val_list.txt"

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_txt_path, transform=None):
        self.data_dir = data_dir
        self.data_transform = transform
        self.data = []

        with open(val_txt_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                img_filename, label = line.split()
                label = int(label)
                self.data.append((img_filename, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, label = self.data[idx]
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        if self.data_transform:
            image = self.data_transform(image)

        return image, label

# Load validation dataset
val_dataset = CustomDataset(data_dir, val_txt_path, data_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = alexnet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
