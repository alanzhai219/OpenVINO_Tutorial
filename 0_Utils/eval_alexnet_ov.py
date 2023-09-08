import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from openvino.runtime import Core
# from model_hack.alexnet import AlexNet
# alexnet_model = AlexNet()
# alexnet_model.load_state_dict(torch.load('alexnet-model.pth'))
ie = Core()

# from model_hack.alexnet_srelu import AlexNet
# alexnet_model = AlexNet()
# 
# # Load the pre-trained AlexNet model
# alexnet_model.load_state_dict(torch.load('checkpoints/alexnet-model.pth'), strict=False)
# alexnet_model.eval()
ext_lib_path = "srelu_cpp/build/libopenvino_srelu_extension.so"
ie.add_extension(ext_lib_path)

alexnet_model = ie.compile_model("ov_model/onnx/alexnet_srelu_symbol.xml", "CPU")

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
'''
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
'''

top1_correct = 0
top5_correct = 0
total = len(val_loader)

with torch.no_grad():
    pbar = tqdm(total=total)
    for images, labels in val_loader:
        images_np = images.numpy()
        outputs_ov = alexnet_model(images_np)
        outputs_np = outputs_ov[0]
        outputs = torch.from_numpy(outputs_np)
        _, predicted = torch.max(outputs.data, 1)
        
        # 计算top-1准确率
        top1_correct += (predicted == labels).sum().item()
        
        # 计算top-5准确率
        _, top5_predicted = torch.topk(outputs, 5, dim=1)
        top5_correct += torch.sum(top5_predicted == labels.view(-1, 1)).item()
        
        pbar.update(len(labels))

top1_accuracy = top1_correct / total
top5_accuracy = top5_correct / total

print(f'Top-1 Accuracy: {top1_accuracy * 100:.2f}%')
print(f'Top-5 Accuracy: {top5_accuracy * 100:.2f}%')

