import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
import os
from PIL import Image
from torch.utils.data import Dataset
from unet import UNet
from data_load import ChangeDetectionDataset



device = select_device('')  # Use an empty string to automatically select a device
yolo_model = attempt_load('weights/yolov5s.pt')  # Adjust the path to your pre-trained model
yolo_model.eval()

in_ch = 3  # Number of input channels (e.g., RGB images)
out_ch = 1  # Number of output channels (e.g., binary mask)

unet_model = UNet(in_ch, out_ch)  # Adjust the input and output channels as per your requirements

loss_fn = nn.CrossEntropyLoss()  # Example loss function, adjust as per your requirements

optimizer = optim.Adam(unet_model.parameters(), lr=0.001)  # Example optimizer, adjust as per your requirements

batch_size = 16
num_epochs = 10

transform = ToTensor()  # Adjust the transformation based on your data requirements
train_dataset = ChangeDetectionDataset('../data_dir')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels = batch  # Adjust this based on your data format

        # Extract features using YOLOv5
        with torch.no_grad():
            yolo_outputs = yolo_model(images)

        # Pass the extracted features through UNet
        unet_outputs = unet_model(yolo_outputs)

        # Compute the loss
        loss = loss_fn(unet_outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_dataset = ChangeDetectionDataset('data_detect', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1)

unet_model.eval()  # Set the UNet model to evaluation mode
for batch in test_loader:
    images, labels = batch  # Adjust this based on your data format

    with torch.no_grad():
        # Extract features using YOLOv5
        yolo_outputs = yolo_model(images)

        # Pass the extracted features through UNet
        unet_outputs = unet_model(yolo_outputs)

        


