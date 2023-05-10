import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet
from yolov5 import YOLOv5

class ChangeDetectionModel(nn.Module):
    def __init__(self, unet_backbone, yolo_backbone):
        super(ChangeDetectionModel, self).__init__()
        self.unet = UNet(unet_backbone)
        self.yolov5 = YOLOv5(yolo_backbone)

    def forward(self, input):
        unet_output = self.unet(input)
        yolo_output = self.yolov5(input)
        return unet_output, yolo_output

# Create an instance of your model
model = ChangeDetectionModel()

# Define your loss function(s)
loss_fn = nn.CrossEntropyLoss()  # Example loss function, adjust as per your requirements

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer, adjust as per your requirements

# Define your training parameters
batch_size = 16
num_epochs = 10

# Create your data loaders
train_dataset = ChangeDetectionDataset('data_train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels = batch  # Adjust this based on your data format

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing
test_dataset = ChangeDetectionDataset('data_detect')
test_loader = DataLoader(test_dataset, batch_size=1)

model.eval()  # Set the model to evaluation mode
for batch in test_loader:
    images = batch  # Adjust this based on your data format

    with torch.no_grad():
        # Perform inference
        outputs = model(images)
