import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as ET

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.label_filenames = sorted(os.listdir(os.path.join(root_dir, 'labels')))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_filenames[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.label_filenames[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Convert label to grayscale

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# class ChangeDetectionDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.image_dir = os.path.join(data_dir, "images")
#         self.label_dir = os.path.join(data_dir, "labels")
#         self.xml_dir = os.path.join(data_dir, "xml")

#         self.image_filenames = sorted(os.listdir(self.image_dir))
#         self.label_filenames = sorted(os.listdir(self.label_dir))
#         self.xml_filenames = sorted(os.listdir(self.xml_dir))

#         self.transform = ToTensor()

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, index):
#         image_filename = self.image_filenames[index]
#         label_filename = self.label_filenames[index]
#         xml_filename = self.xml_filenames[index]

#         image_path = os.path.join(self.image_dir, image_filename)
#         label_path = os.path.join(self.label_dir, label_filename)
#         xml_path = os.path.join(self.xml_dir, xml_filename)

#         image = Image.open(image_path).convert("RGB")
#         label = Image.open(label_path).convert("L")
#         xml_tree = ET.parse(xml_path)

#         sample = {
#             "image": self.transform(image),
#             "label": self.transform(label),
#             "xml_tree": xml_tree
#         }

#         return sample
