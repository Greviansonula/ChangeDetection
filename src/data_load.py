import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as ET

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class ChangeDetectionDataset(Dataset):
    def __init__(self, folder_A, folder_B, folder_ide_label, folder_seg_label, transform=None):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.folder_ide_label = folder_ide_label
        self.folder_seg_label = folder_seg_label
        self.transform = transform
        self.image_files_A = sorted(os.listdir(folder_A))
        self.image_files_B = sorted(os.listdir(folder_B))
        self.xml_files = sorted(os.listdir(folder_ide_label))
        self.tif_files = sorted(os.listdir(folder_seg_label))

    def __len__(self):
        return len(self.image_files_A)

    def __getitem__(self, idx):
        # Load image A
        image_path_A = os.path.join(self.folder_A, self.image_files_A[idx])
        image_A = Image.open(image_path_A).convert("RGB")

        # Load image B
        image_path_B = os.path.join(self.folder_B, self.image_files_B[idx])
        image_B = Image.open(image_path_B).convert("RGB")

        # Load XML file
        xml_path = os.path.join(self.folder_ide_label, self.xml_files[idx])
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract bounding box coordinates from XML
        bboxes = []
        for obj in root.iter("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])

        # Load segmentation label
        tif_path = os.path.join(self.folder_seg_label, self.tif_files[idx])
        seg_label = Image.open(tif_path).convert("L")

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            seg_label = self.transform(seg_label)

        return {"image_A": image_A, "image_B": image_B, "bboxes": bboxes, "seg_label": seg_label}