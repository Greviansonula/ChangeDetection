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
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_A_folder = os.path.join(root_dir, 'A')
        self.image_B_folder = os.path.join(root_dir, 'B')
        self.seg_label_folder = os.path.join(root_dir, 'Seg_label')
        self.ide_label_folder = os.path.join(root_dir, 'Ide_label')

        self.image_filenames = os.listdir(self.image_A_folder)
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_A_path = os.path.join(self.image_A_folder, self.image_filenames[idx])
        image_B_path = os.path.join(self.image_B_folder, self.image_filenames[idx])
        seg_label_path = os.path.join(self.seg_label_folder, self.image_filenames[idx].replace('.tif', '.xml'))

        # Load image A and B
        image_A = Image.open(image_A_path).convert('RGB')
        image_B = Image.open(image_B_path).convert('RGB')

        # Load segmentation label
        tree = ET.parse(seg_label_path)
        root = tree.getroot()
        seg_label = root.find('object').find('name').text

        # Convert segmentation label to tensor
        seg_label = torch.tensor(int(seg_label), dtype=torch.long)

        return image_A, image_B, seg_label