import torch
import torch.nn as nn
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_unet_from_yaml(config):
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, input):
            return self.conv(input)

    class UNet(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet, self).__init__()

            self.conv1 = DoubleConv(in_ch, config['model']['conv1_out'])
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = DoubleConv(config['model']['conv1_out'], config['model']['conv2_out'])
            self.pool2 = nn.MaxPool2d(2)
            self.conv3 = DoubleConv(config['model']['conv2_out'], config['model']['conv3_out'])
            self.pool3 = nn.MaxPool2d(2)
            self.conv4 = DoubleConv(config['model']['conv3_out'], config['model']['conv4_out'])
            self.pool4 = nn.MaxPool2d(2)
            self.conv5 = DoubleConv(config['model']['conv4_out'], config['model']['conv5_out'])
            self.up6 = nn.ConvTranspose2d(config['model']['conv5_out'], config['model']['conv4_out'], 2, stride=2)
            self.conv6 = DoubleConv(config['model']['conv5_out'], config['model']['conv6_out'])
            self.up7 = nn.ConvTranspose2d(config['model']['conv6_out'], config['model']['conv3_out'], 2, stride=2)
            self.conv7 = DoubleConv(config['model']['conv6_out'], config['model']['conv7_out'])
            self.up8 = nn.ConvTranspose2d(config['model']['conv7_out'], config['model']['conv2_out'], 2, stride=2)
            self.conv8 = DoubleConv(config['model']['conv7_out'], config['model']['conv8_out'])
            self.up9 = nn.ConvTranspose2d(config['model']['conv8_out'], config['model']['conv1_out'], 2, stride=2)
            self.conv9 = DoubleConv(config['model']['conv8_out'], config['model']['conv9_out'])
            self.conv10 = nn.Conv2d(config['model']['conv9_out'], out_ch, 1)

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            c1 = self.conv1(x)
            p1 = self.pool1(c1)
            c2 = self.conv2(p1)
            p2 = self.pool2(c2)
            c3 = self.conv3(p2)
            p3 = self.pool3(c3)
            c4 = self.conv4(p3)
            p4 = self.pool4(c4)
            c5 = self.conv5(p4)
            up_6 = self.up6(c5)
            merge6 = torch.cat([up_6, c4], dim=1)
            c6 = self.conv6(merge6)
            up_7 = self.up7(c6)
            merge7 = torch.cat([up_7, c3], dim=1)
            c7 = self.conv7(merge7)
            up_8 = self.up8(c7)
            merge8 = torch.cat([up_8, c2], dim=1)
            c8 = self.conv8(merge8)
            up_9 = self.up9(c8)
            merge9 = torch.cat([up_9, c1], dim=1)
            c9 = self.conv9(merge9)
            c10 = self.conv10(c9)
            out = nn.Sigmoid()(c10)
            return out

    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    model = UNet(in_channels, out_channels)
    return model

config = read_yaml('unet_architecture.yaml')
model = create_unet_from_yaml(config)

