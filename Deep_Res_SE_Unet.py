from skimage import transform
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import skimage.io
import numpy as np
import scipy.io as sio
import torch
from torch.optim import optimizer
from torch.nn.modules import BatchNorm2d

# from torchsummary import summary


class SELayer(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // reduction, in_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class residual_block(nn.Module):
    """(conv1 + relu => conv2 + relu => BN)"""

    def __init__(self, in_ch, out_ch):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.75)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn(x)
        return x


class conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.75)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class conv_bn2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.75)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class conv_transpose(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_transpose, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.75)

    def forward(self, x):
        x = self.conv_trans1(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class conv_transpose_noBN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_transpose_noBN, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_trans1(x)
        x = self.relu(x)

        return x


class out_img(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_img, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.75)
        # self.conv2 = nn.conv2d(out_ch, out_ch, kernel_size = 1, padding = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        # x = self.conv2(x)

        return x


class out_img2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_img2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2 = nn.conv2d(out_ch, out_ch, kernel_size = 1, padding = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)

        return x


class Deep_Res_SE_Unet(nn.Module):
    def __init__(self):
        super(Deep_Res_SE_Unet, self).__init__()

        self.conv1 = conv_bn(1, 32)
        self.residual_1 = residual_block(32, 32)
        self.senet_1 = SELayer(32, 32)

        self.conv2 = conv_bn2(32, 64)
        self.residual_2 = residual_block(64, 64)
        self.senet_2 = SELayer(64, 64)

        self.conv3 = conv_bn2(64, 128)
        self.residual_3 = residual_block(128, 128)
        self.senet_3 = SELayer(128, 128)

        self.conv4 = conv_bn2(128, 256)
        self.residual_4 = residual_block(256, 256)
        self.senet_4 = SELayer(256, 256)
        # Up_conv

        self.up_conv1 = conv_transpose(256, 128)
        self.conv_bn1 = conv_bn(256, 128)
        self.residual_5 = residual_block(128, 128)

        self.up_conv2 = conv_transpose(128, 64)
        self.conv_bn2 = conv_bn(128, 64)
        self.residual_6 = residual_block(64, 64)

        self.up_conv3 = conv_transpose_noBN(64, 32)
        self.conv_bn3 = conv_bn(64, 32)
        self.residual_7 = residual_block(32, 32)

        self.out1 = out_img(32, 64)
        self.out2 = out_img2(64, 1)

    def forward(self, x):

        # encoder
        "encoder-1"
        x1_1 = self.conv1(x)
        x2_1 = self.residual_1(x1_1)
        x3_1 = x2_1 + x1_1
        x4_1 = self.senet_1(x3_1)

        "encoder-2"
        x1_2 = self.conv2(x4_1)
        x2_2 = self.residual_2(x1_2)
        x3_2 = x2_2 + x1_2
        x4_2 = self.senet_2(x3_2)

        "encoder-3"
        x1_3 = self.conv3(x4_2)
        x2_3 = self.residual_3(x1_3)
        x3_3 = x2_3 + x1_3
        x4_3 = self.senet_3(x3_3)

        "encoder-4"
        x1_4 = self.conv4(x4_3)
        x2_4 = self.residual_4(x1_4)
        x3_4 = x2_4 + x1_4
        x4_4 = self.senet_4(x3_4)

        # decoder

        "decoder-1"
        y1_1 = self.up_conv1(x4_4)
        cat1 = torch.cat([x4_3, y1_1], 1)
        y2_1 = self.conv_bn1(cat1)
        y3_1 = self.residual_5(y2_1)
        y4_1 = y3_1 + y2_1

        "decoder-2"
        y1_2 = self.up_conv2(y4_1)
        cat2 = torch.cat([x4_2, y1_2], 1)
        y2_2 = self.conv_bn2(cat2)
        y3_2 = self.residual_6(y2_2)
        y4_2 = y3_2 + y2_2

        "decoder-3"
        y1_3 = self.up_conv3(y4_2)
        cat3 = torch.cat([x4_1, y1_3], 1)
        y2_3 = self.conv_bn3(cat3)
        y3_3 = self.residual_7(y2_3)
        y4_3 = y3_3 + y2_3

        "output_block"
        z1_4 = self.out1(y4_3)
        z2_4 = self.out2(z1_4)

        return z2_4


if __name__ == "__main__":
    model = Deep_Res_SE_Unet()
    device = "cuda:5"
    model.to(device)
    input_image = torch.rand(size=(1, 1, 128, 128))
    input_image = input_image.to(device)
    out = model(input_image)
    print(out.shape)

