# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
from torch._C import device
import numpy as np
import pytorch_ssim
from pytorch_msssim import ssim, ms_ssim
from piqa import ssim
import torch
from datetime import datetime
import datetime


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


crit = SSIMLoss()


def train(data_loader, model, optimizer, device):

    model.train()
    batch_MSEs = []
    for data in tqdm(data_loader):
        # t1 = datetime.datetime.now()
        inputs = data[0]
        targets = data[1]
        labels = data[2]
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs)

        # "PSNR"
        # mse = nn.MSELoss()(outputs, targets)
        # psnr = 10 * torch.log10(1 / mse)

        "SmoothL1Loss"
        loss = torch.nn.SmoothL1Loss()(outputs, targets)
        batch_MSEs.append(loss.item())
        loss.backward()
        optimizer.step()
        # t2 = datetime.datetime.now()
        # print(t2 - t1)
        # print(loss.item())
    batch_MSEs = np.array(batch_MSEs)
    epoch_loss = np.mean(batch_MSEs)
    print(epoch_loss)
    return epoch_loss


def evaluate(data_loader, model, device):

    print("_____________validation_____________")
    model.eval()

    batch_MSEs = []
    with torch.no_grad():
        for idx, data in enumerate(data_loader, 1):
            # t1 = datetime.datetime.now()
            inputs = data[0]
            targets = data[1]
            labels = data[2]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            model.to(device)
            outputs = model(inputs)

            "SmoothL1Loss"
            loss = torch.nn.SmoothL1Loss()(outputs, targets)

            batch_MSEs.append(loss.item())

        batch_MSEs = np.array(batch_MSEs)
        epoch_loss = np.mean(batch_MSEs)
        print("epoch_loss", epoch_loss)
        # print("psnr_value", psnr)
        # print("ssim_value", ssim)

    return epoch_loss

