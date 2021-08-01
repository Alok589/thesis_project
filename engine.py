# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
from torch._C import device
from torch.optim import optimizer
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# from Dense_Unet import Dense_Unet
import pytorch_ssim
from pytorch_msssim import ssim, ms_ssim
from piqa import ssim
from torch.autograd.grad_mode import F
from collections import namedtuple
import torch
from torchvision import models
from torchvision.models.vgg import vgg16
from datetime import datetime
import datetime


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


crit = SSIMLoss()


def train(data_loader, model, optimizer, device):

    model.train()
    batch_MSEs = []
    # vgg_model = vgg16(pretrained=True).to(device)
    for data in tqdm(data_loader):
        # t1 = datetime.datetime.now()
        # remember, we have image and targets
        # in our dataset class
        inputs = data[0]
        targets = data[1]
        labels = data[2]
        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs)

        # "PSNR"
        # mse = nn.MSELoss()(outputs, targets)
        # psnr = 10 * torch.log10(1 / mse)

        "Huber_loss"
        loss = torch.nn.SmoothL1Loss()(outputs, targets)
        # loss = torch.abs(targets - outputs).mean()
        # loss = nn.BCELoss()(outputs, targets)

        batch_MSEs.append(loss.item())

        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # t2 = datetime.datetime.now()
        # print(t2 - t1)
        # print(loss.item())
    batch_MSEs = np.array(batch_MSEs)
    epoch_loss = np.mean(batch_MSEs)
    print(epoch_loss)
    return epoch_loss


def evaluate(data_loader, model, device):

    # """
    # This function does evaluation for one epoch
    # :param data_loader: this is the pytorch dataloader
    # :param model: pytorch model
    # :param device: cuda/cpu
    # """

    # put model in evaluation mode
    print("_____________validation_____________")
    model.eval()

    # init lists to store targets and outputs
    batch_MSEs = []
    # # we use no_grad context
    with torch.no_grad():
        # vgg_model = vgg16(pretrained=True).to(device)
        for idx, data in enumerate(data_loader, 1):
            # t1 = datetime.datetime.now()
            inputs = data[0]
            targets = data[1]
            labels = data[2]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            # do the forward step to generate prediction
            model.to(device)
            outputs = model(inputs)

            "Huber_loss"
            loss = torch.nn.SmoothL1Loss()(outputs, targets)
            # loss = torch.abs(targets - outputs).mean()

            # loss = nn.L1Loss()(outputs, targets)
            # loss = nn.BCELoss()(outputs, targets)

            batch_MSEs.append(loss.item())

        #         # return final output and final targets
        batch_MSEs = np.array(batch_MSEs)
        epoch_loss = np.mean(batch_MSEs)
        print("epoch_loss", epoch_loss)
        # print("psnr_value", psnr)
        # print("ssim_value", ssim)

    return epoch_loss

    # final_targets = []
    # final_outputs = []

    # with torch.no_grad():
    #     for data in data_loader:
    #         inputs = data[0]
    #         targets = data[1]
    #         labels = data[2]
    #         inputs = inputs.to(device, dtype=torch.float)
    #         targets = targets.to(device, dtype=torch.float)
    #         labels = labels.to(device, dtype=torch.float)
    #         output = model(inputs)
    #         targets = targets.detach().cpu().numpy().tolist()
    #         output = output.detach().cpu().numpy().tolist()
    #         final_targets.extend(targets)
    #         final_outputs.extend(output)
    # return final_outputs, final_targets
