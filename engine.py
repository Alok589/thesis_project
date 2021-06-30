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


# class Vgg16(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg16, self).__init__()
#         vgg_pretrained_features = models.vgg16(pretrained=True).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h
#         vgg_outputs = namedtuple(
#             "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
#         )
#         out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
#         return out


# vgg_loss = vgg16()

# LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# # # https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
# class LossNetwork(torch.nn.Module):
#     def __init__(self, vgg_model):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg_model.features
#         self.layer_name_mapping = {
#             "3": "relu1_2",
#             "8": "relu2_2",
#             "15": "relu3_3",
#             "22": "relu4_3",
#         }

#     def forward(self, x):
#         output = {}
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output[self.layer_name_mapping[name]] = x
#         return LossOutput(**output)


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

        "BCE_LOSS"
        # loss = torch.nn.BCELoss()(outputs, targets)

        "MSE_LOSS"
        # mse_loss = nn.MSELoss()(outputs, targets)
        # out_1 = torch.cat([outputs, outputs, outputs], axis=1)
        # loss_network = LossNetwork(vgg_model)
        # output = loss_network(out_1)
        # vgg_crit = output[0][1][2][3].mean()
        # loss = loss + vgg_crit

        # t2 = datetime.datetime.now()
        # print(t2 - t1)

        # vgg_crit = torch.mean(output)

        # loss = nn.MSELoss()(outputs, targets)  # convert into 3d
        # vgg = vgg16().cuda(device)
        # out_1 = torch.cat([outputs, outputs, outputs], axis=1)
        # vgg_crit = vgg(out_1)  ## vgg_crit.shape ==>>  torch.Size([8, 1000])
        # vgg_crit_A = torch.mean(
        #     vgg_crit
        # )  # vgg_crit = vgg_crit.view(1, 8000)  # vgg_crit = torch.flatten(vgg_crit)
        # # vgg_crit_sum = vgg_crit_f.sum()
        # loss = mse_loss + vgg_crit_A

        "SSIM_LOSS"
        # crit = SSIMLoss().cuda(device)
        # ssim = crit(outputs, targets)
        # loss = ssim

        # criterion = 1 - ms_ssim(outputs, targets)
        # #ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )
        # loss = criterion()
        # crit = SSIMLoss().cuda(device)
        # ssim = crit(outputs, targets)
        # loss = ssim

        "MAE_LOSS"
        # loss = torch.abs(targets - outputs).mean()
        # loss = torch.nn.L1Loss()(outputs, targets)
        # "Dice_loss"
        # intersection = (outputs * targets).sum()
        # smooth = 1
        # dice = (2.0 * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        # loss = 1 - dice

        "Dice_BC_Loss"
        # inputs = outputs.view(-1)
        # targets = targets.view(-1)
        # intersection = (outputs * targets).sum()
        # smooth = 1
        # dice_loss = 1 - (2.0 * intersection + smooth) / (
        #     outputs.sum() + targets.sum() + smooth
        # )
        # bce = torch.nn.BCELoss()(outputs, targets)
        # loss = bce + dice_loss

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

            "SSIM_LOSS"
            # crit = SSIMLoss().cuda(device)
            # ssim = crit(outputs, targets)
            # loss = ssim

            "MAE_loss"
            # loss = torch.abs(outputs - targets).mean()
            # loss = torch.nn.L1Loss()(outputs, targets)

           

            "BCE_LOSS"
            # loss = torch.nn.BCELoss()(outputs, targets)

            "MSE_LOSS"
            # loss = nn.MSELoss()(outputs, targets)  # convert into 3d
            # vgg = vgg16().cuda(device)
            # out_1 = torch.cat([outputs, outputs, outputs], axis=1)
            # vgg_crit = vgg(out_1)  ## vgg_crit.shape ==>>  torch.Size([8, 1000])
            # vgg_crit_A = torch.mean(
            #     vgg_crit
            # )  # vgg_crit = vgg_crit.view(1, 8000)  # vgg_crit = torch.flatten(vgg_crit)
            # # vgg_crit_sum = vgg_crit_f.sum()
            # loss = mse_loss + vgg_crit_A

            #         #print("batch"+str(idx) + " loss:" ,batch_mse)

            # mse_loss = nn.MSELoss()(outputs, targets)
            # psnr = 10 * torch.log10(1 / mse_loss)

            # out_1 = torch.cat([outputs, outputs, outputs], axis=1)
            # loss_network = LossNetwork(vgg_model)
            # output = loss_network(out_1)
            # vgg_crit = output[0][1][2][3].mean()
            # # vgg_crit = torch.mean(output)
            # loss = loss + vgg_crit
            # t2 = datetime.datetime.now()
            # print(t2 - t1)

            "Dice"

            # intersection = (outputs * targets).sum()
            # smooth = 1
            # dice = (2.0 * intersection + smooth) / (
            #     outputs.sum() + targets.sum() + smooth
            # )
            # loss = 1 - dice

            "Dice_Loss"
            # inputs = outputs.view(-1)
            # targets = targets.view(-1)
            # intersection = (outputs * targets).sum()
            # smooth = 1
            # dice_loss = 1 - (2.0 * intersection + smooth) / (
            #     outputs.sum() + targets.sum() + smooth
            # )
            # BCE = torch.nn.BCELoss()(outputs, targets)
            # loss = BCE + dice_loss

            "Huber_loss"
            loss = torch.nn.SmoothL1Loss()(outputs, targets)

            batch_MSEs.append(loss.item())

        #         # return final output and final targets
        batch_MSEs = np.array(batch_MSEs)
        epoch_loss = np.mean(batch_MSEs)
        print("epoch_loss", epoch_loss)
        # print("psnr_value", psnr)
        # print("ssim_value", ssim)

    return epoch_loss

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            targets = data[1]
            labels = data[2]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            output = model(inputs)
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            final_targets.extend(targets)
            final_outputs.extend(output)
    return final_outputs, final_targets
