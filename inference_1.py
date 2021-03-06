from skimage import transform as tf
import torchvision
from skimage import transform
from Dense_Unet import Dense_Unet
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from cv2 import exp, transform
from numpy.lib.npyio import save
import dataset
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import numpy as np
import os
import skimage.io
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage import io

# from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer

# from Deep_Res_Unet import Deep_Res_Unet
from engine import evaluate
import cv2
from pytorch_msssim import ssim, ms_ssim
import pytorch_ssim
from piqa import ssim
from skimage.metrics import structural_similarity as ssim

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
from Dense_Unet import Dense_Unet
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
from Deep_Res_SE_Unet import Deep_Res_SE_Unet

# from test_3 import test_loader
import pandas as pd

model = Deep_Res_SE_Unet()
model.load_state_dict(torch.load("/home/thesis_2/model_opt_chp/exp_5m.pt")["model"])
model.eval()
device = "cuda:5"
# test_set_size = 22560
test_set_size = 5


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


def evaluate_1_image(idx, savePlot):

    image_idx = idx  # 12564  # 2126  # 5020  # 50565  # 1086#9022 #99 #1039 #25 #5020
    x = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]
    y = np.load("/home/thesis_2/Emnist_dataset/emnist_imgs.npy")[image_idx]
    y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    # y = y.to(device, dtype=torch.tensor)
    z = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]

    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    data = torch.tensor(x)

    op = model(data)

    pred = op.to(device)

    crit = SSIMLoss().cuda(device)
    y = np.expand_dims(y, 0)
    y = np.expand_dims(y, 0)
    y = torch.tensor(y).to(torch.float32).to(device)

    mse_loss = torch.mean(torch.abs(y - pred) ** 2)
    psnr = 10 * torch.log10(1 / mse_loss)
    ssim = crit(y, pred)

    if savePlot:
        fig = plt.figure()
        plt.suptitle(f"ssim: {ssim.item()}, psnr:{psnr.item()}")

        plt.subplot(1, 4, 1)
        plt.imshow(z)
        plt.title("measurements")

        plt.subplot(1, 4, 2)
        pred1 = pred[0][0].cpu().detach().numpy()
        plt.imshow(pred1)
        plt.title("pred")

        y1 = y[0][0].cpu().detach().numpy()
        plt.subplot(1, 4, 3)
        plt.imshow(y1)
        plt.title("real_image")

        plt.subplot(1, 4, 4)
        plt.imshow(np.abs(pred1 - y1))
        plt.title("Pred - real_image")

        plot_name = f"{idx}.png"
        plt.savefig(os.path.join("/home/thesis_2/new_inf", plot_name))

    return ssim.item(), psnr.item()


psnr_list = []
ssim_list = []


for idx in range(test_set_size):
    ssim_val, psnr_val = evaluate_1_image(idx, savePlot=True)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    print(f"image:{idx}, ssim:{ssim_val}, psnr:{psnr_val}")


psnr_list = np.array(psnr_list)
ssim_list = np.array(ssim_list)
# rmse_list = np.array(rmse_list)

indices = np.arange(test_set_size)
# result_array = np.hstack([indices, psnr_list, ssim_list])
result_array1 = np.hstack([psnr_list])
result_array2 = np.hstack([ssim_list])
# result_array3 = np.hstack([rmse_list])

# pd.DataFrame(result_array).to_csv("/home/thesis_2/result_03.csv")
pd.DataFrame({"psnr": result_array1, "ssim": result_array2}).to_csv(
    "/home/thesis_2/csv_files/result_03.csv"
)

# df = pd.read_csv("/home/thesis_2/result_03.csv")
# mean1 = df["psnr"].mean()
# mean2 = df["ssim"].mean()
