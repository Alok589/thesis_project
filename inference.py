from skimage import transform as tf
import torchvision
from skimage import transform

# from Dense_Unet import Dense_Unet
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
from Deep_Res_SE_Unet import Deep_Res_SE_Unet

##################
from skimage import transform as tf
import torchvision
from skimage import transform
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
from Res_Unet import Res_Unet
from new_model import new_model


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


# model = Deep_Res_SE_Unet()
model = new_model()
model.load_state_dict(
    torch.load("/home/thesis_2/model_opt_chp/transfer_reduced_depth.pt")["model"]
)
model.eval()
device = "cuda:5"

# X = skimage.io.imread(
#     "/home/thesis_bk/dataset/measurements/n01440764/n01440764_457..png"
# )

image_idx = (
    1039  # 52156  # 12564  # 2126  # 5020  # 50565  # 1086#9022 #99 #1039 #25 #5020
)
x = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]

mu, sigma = 0, 0.02  # mean and standard deviation
s = np.random.normal(mu, sigma, [128, 128])
x = x + s
x = torch.tensor(x).to(torch.float32)

y = np.load("/home/thesis_2/Emnist_dataset/emnist_imgs.npy")[image_idx]
y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
z = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]


x = np.expand_dims(x, 0)
x = np.expand_dims(x, 0)
data = torch.tensor(x)
op = model(data)

pred = op.to(device)

y = np.expand_dims(y, 0)
y = np.expand_dims(y, 0)
y = torch.tensor(y).to(torch.float32).to(device)

mse_loss = torch.mean(torch.abs(y - pred) ** 2)
psnr = 10 * torch.log10(1 / mse_loss)

crit = SSIMLoss().cuda(device)
ssim = crit(y, pred)

fig = plt.figure()
plt.suptitle(f"ssim: {ssim.item()}, psnr:{psnr.item()}")

plt.subplot(1, 4, 1)
plt.imshow(z)
plt.axis("off")

# plt.title("measurements")

plt.subplot(1, 4, 2)
pred1 = pred[0][0].cpu().detach().numpy()
plt.imshow(pred1)
plt.axis("off")

# plt.title("pred")

y1 = y[0][0].cpu().detach().numpy()
plt.subplot(1, 4, 3)
plt.imshow(y1)
plt.axis("off")

# plt.title("real_image")

plt.subplot(1, 4, 4)
plt.imshow(np.abs(pred1 - y1))
plt.axis("off")
# plt.title("Pred - real_image")

# plot_name = f"{idx}.png"

plot_name = "transfer_reduced_depth" + ".png"
plt.savefig(os.path.join("/home/thesis_2/inference_plots", plot_name))

# print("recon", recn)
# print(recn.shape)
# recn = np.swapaxes(recn, 0, 1)

# plt.figure()
# skimage.io.imshow(op)
# io.show()
# plt.savefig(os.path.join("/home/thesis_2/inference_plots", plot_name))

# fig = plt.figure()


# # mse_loss = nn.MSELoss()(outputs, targets)
# # psnr = 10 * torch.log10(1 / mse_loss)

# mse_loss = np.mean(np.abs(y - pred) ** 2)
# psnr = 10 * np.log10(1 / mse_loss)

# # crit = SSIMLoss().cuda(device)
# # ssim = crit(pred, y)


# plt.suptitle(f"psnr: {psnr}")
# plt.suptitle(f"ssim: {ssim}")


# plt.subplot(1, 4, 1)
# plt.imshow(z)
# plt.title("measurements")

# # ssim = ssim(y, pred)
# # plt.subplot(f"ssim: {ssim}")
# ####################################################
# plt.subplot(1, 4, 2)
# plt.imshow(pred)
# plt.title("pred")

# # plt.subplot(1, 4, 2)
# # plt.hist(pred, bins=10)

# plt.subplot(1, 4, 3)
# plt.imshow(y)
# plt.title("real_image")

# plt.subplot(1, 4, 4)
# plt.imshow(np.abs(pred - y))
# plt.title("Pred - real_image")


# # plt.subplot(1, 4, 4)
# # plt.hist(y, bins=10)

# # plot_name = "comparision" + ".png"
# # plt.savefig(os.path.join("/home/thesis_2/histogram_plots", plot_name))


# # plt.subplot(1, 3, 3)
# # plt.imshow(y)
# # plt.title("Real_img")

# # plt.subplot(1, 3, 3)
# # plt.bar(pred, height=(5, 5))
# # plt.title("hist")


# # plt.subplot(1, 4, 4)
# # plt.imshow(np.abs(pred - y))
# # plt.title("|pred - Real_img|")
# plot_name = "1" + ".png"
# # plt.savefig("infe_3.png")
# plt.savefig(os.path.join("/home/thesis_2/inference_plots", plot_name))
# plt.figure()
# plt.imshow(pred)
# plt.savefig("inference_1.png")


# directory = "models_weights"
# parent_dir = "/thesis/"
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)


# torch.save(model.state_dict(), path)
# model.load_state_dict(torch.load(path))
# model.eval(path="entire_model.pt")

# PATH = "entire_model.pt"

# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval(PATH="entire_model.pt")

# plt.subplot(1, 4, 1)
# img_fre1 = np.fft.fft2(z)
# plt.imshow(np.log(1 + np.abs(img_fre1)), "gray")
# plt.title("spectrum")


# plt.subplot(1, 4, 2)
# img_shift = np.fft.fftshift(img_fre1)
# plt.imshow(np.log(1 + np.abs(img_shift)), "gray")
# plt.title("shifted")

# plt.subplot(1, 4, 3)
# img_deshift = np.fft.ifftshift(img_shift)
# plt.imshow(np.log(1 + np.abs(img_deshift)), "gray")
# plt.title("inverse")

# plt.subplot(1, 4, 4)
# img_reversed = np.fft.irfft(img_deshift)
# plt.imshow(np.log(1 + np.abs(img_reversed)), "gray")
# plt.title("inverse")

# plot_name = "spectrum" + ".png"
# plt.savefig(os.path.join("/home/thesis_2/histogram_plots", plot_name))
