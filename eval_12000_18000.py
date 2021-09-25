import matplotlib.pyplot as plt
from cv2 import exp, transform
import numpy as np
import os
import cv2
from pytorch_msssim import ssim, ms_ssim
from Deep_Res_SE_Unet import Deep_Res_SE_Unet
import pandas as pd
import torch
from Res_Unet import Res_Unet

model = Deep_Res_SE_Unet()
# model = Res_Unet()
model.load_state_dict(
    torch.load("/home/thesis_2/model_opt_chp/exp_Res_Unet.pt")["model"]
)
model.eval()
device = "cuda:5"
# test_set_size = 22560
# test_set_size = np.arange(12000, 18000)


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


def evaluate_3_image(idx, savePlot):

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


for idx in range(12000, 18000):
    ssim_val, psnr_val = evaluate_3_image(idx, savePlot=False)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    print(f"image:{idx}, ssim:{ssim_val}, psnr:{psnr_val}")


psnr_list = np.array(psnr_list)
ssim_list = np.array(ssim_list)

# indices = np.arange(test_set_size)
result_array1 = np.hstack([psnr_list])
result_array2 = np.hstack([ssim_list])
pd.DataFrame({"psnr": result_array1, "ssim": result_array2}).to_csv(
    "/home/thesis_2/Res_Unet_csvfiles/result_03.csv"
)
