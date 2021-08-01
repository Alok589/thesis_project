import os
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import dataset
import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from PIL import Image
import cv2
import engine
from torch.optim.lr_scheduler import StepLR
from torch import nn

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.hub import tqdm
from skimage.filters import rank
import scipy.io as sio
import skimage.transform as skt
from PIL import Image
from PIL import ImageFile
from Deep_Res_SE_Unet import Deep_Res_SE_Unet
import tensorboard
import torch.nn as nn

if __name__ == "__main__":
    project_path = "/home/thesis_2/"
    data_path = "/home/thesis_2/mnist_dataset/"

    loss_curves = os.path.join(project_path, "loss_curves")
    model_opt_chp = os.path.join(project_path, "model_opt_chp")

    # file_names = ["Emnist_imgs.npy", "Emnist_measures.npy", "Emnist_labels.npy"]
    is_model_trained = False
    ck_pt_path = "/home/thesis_2/model_opt_chp/RSEUnet.pt"

    if is_model_trained:
        checkpoint = torch.load(ck_pt_path)

    exp = "tranfer_0"
    device = "cuda:6"
    epochs = 30
    model = Deep_Res_SE_Unet()
    model.to(device)
    model.load_state_dict(
        torch.load("/home/thesis_2/model_opt_chp/tranfer_0.pt")["model"]
    )

    for param in model.parameters():
        param.requires_grad = False
    # model.conv4.conv1.weight.requires_grad = True
    # model.conv4.conv1.bias.requires_grad = True
    # model.conv4.bn.weight.requires_grad = True
    # model.conv4.bn.bias.requires_grad = True
    # model.residual_4.conv1.weight.requires_grad = True
    # model.residual_4.conv1.bias.requires_grad = True
    # model.residual_4.conv2.weight.requires_grad = True
    # model.residual_4.conv2.bias.requires_grad = True
    # model.residual_4.bn.weight.requires_grad = True
    # model.residual_4.bn.bias.requires_grad = True
    # model.senet_4.fc[0].weight.requires_grad = True
    # model.senet_4.fc[2].weight.requires_grad = True
    # model.up_conv1.conv_trans1.weight.requires_grad = True
    # model.up_conv1.conv_trans1.bias.requires_grad = True
    # model.up_conv1.bn.weight.requires_grad = True
    # model.up_conv1.bn.bias.requires_grad = True
    # model.conv_bn1.conv1.weight.requires_grad = True
    # model.conv_bn1.conv1.bias.requires_grad = True
    # model.conv_bn1.bn.weight.requires_grad = True
    # model.conv_bn1.bn.bias.requires_grad = True
    # model.residual_5.conv1.weight.requires_grad = True
    # model.residual_5.conv1.bias.requires_grad = True
    # model.residual_5.conv2.weight.requires_grad = True
    # model.residual_5.conv2.bias.requires_grad = True
    # model.residual_5.bn.weight.requires_grad = True
    # model.residual_5.bn.bias.requires_grad = True
    # model.senet_5.fc[0].weight.requires_grad = True
    # model.senet_5.fc[2].weight.requires_grad = True
    # model.up_conv2.conv_trans1.weight.requires_grad = True
    # model.up_conv2.conv_trans1.bias.requires_grad = True
    # model.up_conv2.bn.weight.requires_grad = True
    # model.up_conv2.bn.bias.requires_grad = True
    # model.conv_bn2.conv1.weight.requires_grad = True
    # model.conv_bn2.conv1.bias.requires_grad = True
    # model.conv_bn2.bn.weight.requires_grad = True
    # model.conv_bn2.bn.bias.requires_grad = True
    # model.residual_6.conv1.weight.requires_grad = True
    # model.residual_6.conv1.bias.requires_grad = True
    # model.residual_6.conv2.weight.requires_grad = True
    # model.residual_6.conv2.bias.requires_grad = True
    # model.residual_6.bn.weight.requires_grad = True
    # model.residual_6.bn.bias.requires_grad = True
    # model.senet_6.fc[0].weight.requires_grad = True
    # model.senet_6.fc[2].weight.requires_grad = True
    # model.up_conv3.conv_trans1.weight.requires_grad = True
    # model.up_conv3.conv_trans1.bias.requires_grad = True
    # model.conv_bn3.conv1.weight.requires_grad = True
    # model.conv_bn3.conv1.bias.requires_grad = True
    # model.conv_bn3.bn.weight.requires_grad = True
    # model.conv_bn3.bn.bias.requires_grad = True
    # model.residual_7.conv1.weight.requires_grad = True
    # model.residual_7.conv1.bias.requires_grad = True
    # model.residual_7.conv2.weight.requires_grad = True
    # model.residual_7.conv2.bias.requires_grad = True
    # model.residual_7.bn.weight.requires_grad = True
    # model.residual_7.bn.bias.requires_grad = True
    # model.senet_7.fc[0].weight.requires_grad = True
    # model.senet_7.fc[2].weight.requires_grad = True
    # model.out1.conv1.weight.requires_grad = True
    # model.out1.conv1.bias.requires_grad = True
    # model.out1.bn.weight.requires_grad = True
    # model.out1.bn.bias.requires_grad = True
    # model.out2.conv1.weight.requires_grad = True
    # model.out2.conv1.bias.requires_grad = True
    # param = model.state_dict()
    # keys = param.keys()
    # keys = np.vstack(keys[54:150])
    model.conv3.conv1.weight.requires_grad = True
    model.conv3.conv1.bias.requires_grad = True
    model.conv3.bn.weight.requires_grad = True
    model.conv3.bn.bias.requires_grad = True
    model.residual_3.conv1.weight.requires_grad = True
    model.residual_3.conv1.bias.requires_grad = True
    model.residual_3.conv2.weight.requires_grad = True
    model.residual_3.conv2.bias.requires_grad = True
    model.residual_3.bn.weight.requires_grad = True
    model.residual_3.bn.bias.requires_grad = True
    model.senet_3.fc[0].weight.requires_grad = True
    model.senet_3.fc[2].weight.requires_grad = True
    model.conv4.conv1.weight.requires_grad = True
    model.conv4.conv1.bias.requires_grad = True
    model.conv4.bn.weight.requires_grad = True
    model.conv4.bn.bias.requires_grad = True
    model.residual_4.conv1.weight.requires_grad = True
    model.residual_4.conv1.bias.requires_grad = True
    model.residual_4.conv2.weight.requires_grad = True
    model.residual_4.conv2.bias.requires_grad = True
    model.residual_4.bn.weight.requires_grad = True
    model.residual_4.bn.bias.requires_grad = True
    model.senet_4.fc[0].weight.requires_grad = True
    model.senet_4.fc[2].weight.requires_grad = True
    model.up_conv1.conv_trans1.weight.requires_grad = True
    model.up_conv1.conv_trans1.bias.requires_grad = True
    model.up_conv1.bn.weight.requires_grad = True
    model.up_conv1.bn.bias.requires_grad = True
    model.conv_bn1.conv1.weight.requires_grad = True
    model.conv_bn1.conv1.bias.requires_grad = True
    model.conv_bn1.bn.weight.requires_grad = True
    model.conv_bn1.bn.bias.requires_grad = True
    model.residual_5.conv1.weight.requires_grad = True
    model.residual_5.conv1.bias.requires_grad = True
    model.residual_5.conv2.weight.requires_grad = True
    model.residual_5.conv2.bias.requires_grad = True
    model.residual_5.bn.weight.requires_grad = True
    model.residual_5.bn.bias.requires_grad = True
    model.up_conv2.conv_trans1.weight.requires_grad = True
    model.up_conv2.conv_trans1.bias.requires_grad = True
    model.up_conv2.bn.weight.requires_grad = True
    model.up_conv2.bn.bias.requires_grad = True
    model.conv_bn2.conv1.weight.requires_grad = True
    model.conv_bn2.conv1.bias.requires_grad = True
    model.conv_bn2.bn.weight.requires_grad = True
    model.conv_bn2.bn.bias.requires_grad = True
    model.residual_6.conv1.weight.requires_grad = True
    model.residual_6.conv1.bias.requires_grad = True
    model.residual_6.conv2.weight.requires_grad = True
    model.residual_6.conv2.bias.requires_grad = True
    model.residual_6.bn.weight.requires_grad = True
    model.residual_6.bn.bias.requires_grad = True
    model.up_conv3.conv_trans1.weight.requires_grad = True
    model.up_conv3.conv_trans1.bias.requires_grad = True
    model.conv_bn3.conv1.weight.requires_grad = True
    model.conv_bn3.conv1.bias.requires_grad = True
    model.conv_bn3.bn.weight.requires_grad = True
    model.conv_bn3.bn.bias.requires_grad = True
    model.residual_7.conv1.weight.requires_grad = True
    model.residual_7.conv1.bias.requires_grad = True
    model.residual_7.conv2.weight.requires_grad = True
    model.residual_7.conv2.bias.requires_grad = True
    model.residual_7.bn.weight.requires_grad = True
    model.residual_7.bn.bias.requires_grad = True
    model.out1.conv1.weight.requires_grad = True
    model.out1.conv1.bias.requires_grad = True
    model.out1.bn.weight.requires_grad = True
    model.out1.bn.bias.requires_grad = True
    model.out2.conv1.weight.requires_grad = True
    model.out2.conv1.bias.requires_grad = True

    if is_model_trained:
        model.load_state_dict(checkpoint["model"])

    meas_images = np.load("/home/thesis_2/mnist_dataset/mnist_measures.npy")
    real_images = np.load("/home/thesis_2/mnist_dataset/mnist_imgs.npy")
    labels = np.load("/home/thesis_2/mnist_dataset/mnist_labels.npy")

    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, [128, 128])
    meas_images = meas_images + s

    # img_indices = np.arange(112800)
    img_indices = np.arange(60000)

    train_indices, test_indices, _, _ = train_test_split(
        img_indices, img_indices, test_size=0.10
    )

    train_indices, val_indices, _, _ = train_test_split(
        train_indices, train_indices, test_size=0.20
    )

    train_X = meas_images[train_indices]  # [:70000]
    val_X = meas_images[val_indices]  # [:70000]
    test_X = meas_images[test_indices]  # [:70000]

    train_Y = real_images[train_indices]  # [:70000]
    val_Y = real_images[val_indices]  # [:70000]
    test_Y = real_images[test_indices]  # [:70000]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    val_labels = labels[val_indices]

    train_dataset = dataset.ClassificationDataset(
        meas_images=train_X, real_images=train_Y, labels=train_labels
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, shuffle=True, num_workers=2
    )

    valid_dataset = dataset.ClassificationDataset(
        meas_images=val_X, real_images=val_Y, labels=val_labels
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=12, shuffle=False, num_workers=2
    )

    test_dataset = dataset.ClassificationDataset(
        meas_images=test_X, real_images=test_Y, labels=test_labels
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=2
    )

    writer = SummaryWriter("tensorboard/" + exp + "/")
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, amsgrad=True
    )

    if is_model_trained:
        # optimizer = optimizer.load.checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint["optimizer"])

    # for (name, module) in model.named_children():
    #     if name == ("out2"):
    #         for layer in module.children():
    #             for param in layer.parameters():
    #                 param.requires_grad = True
    # for (name, module) in model.named_children():
    #     if name == (
    #         "conv1",
    #         "conv2",
    #         "conv3",
    #         "conv4",
    #         "residual_1",
    #         "residual_2",
    #         "residual_3",
    #         "residual_4",
    #         "residual_5",
    #         "residual_6",
    #         "residual_7",
    #         "senet_1",
    #         "senet_2",
    #         "senet_3",
    #         "senet_4",
    #         "up_conv1",
    #         "up_conv2",
    #         "up_conv3",
    #         "conv_bn1",
    #         "conv_bn2",
    #         "conv_bn3",
    #     ):
    #         for layer in module.children():
    #             for param in layer.parameters():
    #                 param.requires_grad = False

    # for (name, module) in model.named_children():
    #     if name == ("out1", "out2"):
    #         for layer in module.children():
    #             for param in layer.parameters():
    #                 param.requires_grad = True

    # for param in model.parameters():
    #     param.requires_grad = False

    # model.out1 = nn.Sequential(
    #     (
    #         nn.Conv2d(32, 64, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout2d(p=0.5),
    #         nn.BatchNorm2d(64, momentum=0.75),
    #         nn.Conv2d(64, 1, kernel_size=1),
    #     )
    # )

    # model.out1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace = True), nn.Dropout2d(p=0.5), nn.BatchNorm2d(64, momentum=0.75))
    # model.out2 = nn.Conv2d(64, 1, kernel_size=1)

    # model.up_conv3 = nn.Sequential(
    #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(32, momentum=0.75),
    #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(32, momentum=0.75),
    #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(64, momentum=0.75),
    #     nn.Conv2d(64, 1, kernel_size=1),
    # )

    # class identity(nn.Module):
    #     def __init__(self):
    #         super(identity, self).__init__()

    #     def forward(self, x):
    #         return x

    # model.conv_bn3 = identity()
    # model.residual_7 = identity()
    # model.out1 = identity()
    # model.out2 = identity()

    # model = model.to(device)
    # if is_model_trained:
    #     start_epoch = checkpoint["epoch"]
    #     end_epoch = checkpoint["epoch"] + epochs
    # else:
    #     start_epoch = 0
    #     end_epoch = epochs

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        print("epoch " + str(epoch))
        train_loss = engine.train(train_loader, model, optimizer, device=device)
        val_loss = engine.evaluate(valid_loader, model, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("train", train_loss, epoch)
        writer.add_scalar("val", val_loss, epoch)
        writer.add_scalars(
            "train and val losses", {"train": train_loss, "val": val_loss}, epoch
        )

    writer.close()

    checkpoint = {
        "epoch": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(model_opt_chp, "tranfer_0.pt"))

    plt.figure()
    plt.plot(list(range(1, epochs + 1)), train_losses, label="train")
    plt.plot(list(range(1, epochs + 1)), val_losses, label="val")
    plt.legend()
    plt.savefig(os.path.join(loss_curves, exp + ".png"))
