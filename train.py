import os
import numpy as np
import torch
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
from Deep_Res_SE_Unet import Deep_Res_SE_Unet
from new_model import new_model


if __name__ == "__main__":

    project_path = "/home/thesis_2/"
    data_path = "/home/thesis_2/Emnist_dataset/"

    loss_curves = os.path.join(project_path, "loss_curves")
    model_opt_chp = os.path.join(project_path, "model_opt_chp")

    file_names = ["emnist_imgs.npy", "emnist_measures.npy", "emnist_labels.npy"]

    exp = "reduced_depth.pt"
    device = "cuda:6"
    epochs = 5
    is_model_trained = False
    ck_pt_path = "/home/thesis_2/model_opt_chp/reduced_depth.pt"

    if is_model_trained:
        checkpoint = torch.load(ck_pt_path)

    meas_images = np.load("/home/thesis_2/mnist_dataset/mnist_measures.npy")
    real_images = np.load("/home/thesis_2/mnist_dataset/mnist_imgs.npy")
    labels = np.load("/home/thesis_2/mnist_dataset/mnist_labels.npy")

    # mean, std = 0, 0.111234
    # noise = np.random.normal(mean, std, [128, 128])
    # meas_images = meas_images + noise

    "resizing to 0-255"
    # norm_image = cv2.normalize(meas_images, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # norm_image = norm_image.astype(np.uint8)

    # img_indices = np.arange(112800)
    img_indices = np.arange(60000)

    train_indices, test_indices, _, _ = train_test_split(
        img_indices, img_indices, test_size=0.20
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

    # model = Deep_Res_SE_Unet()
    model = new_model()

    if is_model_trained:
        model.load_state_dict(checkpoint["model"])

    model.to(device)

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
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    writer = SummaryWriter("tensorboard/" + exp + "/")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # lambda1 = lambda epochs: 0.65 ** epochs
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    if is_model_trained:
        # optimizer = optimizer.load.checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])

    if is_model_trained:
        start_epoch = checkpoint["epoch"]
        end_epoch = checkpoint["epoch"] + epochs
    else:
        start_epoch = 0
        end_epoch = epochs

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(start_epoch, end_epoch)):
        # scheduler.step()
        # print("Epoch:", epoch, "LR:", scheduler.get_lr())
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

        # if epoch % 25 == 0:
        #     print(epoch)
    writer.close()

    # torch.save(model.state_dict(), os.path.join(models_weights, exp + ".pt"))

    checkpoint = {
        "epoch": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # torch.save(checkpoint, os.path.join(model_opt_chp, exp + ".pt"))
    torch.save(checkpoint, os.path.join(model_opt_chp, "reduced_depth.pt"))
    # # checkpoint = torch.load('checkpoint.pth')

    plt.figure()
    plt.plot(list(range(1, epochs + 1)), train_losses, label="train")
    plt.plot(list(range(1, epochs + 1)), val_losses, label="val")
    plt.legend()

    # plt.savefig(os.path.join(loss_curves, exp + ".png"))
    plt.savefig(os.path.join(loss_curves, exp + ".png"))

    # print("")

