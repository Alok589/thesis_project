import torch
import numpy as np
import torchvision
from PIL import Image
from PIL import ImageFile
import skimage
import torchvision.transforms as transforms
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, meas_images, real_images, labels):
        self.meas_images = meas_images
        self.real_images = real_images
        self.labels = labels

    def __len__(self):
        return len(self.meas_images)

    def __getitem__(self, item):

        meas = self.meas_images[item]
        gt_image = self.real_images[item]
        gt_image = cv2.resize(gt_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)


        # img2 = np.zeros((3, 128, 128))
        # img2[0, :, :] = meas
        # img2[1, :, :] = meas
        # img2[2, :, :] = meas


        # gt_img2 = np.zeros((3,128, 128))
        # gt_img2[0, :, :] = gt_image
        # gt_img2[1, :, :] = gt_image
        # gt_img2[2, :, :] = gt_image


        labels = self.labels[item]

        meas = np.expand_dims(meas, 0)
        gt_image = np.expand_dims(gt_image, 0)

        # return img2, gt_img2, labels
        return meas, gt_image, labels
