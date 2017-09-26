import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import os


def resize_img(img):
    return cv2.resize(img, (224, 224))


def get_preprocessed_img(dataset_path):

    # put the paths of images to the list
    smoke_img_list = [glob(os.path.join(dataset_path, 'smoke/*.jpg')) for path in os.listdir(dataset_path)][0]
    nonsmoke_img_list = [glob(os.path.join(dataset_path, 'non/*.jpg')) for path in os.listdir(dataset_path)][0]

    # load the image and resize to 224x224 for the vgg input
    smoke_img = np.array([resize_img(plt.imread(img_path)) for img_path in smoke_img_list])
    nonsmoke_img = np.array([resize_img(plt.imread(img_path)) for img_path in nonsmoke_img_list])

    return smoke_img, nonsmoke_img
