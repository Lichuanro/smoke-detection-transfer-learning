from preprocess_img import get_preprocessed_img
from train import train
from evaluation import get_evaluation
from keras.models import load_model
import os
import numpy as np

# set the image dataset path
dataset_path = '/input/1'

# get the preprocessed image for training
smoke_img, nonsmoke_img = get_preprocessed_img(dataset_path)

# create the label
smoke_labels = np.ones(smoke_img.shape[0], dtype=np.int8)
nonsmoke_labels = np.zeros(nonsmoke_img.shape[0], dtype=np.int8)

# concatenate both the smoke and non-smoke image and label for training

smoke_pos = int(smoke_img.shape[0] * 0.8)
nonsmoke_pos = int(nonsmoke_img.shape[0] * 0.8)

X_train = np.concatenate((smoke_img[:smoke_pos], nonsmoke_img[:nonsmoke_pos]))
y_train = np.concatenate((smoke_labels[:smoke_pos], nonsmoke_labels[:nonsmoke_pos]))

X_validation = np.concatenate((smoke_img[smoke_pos:], nonsmoke_img[nonsmoke_pos:]))
y_validation = np.concatenate((smoke_labels[smoke_pos:], nonsmoke_labels[nonsmoke_pos:]))

# train the model if not exist
if not os.path.exists('./smoke_detection.h5'):
    train(X_train, y_train, X_validation, y_validation)

# load the model
model = load_model('smoke_detection.h5')

# get evaluation
test_dataset_path = '/input/2'
test_smoke_img, test_nonsmoke_img = get_preprocessed_img(test_dataset_path)
dr, far, acr = get_evaluation(test_smoke_img, test_nonsmoke_img, model)