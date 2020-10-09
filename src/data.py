# coding=utf-8
# Created on 2020-05-21 09:35
# Copyright © 2020 Alan. All rights reserved.

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw

import os
import numpy as np
import glob
from PIL import Image, ImageFile
import keras
from src.utils import cprint
import csv


ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_eligible_img(img_path, imgsize, lower=2, wh_ratio=2.5):
    """
    Custimized filter for input img. Only the images that are satisfied with the content returns True
    :param img_path: Path of the image
    :return: True if the image satisfies the condition
    """
    img = Image.open(img_path)
    flag_min_size = img.size[1] >= imgsize/lower and img.size[0] >= imgsize/lower
    flag_wh_ratio = (wh_ratio >= img.size[1]/img.size[0] >= 1/wh_ratio)
    return flag_min_size and flag_wh_ratio


def create_dataset(path, model_path, imgsize, val_split):
    """
    (X_train, X_valid, y_train, y_valid) => "image_name_path" and its label as int id [0~classes-1]
    class_weights, as provided by sklearn
    :return: X_train, X_valid, y_train, y_valid, num_classes, class_weights
    """
    category = os.listdir(path)
    classes = []

    # count all the classes
    for cat in category:
        if os.path.isdir(os.path.join(path, cat)):
            classes.append(cat)
    num_classes = len(classes)

    # create X, y pairs and compute class weights
    X, y = [], []
    model_to_dict_path = os.path.join(model_path, "reflection.csv")
    with open(model_to_dict_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(["label", "encoded"])

        for index in range(len(classes)):
            sub_class = classes[index]
            writer.writerow([sub_class, index])
            img_files = glob.glob(os.path.join(os.path.join(path, sub_class), "*.jpg"))
            for img_path in img_files:
                if is_eligible_img(img_path=img_path, imgsize=imgsize):
                    X.append(img_path)
                    y.append(index)

    class_weights = cw.compute_class_weight("balanced", np.unique(y), y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_split)
    return X_train, X_test, y_train, y_test, num_classes, class_weights


def get_dataset(dataset_path, model_path, batch_size=32, imgsize=260, val_split=0.2, debug=False):
    """
    :param debug: debug mode returns the first100 images
    """
    X_train, X_valid, y_train, y_valid, num_classes, class_weights = create_dataset(
        dataset_path, model_path, imgsize, val_split)
    # Debug mode

    if debug:
        X_train, X_valid, y_train, y_valid = X_train[:100], X_valid[:100], y_train[:100], y_valid[:100]
    train_set = (X_train, y_train)
    valid_set = (X_valid, y_valid)

    cprint("[INFO]", "Data generator built! train data size {}, valid data size {}, classes {} \n".format(
        len(y_train), len(y_valid), num_classes))

    data_transform = transforms.Compose([
        # transforms.CenterCrop(imgsize),
        transforms.Resize((imgsize, imgsize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_params = {"n_classes": num_classes,
                      "class_weights": class_weights}

    gen_train = DataGenerator(train_set, data_transform, dataset_path, num_classes, batch_size)
    gen_valid = DataGenerator(valid_set, data_transform, dataset_path, num_classes, batch_size)
    return gen_train, gen_valid, dataset_params


class DataGenerator(keras.utils.Sequence):
    """
    Data generator builder for keras: fit_generator
    """
    def __init__(self, dataset, transform, root_path, n_classes, batch_size, noise_ratio=0):
        self.X, self.y = dataset
        self.transform = transform
        self.root_path = root_path
        self.n_classes = n_classes
        self.batch_size = int(batch_size)
        # TODO: add noise for training data
        self.ratio = noise_ratio
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = np.stack(X), np.stack(y)
        return np.moveaxis(X, 1, 3), keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation(self, indexes):
        X, y = [], []
        for idx in indexes:
            file_path = os.path.join(self.root_path, self.X[idx])
            X.append(self.transform(Image.open(file_path).convert("RGB")))
            # q = np.array(self.transform(Image.open(file_path).convert("RGB")))
            # print(q.max(), q.min(), q.mean())
            y.append(self.y[idx])
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        np.random.shuffle(self.indexes)
