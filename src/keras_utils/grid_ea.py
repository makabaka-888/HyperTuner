# coding=utf-8
# Created on 2020-09-12 17:18
# Copyright © 2020 Alan. All rights reserved.

from config import GridConfig
from src.utils import cprint
from src.load_base_model import get_base_model
from src.keras_utils.customized_callbacks import LRTensorBoard
from src.utils import fetch_time

from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model

import os

TMP_WEIGHT = "tmp_parallel_weights.h5"


class GridEA:
    def __init__(self, model_queue, params, backbone, model_path, log_path, imgsize, gpu_cnt, gen_train, gen_valid):
        self.model_queue = model_queue
        self.params = params
        self.backbone = backbone
        self.model_path = model_path
        self.log_path = log_path
        self.imgsize = imgsize
        self.gpu_cnt = gpu_cnt
        self.gen_train, self.gen_valid = gen_train, gen_valid

    def optimize(self):
        """
        gird: (l1_regularizer, l2_regularizer, log_lr)
        """
        # enumerate initial grids
        for log_lr in self.model_queue.log_lr_grid:
            model = self.generate_model(num_classes=self.params["n_classes"], backbone=self.backbone)

            cprint("[Grid Search]", "Training at log_lr: " + str(log_lr))

            # model = add_regularization(model=model, conv_reg=1e-6, dense_reg=1e-5)
            self.fit(model=model, log_lr=log_lr)

    def generate_model(self, num_classes, backbone):
        # load pre-trained model
        model = get_base_model(backbone=backbone, imgshape=self.imgsize)
        x = model.layers[-1].output
        pred = Dense(num_classes, activation="softmax")(x)
        return Model(input=model.input, output=pred)

    def fit(self, model, log_lr):
        """
        fit model on either single gpu method or multiple gpu method
        :param log_lr:
        """
        for layer in model.layers:
            layer.trainable = True

        # turn log learning rate into learning rate
        lr = 10 ** log_lr
        mode = "grid_ea"
        curr_time = fetch_time()

        model_path = os.path.join(self.model_path, mode, curr_time)
        log_path = os.path.join(self.log_path, mode, curr_time)

        early_stopping = EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True)
        tensor_board = LRTensorBoard(log_dir=log_path)

        optimizer = Adam(lr)

        # multi gpu
        if self.gpu_cnt > 1:
            print("converting model to parallel...")
            parallel_model = multi_gpu_model(model, gpus=self.gpu_cnt)
            parallel_model.compile(optimizer=optimizer,
                                   loss="categorical_crossentropy",
                                   metrics=["categorical_crossentropy", "acc"])

            history = parallel_model.fit_generator(generator=self.gen_train,
                                                   epochs=GridConfig["max_epoch"],
                                                   verbose=1,
                                                   validation_data=self.gen_valid,
                                                   callbacks=[early_stopping, tensor_board],
                                                   class_weight=self.params["class_weights"])

            parallel_model.layers[-2].save_weights(TMP_WEIGHT)
            model.load_weights(TMP_WEIGHT)

        # single gpu
        else:
            model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["categorical_crossentropy", "acc"])

            history = model.fit_generator(generator=self.gen_train,
                                          epochs=GridConfig["max_epoch"],
                                          verbose=1,
                                          validation_data=self.gen_valid,
                                          callbacks=[early_stopping, tensor_board],
                                          class_weight=self.params["class_weights"])

        # score at percentage scale, e.g. 88.43
        score = max(history.history["val_acc"]) * 100

        num_epcoch = len(history.history["val_acc"])

        # save model
        model_path = model_path + str(round(score, 2)) + ".h5"
        model.save(model_path)

        # add to model queue
        self.model_queue.append_info(model_path=model_path, log_lr=log_lr, train_params=(score, num_epcoch))
