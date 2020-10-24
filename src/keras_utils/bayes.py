# coding=utf-8
# Created on 2020-09-12 17:19
# Copyright © 2020 Alan. All rights reserved.

from src.utils import cprint, fetch_time
from src.keras_utils.customized_callbacks import LRTensorBoard
from src.load_base_model import get_base_model
from config import BayesConfig

import os
import efficientnet
import keras
from bayes_opt import BayesianOptimization
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping


TMP_WEIGHT = "tmp_parallel_weights.h5"


class BayesOpt:
    def __init__(self, model_queue, params, backbone, model_path, log_path, imgsize, gpu_cnt, gen_train, gen_valid,
                 config):
        self.model_queue = model_queue
        self.params = params
        self.backbone = backbone
        self.model_path = model_path
        self.log_path = log_path
        self.imgsize = imgsize
        self.gpu_cnt = gpu_cnt
        self.gen_train, self.gen_valid = gen_train, gen_valid
        self.config = config
        self.pooling_size = self.get_pooling_size()

    def optimize(self):
        """
        Bayesian Optimization function
        """
        if self.pooling_size <= 2 * self.params["n_classes"]:
            return

        optimizer = BayesianOptimization(f=self.eval,
                                         pbounds=BayesConfig["pbounds"],
                                         verbose=2,
                                         random_state=42)

        optimizer.maximize(init_points=BayesConfig["bayes_init_points"], n_iter=BayesConfig["bayes_num_iter"])

    def get_pooling_size(self):
        model = get_base_model(backbone=self.backbone, imgshape=self.imgsize)
        last_layer_shape = model.layers[-1].input_shape[-1]
        return last_layer_shape

    def eval(self, global_log_lr, neuron_shrink):
        model_path, em_log_lr = self.model_queue.k_crossover(1)
        model = self.generate_model(base_name=model_path,
                                    neuron_shrink=neuron_shrink,
                                    num_classes=self.params["n_classes"])

        # train on dense layers
        score_converge = self.fit(model=model, log_lr=em_log_lr, layers="top")
        # train on all layers
        score_global = self.fit(model=model, log_lr=global_log_lr, layers="all")
        return max(score_converge, score_global)

    def generate_model(self, base_name, neuron_shrink=0.01, num_classes=30):
        """
        dense(1)_dim = global_pooling * neuron_shrink
        :param neuron_shrink: shrink factor for neuron number: dense(k+1)_dim = dense(k) * neuronShrink
        :param num_classes: num of classes for final output
        cov_layers -> global_avg_pooling -> dense1 -> dense2 -> ... -> denseN with fixed neuronShrink size
        """
        # load existing model
        if "EfficientNet" in base_name:
            model = efficientnet.load_model(base_name)
        else:
            model = keras.models.load_model(base_name)

        while "dense" in model.layers[-1].name:
            model.layers.pop()

        neuron_count = int(model.layers[-1].input_shape[-1] * neuron_shrink)
        neuron_num = []

        while neuron_count >= 2 * num_classes:
            neuron_count = int(neuron_count * neuron_shrink)
            neuron_num.append(neuron_count)

        x = model.layers[-1].output
        for num in neuron_num:
            x = Dense(num, activation="relu")(x)
            x = Dropout(0.5)(x)
        pred = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = Model(input=model.input, output=pred)
        return model

    def fit(self, model, log_lr, layers):
        """
        fit on either single gpu method or multiple gpu method
        :param layers: "all": train on all layers, "top": train on dense layers
        """

        # turn log learning rate into learning rate
        lr = 10 ** log_lr

        # set dense layers as trainable
        if layers == "top":
            mode = "bayes_converge"
            optimizer = Adam(lr)
            for layer in model.layers:
                if "dense" in layer.name:
                    layer.trainable = True
                else:
                    layer.trainable = False

        # set all layers as trainable
        else:
            mode = "bayes_global"
            optimizer = SGD(lr)
            for layer in model.layers:
                layer.trainable = True

        curr_time = fetch_time()

        model_path = os.path.join(self.model_path, mode, curr_time)
        log_path = os.path.join(self.log_path, mode, curr_time)

        early_stopping = EarlyStopping(monitor="val_acc", patience=BayesConfig["patience"], restore_best_weights=True)
        tensor_board = LRTensorBoard(log_dir=log_path)

        # multi gpu acceleration
        if self.gpu_cnt > 1:
            print("converting model to parallel")
            parallel_model = keras.utils.multi_gpu_model(model, gpus=self.gpu_cnt)
            parallel_model.compile(optimizer=optimizer,
                                   loss="categorical_crossentropy",
                                   metrics=[keras.losses.categorical_crossentropy, "acc"])

            history = parallel_model.fit_generator(generator=self.gen_train,
                                                   epochs=BayesConfig["max_epoch"],
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
                                          epochs=BayesConfig["max_epoch"],
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
        # add to model queue for faster converge
        self.model_queue.append_info(model_path=model_path, log_lr=log_lr, train_params=(score, num_epcoch))
        return score
