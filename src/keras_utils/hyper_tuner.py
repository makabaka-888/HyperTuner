# coding=utf-8
# Created on 2020-09-12 17:21
# Copyright © 2020 Alan. All rights reserved.

from config import *
from src.model_queue import ModelQueue
from src.keras_utils.grid_ea import GridEA
from src.keras_utils.bayes import BayesOpt


class HyperTuner:
    def __init__(self, data_params, imgsize, backbone, gen_train, gen_valid, model_path, log_path, gpu_cnt):

        self.model_queue = ModelQueue(queue_size=ModelQueueConfig["queue_size"], log_lr_grid=ModelQueueConfig["log_lr_grid"])

        self.grid_ea = GridEA(self.model_queue, data_params, backbone, model_path, log_path,
                              imgsize, gpu_cnt, gen_train, gen_valid)

        self.bayes_opt = BayesOpt(self.model_queue, data_params, backbone, model_path, log_path,
                                  imgsize, gpu_cnt, gen_train, gen_valid, BayesConfig)

    def optimize(self):
        self.grid_ea.optimize()
        # work on refining the config.neuronShrink
        self.bayes_opt.optimize()
