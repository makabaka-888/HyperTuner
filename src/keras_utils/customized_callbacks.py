# coding=utf-8
# Created on 2020-09-10 22:44
# Copyright © 2020 Alan. All rights reserved.


import numpy as np
from keras.callbacks import Callback, TensorBoard
from keras.optimizers import adam
import keras.backend as K


class SGDRScheduler(Callback):
    """
    SGD learning rate scheduler with periodic restarts.
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    """
    def __init__(self, min_lr, max_lr, lr_decay=0.9, cycle_length=2, mult_factor=1.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.next_restart = cycle_length
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def on_train_begin(self, logs={}):
        """
        Initialize the learning rate to the minimum value at the start of training
        """
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_epoch_end(self, epoch, logs={}):
        """
        Check for end of current cycle, apply restarts when necessary
        """
        self.max_lr *= self.lr_decay
        self.min_lr *= self.lr_decay
        lr = self.min_lr + (self.max_lr - self.min_lr) * (self.next_restart - epoch) / self.cycle_length
        K.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if epoch + 1 == self.next_restart:
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length


class LRTensorBoard(TensorBoard):
    """
    self.model.optimizer.lr: customized tensorboard for revealing learning rate
    default path = "./logs"
    """
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def adam_lr_getter(self):
        decay = self.model.optimizer.decay
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        """
        if isinstance(self.model.optimizer, adam):
            logs.update({"lr": self.adam_lr_getter()})
        else:
            logs.update({"lr": K.eval(self.model.optimizer.lr)})
        """
        logs.update({"lr": K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
