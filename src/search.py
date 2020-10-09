# coding=utf-8
# Created  on 2020-07-08 09:39
# Copyright © 2020 Alan. All rights reserved.

from src.keras_utils.hyper_tuner import HyperTuner
from src.utils import *
from src.data import get_dataset

TMP_WEIGHT = "tmp_parallel_weights.h5"


def model_search(dataset, backbone, val_split, imgsize, batch_size, output_path, gpu_cnt, debug_mode=False):
    """
    Function for model search
    :param dataset: dataset path
    :param backbone: one of the 'Load_Base_Model' file, please refer to ./src/load_base_model.py file
    :param output_path: model .h5 file output
    :param gpu_cnt: the gpu number to use
    """
    model_path, log_path = make_file(output_path, backbone)

    gen_train, gen_valid, params = get_dataset(
        dataset_path=dataset,
        model_path=model_path,
        batch_size=batch_size,
        imgsize=imgsize,
        val_split=val_split,
        debug=debug_mode)

    ht = HyperTuner(
        data_params=params,
        imgsize=imgsize,
        backbone=backbone,
        gen_train=gen_train,
        gen_valid=gen_valid,
        model_path=model_path,
        log_path=log_path,
        gpu_cnt=gpu_cnt)

    # HyperTuning Optimization
    ht.optimize()
