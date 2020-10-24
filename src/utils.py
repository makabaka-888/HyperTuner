# coding=utf-8
# Created on 2020-07-09 10:22
# Copyright © 2020 Alan. All rights reserved.
from termcolor import colored
import datetime
import os


def cprint(level, msg):
    """
    colored visualization for important messages
    """
    text = colored(level, "green", attrs=["bold"])
    print("\n" + text + " " + str(msg))


def fetch_time():
    """
    simple time in string format
    """
    curr_time = str(datetime.datetime.now()).replace(" ", ":").split(".")[0]
    return curr_time


def hms_string(sec_elapsed):
    """
    timers for recording training time cost
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def get_weight_sum(model):
    cov_num, depth_num = 0, 0
    for mod_lay in model.layers:
        mod_lay.trainable = True
        if "depthwise" == mod_lay.name.split("_")[0]:
            depth_num = max(depth_num, int(mod_lay.name.split("_")[-1]))
        if "conv2d" == mod_lay.name.split("_")[0]:
            cov_num = max(cov_num, int(mod_lay.name.split("_")[-1]))
    return {"depthwise": depth_num, "conv2d": cov_num}


def make_file(output_path, backbone):
    """
    default model path: ./args.output_path/[args.backbone + current_time]
    default log path: ./args.output_path/[log + args.backbone + current_time]
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    curr_time = fetch_time()
    model_path = os.path.join(output_path, "_".join([backbone, curr_time]))
    log_path = os.path.join(output_path, "_".join([backbone, curr_time, "log"]))
    os.mkdir(model_path)
    os.mkdir(log_path)
    for mode_name in ["grid_ea", "bayes_converge", "bayes_global"]:
        os.mkdir(os.path.join(model_path, mode_name))
        os.mkdir(os.path.join(log_path, mode_name))
    cprint("[Model Path]", model_path)
    cprint("[Log Path]", log_path)
    return model_path, log_path
