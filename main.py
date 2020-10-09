# coding=utf-8
# Created on 2020-07-08 11:10
# Copyright © 2020 Alan. All rights reserved.
import warnings
from warnings import simplefilter
import logging
import argparse
import os

warnings.simplefilter("ignore")
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True

from src.search import model_search

ap = argparse.ArgumentParser()
# --------------------------- dataloading & outputs -----------------------------
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset (i.e., directory of images)",
                default="path/to/dataset"
                )
ap.add_argument("-op", "--output_path", required=False, help="path to save output model", default="./models")
ap.add_argument("-v", "--val_split", required=False, help="validation_split", type=float, default=0.15)
ap.add_argument("-im", "--imgsize", required=False, help="normalized image shape", type=int, default=260)
ap.add_argument("-b", "--batch_size", required=False, help="batch size", type=int, default=32)
ap.add_argument("-gpu", "--gpu", required=False, help="gpu assignment", type=str, default="0")
ap.add_argument("-db", "--debug", required=False, help="debug status", type=bool, default=False)

# --------------------------- model backbone -----------------------------
ap.add_argument("-bk", "--backbone", required=False, help="backbone for the model", default="EfficientNetB2")
args = vars(ap.parse_args())
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
gpu_cnt = sum(char.isdigit() for char in args["gpu"])

if __name__ == '__main__':
    model_search(dataset=args["dataset"],
                 backbone=args["backbone"],
                 val_split=args["val_split"],
                 imgsize=args["imgsize"],
                 batch_size=args["batch_size"],
                 output_path=args["output_path"],
                 gpu_cnt=gpu_cnt,
                 debug_mode=args["debug"])

