# coding=utf-8
# Created  on 2020-07-08 09:54
# Copyright © 2020 Alan. All rights reserved.
# for evaluating the single image

from torchvision import transforms
import efficientnet
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json

import os
MODEL_NAME = "bag_color_86_automl_1.h5"
transform = transforms.Compose([
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
model = efficientnet.load_model(MODEL_NAME)
img_list = os.listdir('./test_image')
print(img_list)
dic = {}
for img_name in img_list:
    img = transform(Image.open(os.path.join('./test_image', img_name)).convert("RGB"))
    img = np.array(img)
    img = img[np.newaxis, :, :, :]
    img = np.moveaxis(img, 1, 3)
    pre = model.predict(img)
    dic[img_name] = pre.tolist()

with open(MODEL_NAME + '.json', 'w') as f:
    json.dump(dic, f)
