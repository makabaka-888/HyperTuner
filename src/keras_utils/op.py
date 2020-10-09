# coding=utf-8
# Created on 2020-05-21 09:38
# Copyright © 2020 Alan. All rights reserved.

import tempfile
import os
import keras


def add_regularization(model, conv_reg=0, dense_reg=0):
    """
    if not isinstance(regularizer, keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of keras.regularizers.Regularizer")
        return model
    """
    for layer in model.layers:
        for attr in ["kernel_regularizer"]:
            if hasattr(layer, attr):

                # set conv l2 regularization
                if "conv" in layer.name:
                    if conv_reg == 0:
                        setattr(layer, attr, None)
                    else:
                        setattr(layer, attr, keras.regularizers.l2(conv_reg))

                # set dense l1 regularization
                elif "dense" in layer.name:
                    if dense_reg == 0:
                        setattr(layer, attr, None)
                    else:
                        setattr(layer, attr, keras.regularizers.l1(dense_reg))

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

