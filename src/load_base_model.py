class Load_Base_Model:
    def load_sota(model_name):
        """
        please refer to keras version and other deep learning dependencies
        """
        if model_name == "vgg16":
            from keras.applications.vgg16 import VGG16 as base
        if model_name == "vgg19":
            from keras.applications.vgg19 import VGG19 as base
        if model_name == "resnet50":
            from keras.applications.resnet50 import ResNet50 as base
        if model_name == "resnet101":
            from keras.applications.resnet50 import ResNet101 as base
        if model_name == "resnet152":
            from keras.applications.resnet50 import ResNet152 as base
        if model_name == "xception":
            from keras.applications.xception import Xception as base

        if model_name == "mobilenet":
            from keras.applications.mobilenet import MobileNet as base

        if model_name == "inceptionv3":
            from keras.applications.inception_v3 import InceptionV3 as base
        if model_name == "inceptionresnetv2":
            from keras.applications.inception_resnet_v2 import InceptionResNetV2 as base
        if model_name == "densenet121":
            from keras.applications.densenet import DenseNet121 as base
        if model_name == "densenet169":
            from keras.applications.densenet import DenseNet169 as base
        if model_name == "nasnetlarge":
            from keras.applications.nasnet import NASNetLarge as base
        if model_name == "nasnetmobile":
            from keras.applications.nasnet import NASNetMobile as base
        if model_name == "EfficientNetB0":
            from efficientnet import EfficientNetB0 as base
        if model_name == "EfficientNetB1":
            from efficientnet import EfficientNetB1 as base
        if model_name == "EfficientNetB2":
            from efficientnet import EfficientNetB2 as base
        if model_name == "EfficientNetB3":
            from efficientnet import EfficientNetB3 as base
        if model_name == "EfficientNetB4":
            from efficientnet import EfficientNetB4 as base
        return base


def get_base_model(backbone, imgshape, weights="imagenet"):
    """
    :param imgshape: refer to the research papers
    :param weights: pre-trained model weights, most set to "imagenet"
    """
    base = Load_Base_Model.load_sota(backbone)
    model = base(include_top=False, input_shape=(imgshape, imgshape, 3), weights=weights, pooling="avg")
    return model
