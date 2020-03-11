from saved_models.vgg16_fcn.cifar10vgg import cifar10vgg
from saved_models.vgg16_fcn.cifar100vgg import cifar100vgg
from tensorflow.keras.models import Model, load_model

SAVED_MODELS_DIR = "./saved_models/"

def load_custom_model_for_ds(in_ds_name, model_type):
    model = None
    #**********************************************************
    if  in_ds_name=="MNIST" and model_type=="CUSTOM-MNIST":
        model = load_model(SAVED_MODELS_DIR+'mnist_custom_cnn.h5')
        if not model is None:
            print('The weights of CUSTOM-MNIST model was loaded.')
    #**********************************************************
    elif in_ds_name=="CIFAR10" and model_type=="VGG16":
        model_handle = cifar10vgg(False)
        model = model_handle.get_model()
        if not model is None:
            print('The weights of CIFAR10-VGG16 model was loaded.')
    #**********************************************************
    elif in_ds_name=="CIFAR100" and model_type=="VGG16":
        model_handle = cifar100vgg(False)
        model = model_handle.get_model()
        if not model is None:
            print('The weights of CIFAR100-VGG16 model was loaded.')
    elif in_ds_name=="CIFAR100" and model_type=="ResNet-V1-44":
        model = load_model(SAVED_MODELS_DIR+'cifar100_ResNet44v1_model.171.h5')
        if not model is None:
            print('The ResNet-V1-44 model for CIFAR100 was loaded.')

    elif in_ds_name=="CIFAR10" and model_type=="ResNet-V1-44":
        model = load_model(SAVED_MODELS_DIR+'cifar10_ResNet44v1_model.150.h5')
        if not model is None:
            print('The ResNet-V1-44 model for CIFAR10 was loaded.')
    return model 

