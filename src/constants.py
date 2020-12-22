from torchvision import models
from torch import nn
import kornia.augmentation as K

PATCH_TRANSFORMS = {
    'translate': (.3, .3),
    'degrees': 45.,
    'scale': (0.9, 1.1)
}

DS_TO_DIM = {
    'cifar': 32,
    'imagenet': 224,
    'solids': 224,
    'city': 224
}

DS_TO_CLASSES = {
    'cifar': 10,
    'imagenet': 1000
}

CORRUPTION_TYPES = [
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'fog',
    'frost',
    'gaussian_blur',
    'gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'saturate',
    'shot_noise',
    'snow',
    'spatter',
    'speckle_noise',
    'zoom_blur',
]

NAME_TO_ARCH = {
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'mnasnet': models.mnasnet1_0,
}

DS_NAME_TO_CORRUPTION_PATH = {
    'cifar10': '/mnt/ssd/datasets/CIFAR-10-C',
    'cifar10_boosted': '/mnt/ssd/datasets/CIFAR-10-boosted-C'
}

THREE_D_CORRUPTIONS = nn.Sequential(
    K.RandomPerspective(p=0.5)
)

SCENE_DICT = lambda custom_file: {
    "type": "scene",
    "myintegrator": {
        "type": "aov",
        "aovs": "chungus:uv",
        "sub_integrator": {
            "type": "direct"
        }
    },
    "myemitter": {
        "type": "constant",
        "radiance": 0.7
    },
    "myobject": {
        "type": "ply",
        "filename": custom_file,
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "bitmap",
                "filename": "/base_texture.jpg"
            }
        }
    }
}