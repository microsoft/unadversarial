import torch as ch
import dill
from IPython import embed
from torchvision import transforms
from os import path

CLASS = 0
PATCH_SIZE = 150 # 25,50,100,150
ROBUST = True


to_pil = transforms.ToPILImage()
center_crop = transforms.CenterCrop(PATCH_SIZE)

base_path = 'BASEPATH'

eps = '3' if ROBUST else '0'
model_path = f'patchsize_{PATCH_SIZE}_patchlr_0.001_arch_resnet18_booster_resnet18_l2_eps{eps}.ckpt/checkpoint.pt.best'

checkpoint = ch.load(path.join(base_path, model_path), pickle_module=dill)
sd = checkpoint['model']
patches = sd['module.booster.patches'].cpu()

im = center_crop(to_pil(patches[CLASS]))
im.save(f'class_{CLASS}.png')
