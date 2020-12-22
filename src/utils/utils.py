from PIL import Image
from torchvision import transforms
import torch as ch
import numpy as np
from os import remove
try:
    from OpenEXR import InputFile
    from .exr import channels_to_ndarray
except:
    pass
"""
File for general utilities:
- Image saving and loading
- Linear model prototype
"""

#################### IMAGE SAVE AND LOAD ####################
def exr_to_np(fname, rem=False):
    return channels_to_ndarray(InputFile(fname), "RGBA")

ttx = transforms.ToTensor()
def read_png(fname):
    return np.array(Image.open(fname)) / 255.
    #return ttx(Image.open(fname))#ch.tensor(Image.open(fname).asarray())

piltx = transforms.ToPILImage()
def save_png(tens, fname):
    im = piltx(tens)
    im.save(fname)

#################### LINEAR MODEL ####################
class LinearModel(ch.nn.Module):
   def __init__(self, num_classes=10, input_dim=32):
      super().__init__()
      self.fc1 = ch.nn.Linear(input_dim**2*3, num_classes)
        
   def forward(self, x, *args, **kwargs):
      out = x.reshape(x.shape[0], -1)
      return self.fc1(out)

