import torch as ch
from torch.nn.functional import interpolate
from torch.distributions import Uniform
from kornia import color, geometry, augmentation
from imgaug.augmenters import JpegCompression
from .utils import torch_to_np, np_to_torch, rgb2hsv, hsv2rgb
from .utils import elastic_transform as np_elastic_transform
import numpy as np
from PIL import Image
from torchvision import transforms
from io import BytesIO

def contrast(inp, contrast_level):
    means = inp.mean(dim=(2, 3), keepdims=True)
    return ch.clamp((inp - means) * contrast_level + means, 0, 1)

def brightness(inp, delta):
    #import ipdb; ipdb.set_trace()
    #hsv_inp = color.rgb_to_hsv(inp)
    hsv_inp = rgb2hsv(inp)
    hsv_inp[:,2,...] = ch.clamp(hsv_inp[:,2,...] + delta, 0, 1)
    #return ch.clamp(color.hsv_to_rgb(hsv_inp), 0, 1)
    return ch.clamp(hsv2rgb(hsv_inp), 0, 1)

def jpeg_compression(inp, quality):
    res = []
    to_im, to_tens = transforms.ToPILImage(), transforms.ToTensor()
    for x in inp.cpu():
        output = BytesIO()
        to_im(x).save(output, 'JPEG', quality=quality)
        res.append(to_tens(Image.open(output)))
    # aug = JpegCompression(100 - quality)
    # inp_np = (inp.detach().cpu().numpy() * 255).transpose(0, 2, 3, 1).astype(np.uint8)
    # jpeg_im = ch.tensor(aug(images=inp_np).transpose(0, 3, 1, 2).astype(np.float32) / 255.)
    return ch.stack(res) # jpeg_im.to(inp.device)

def pixelate(inp, scale):
    b, c, w, h = inp.shape
    inp = interpolate(inp, (int(w * scale), int(h * scale)), mode='bilinear')
    return interpolate(inp, (w, h), mode='nearest')

def elastic_transform(inp, strength, smoothness, warp_radius):
    np_arr = torch_to_np(inp)
    res = [np_elastic_transform(x, (strength, \
            smoothness, warp_radius)) for x in np_arr]
    return np_to_torch(np.array(res)).to(inp.device)