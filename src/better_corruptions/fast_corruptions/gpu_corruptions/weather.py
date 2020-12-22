import torch as ch
import numpy as np
import math
from torch.distributions.uniform import Uniform 
from kornia import geometry, augmentation, color, filters
from pathlib import Path
from .utils import plasma_fractal, gaussian_motionfilter2d
from .blur import motion_blur

PRECOMP_FROST = ch.load(Path(__file__).parent / '../ext_files/frost.pt')

def fog(inp, fog_mixin, wibble_decay):
    b, c, w, h = inp.shape
    nearest_pow = int(2 ** math.ceil(math.log2(max(w, h))))
    max_val = inp.view(inp.shape[0], -1).max(1).values.view(-1, 1, 1, 1)
    np_fractal = np.stack([plasma_fractal(nearest_pow, wibble_decay) for _ in range(b)])
    fractal = ch.tensor(np_fractal[:, :w, :h][:,None,...])
    inp += fog_mixin * fractal.cuda()
    return ch.clamp(inp * max_val / (max_val + fog_mixin), 0, 1)

def frost(inp, inp_coeff, frost_coeff):
    idx = ch.randint(len(PRECOMP_FROST), size=(len(inp),))
    frost_figs = PRECOMP_FROST[idx].cuda()
    cropper = augmentation.RandomCrop((inp.shape[2], inp.shape[3])).cuda()
    return ch.clamp(inp_coeff * inp + \
                    frost_coeff * cropper(frost_figs), 0, 1)

def snow(inp, snow_mean, snow_std, snow_zoom, snow_thresh,
         snow_kernel_rad, snow_kernel_std, snow_mixin):
    snow_shape = [inp.shape[0], 1, inp.shape[2], inp.shape[3]]
    snow_layer = ch.randn(*snow_shape).cuda() * snow_std + snow_mean
    # Scaling
    zoomed = geometry.transform.rescale(snow_layer, float(snow_zoom))
    trim_top = (zoomed.shape[-1] - inp.shape[-1]) // 2
    snow_layer = zoomed[:,:,trim_top:trim_top + inp.shape[-1], trim_top:trim_top + inp.shape[-1]]
    
    # Thresholding
    snow_layer[snow_layer < snow_thresh] = 0.
    snow_layer = motion_blur(snow_layer, snow_kernel_rad, snow_kernel_std, angle_offset=-90.)

    grayscaled_inp = ch.max(inp, color.rgb_to_grayscale(inp) * 1.5 + 0.5)
    inp = snow_mixin * inp + (1 - snow_mixin) * grayscaled_inp 
    return ch.clamp(inp + snow_layer + geometry.transform.rot180(snow_layer), 0, 1)