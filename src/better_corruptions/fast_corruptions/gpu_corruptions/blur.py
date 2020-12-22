import torch as ch 
import numpy as np
import math
from torch.distributions.uniform import Uniform
from kornia import filters, augmentation, geometry
from torch.nn import functional as F

from .utils import gaussian_motionfilter2d

def gaussian_blur(inp, blur_std):
    kernel_size = math.ceil((2 * blur_std) * 2)
    if kernel_size % 2 == 0: kernel_size += 1
    return filters.gaussian_blur2d(inp, 
                                   (kernel_size, kernel_size), 
                                   (blur_std, blur_std))

def glass_blur(inp, blur_std, rad, trials):
    inp = gaussian_blur(inp, blur_std)

    batch_size, _, w, h = inp.shape
    coords = ch.cartesian_prod(ch.arange(batch_size), 
                               rad + ch.arange(w - 2 * rad), 
                               rad + ch.arange(h - 2 * rad)).T 
    for _ in range(trials):
        # should probably be (-rad, rad+1) but this is how it's done in ImageNet-C
        xy_diffs = ch.randint(-rad, rad, (2, coords.shape[1]))
        new_coords = coords + ch.cat([ch.zeros(1, coords.shape[1]), xy_diffs]).long()

        # Swap coords and new_coords
        (b1, x1, y1), (b2, x2, y2) = coords, new_coords
        cp1, cp2 = inp[b1, :, x1, y1].clone(), inp[b2, :, x2, y2].clone()
        inp[b2, :, x2, y2] = cp1
        inp[b1, :, x1, y1] = cp2

    return ch.clamp(gaussian_blur(inp, blur_std), 0, 1)

def defocus_blur(inp, disk_radius, alias_blur):
    kernel_size = (3, 3) if disk_radius <= 8 else (5, 5)
    mesh_range = ch.arange(-max(8, disk_radius), max(8, disk_radius) + 1)
    X, Y = ch.meshgrid(mesh_range, mesh_range)

    aliased_disk = ((X.pow(2) + Y.pow(2)) <= disk_radius ** 2).float()
    aliased_disk /= aliased_disk.sum()
    kernel = filters.gaussian_blur2d(aliased_disk[None,None,...], kernel_size, 
                                     (alias_blur, alias_blur))[0]
    return ch.clamp(filters.filter2D(inp, kernel), 0, 1)

def motion_blur(inp, k_radius, k_stdev, angle_offset=0):
    n, c, h, w = inp.shape
    angle = angle_offset + Uniform(-45., 45.).sample((inp.shape[0],))
    clip_h, clip_v = [(fn(ch.deg2rad(angle)) * k_radius).round().long() for fn in (ch.cos, ch.sin)]
    # Move everything onto the right device
    clip_h, clip_v = clip_h.to(inp.device), clip_v.to(inp.device)

    kernel = gaussian_motionfilter2d(k_radius * 2 + 1, k_stdev, angle)
    inp = F.pad(inp, [k_radius] * 4, 'reflect')
    new_inp = filters.filter2D(inp, kernel, 'reflect', normalized=True)
    
    if ch.cuda.is_available():
        y_grid, x_grid = ch.meshgrid(ch.arange(h), ch.arange(w))
        grid = ch.stack((x_grid, y_grid), -1).repeat(n, 1, 1, 1).to(inp.device).float()
        grid[...,0] = (grid[...,0] + k_radius + clip_h.view(-1, 1, 1)) * 2 / (new_inp.shape[2] - 1) - 1
        grid[...,1] = (grid[...,1] + k_radius + clip_v.view(-1, 1, 1)) * 2 / (new_inp.shape[3] - 1) - 1
        res = F.grid_sample(new_inp, grid, 'nearest')
    else:
        res = ch.stack([im[:, k_radius+x:k_radius+x+h, k_radius+y:k_radius+y+w] \
                                    for im, y, x in zip(new_inp, clip_h, clip_v)])
    return ch.clamp(res, 0., 1.)

def zoom_blur(inp, max_zoom, zoom_step):
    zooms = ch.arange(1., max_zoom, zoom_step)
    n, c, h, w = inp.shape
    new_inp = ch.zeros_like(inp).cuda()
    for z in zooms:
        zoomed = geometry.transform.rescale(inp, z.item(), "bilinear")
        trim_top = (zoomed.shape[-1] - h) // 2
        new_inp += zoomed[:,:,trim_top:trim_top + h, trim_top:trim_top + h]
    return ch.clamp((inp + new_inp) / (zooms.shape[0] + 1), 0, 1)