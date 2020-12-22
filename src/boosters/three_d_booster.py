import torch
from torch import nn
import torch as ch
from ..utils import render
from ..constants import SCENE_DICT
from . import generic_booster as gb
from multiprocessing import Queue, Process, get_context
from kornia.augmentation import RandomAffine
from kornia.geometry import warp_affine
import dill
import time 
from itertools import cycle

from pathlib import Path
import os
from uuid import uuid4
from PIL import Image
import numpy as np

from robustness.tools import vis_tools
from matplotlib import pyplot as plt 

class ThreeDBooster(gb.GenericBooster):
    def __init__(self, num_classes, tex_size, image_size, 
                 batch_size, *, num_gpus, num_texcoords,
                 render_options, forward_render=False, init='const',
                 debug=False, custom_file=None, corruptions=None):
        super().__init__()
        if init == 'const':
           initialization = ch.zeros(num_classes, 3, tex_size, tex_size) + 0.5
        elif init == 'rand':
            initialization = ch.normal(mean=0.5, std=0.1, size=(num_classes, 3, tex_size, tex_size))
        else:
            raise ValueError('Unknown initialization for the booster.') 

        # ctx = get_context('spawn')
        ctx = get_context('fork')
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue()

        effective_batch = (batch_size // num_gpus + 1) * num_gpus 
        self.dones_sh_mem = ch.zeros(effective_batch).bool().share_memory_()
        self.texture_sh_mem = ch.zeros(effective_batch, tex_size*tex_size*3).share_memory_()
        self.render_sh_mem = ch.zeros(effective_batch, image_size, image_size, 4).share_memory_()
        self.uv_map_sh_mem = ch.zeros(effective_batch, image_size, image_size, 4).share_memory_()
        render_info = {
            "image_size": image_size,
            "samples": render_options['samples'],
            "scale_range": (render_options['min_zoom'], render_options['max_zoom']),
            "light_range": (render_options['min_light'], render_options['max_light'])
        }
        args = (SCENE_DICT(custom_file), render_info, self.in_q, self.out_q, 
            self.dones_sh_mem, self.texture_sh_mem, self.render_sh_mem, self.uv_map_sh_mem)
        gpus_to_use = cycle(['0', '1', '2', '3', '4']) # TODO: fix this
        Image.fromarray(np.zeros((tex_size, tex_size, 3)).astype(np.uint8)).save('/base_texture.jpg')
        for _ in range(num_texcoords):
            os.environ['CUDA_VISIBLE_DEVICES'] = next(gpus_to_use)
            p = ctx.Process(target=render.render, args=args)
            p.start()

        self.num_texcoords = num_texcoords
        self.textures = nn.Parameter(initialization, requires_grad=True)
        self.translate_tx = RandomAffine(degrees=0., p=1.0,
            translate=(0.4, 0.4), return_transform=True)
        self.tex_size = tex_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.debug = debug
        self.render_forward = forward_render
        self.corruptions = corruptions

    def diff_render_with_tex(self, tex_im, uv_im, bg_im):
        """
        Differentiable renderer.
        - uv_im (C x N x H x W)
        - tex_im (N x C x H x W)
        """
        uv_im = uv_im.permute(0, 3, 1, 2)
        N, C, H, W = uv_im.shape
        uv_im, tx_mat = self.translate_tx(uv_im)
        uv_im = uv_im.transpose(0, 1)
        assert C == 4, f"uv_im must have an alpha channel. Current shape: {uv_im.shape}"
        # Flatten everything to {C x BS x W*H}
        uv_flat = uv_im.view(C, N, -1)
        tex_flat = tex_im.transpose(0, 1).view(C-1, N, -1)
        bg_flat = bg_im.transpose(0, 1).view(C-1, N, -1)
        # Round the image to discrete value
        coord_im = (uv_im * self.tex_size).long()
        # Coords is {BS x (H*W)} mapping UV coords -> indices into tex_im
        coords = (coord_im[0] + coord_im[1] * self.tex_size).view(N, -1)
        tex_coords = ch.stack([tex_flat[:,i,coords[i]] for i in range(N)], dim=1)
        render = ch.where(uv_flat[-1] >= 1., tex_coords, bg_flat).transpose(0, 1)
        return render.view(N, C-1, H, W), tx_mat

    def step_booster(self, lr):
        step_direction = self.textures.grad.sign()
        step = (-lr) * step_direction
        self.textures.data.add_(step)
        self.textures.data.clamp_(0, 1)

    def forward(self, inp, target, save_dir=None):
        inp = inp.detach()
        # Shape variables
        bs, _, h, w = inp.shape
        gpu_num = inp.device.index
        max_bs = self.batch_size // self.num_gpus + 1
        ind = ch.zeros_like(target) if self.textures.shape[0] == 1 else target
        start, end = (gpu_num * max_bs, gpu_num * max_bs + bs)
        # Get texture to render
        tex_to_render = self.textures[ind,...].permute(0, 2, 3, 1)
        tex_to_render = tex_to_render.reshape(bs, -1).detach().cpu()
        self.texture_sh_mem[start:end] = tex_to_render
        [self.in_q.put(i) for i in range(start, end)]
        if self.debug: a = time.time()
        while not ch.all(self.dones_sh_mem[start:end]): pass
        if self.debug: print(f"Time spent waiting: {time.time() - a:.2f}s")
        self.dones_sh_mem[start:end] = False

        # Backwards pass / diff rendering
        uv_data = self.uv_map_sh_mem[start:end].to(inp.device)
        tex_to_render = self.textures[ind,...]
        renders, tx_mat = self.diff_render_with_tex(tex_to_render, uv_data, inp)
        renders = ch.clamp(renders, 0, 1)

        # Real rendering
        if self.render_forward:
            real_render = self.render_sh_mem[start:end].to(inp.device).detach()
            real_render = real_render.permute(0, 3, 1, 2).contiguous()
            real_render = warp_affine(real_render, tx_mat[:,:2,:], dsize=(h, w))
            real_render_alpha = real_render[:,3,...][:,None,...]
            real_render = real_render[:,:3] * real_render_alpha + inp * (1 - real_render_alpha)
            real_render = ch.clamp(real_render, 0, 1)
        else:
            real_render = renders

        if (save_dir is not None) and (inp.device.index == 0):
            vis_tools.show_image_row([real_render[:10].cpu()])
            plt.savefig(str(save_dir / "real_render_batch.png"))
            plt.close()
            if self.corruptions is not None:
                vis_tools.show_image_row([self.corruptions(real_render)[:10].cpu()])
                plt.savefig(str(save_dir / "real_render_batch_wc.png"))
                plt.close()
            vis_tools.show_image_row([renders[:10].cpu()])
            plt.savefig(str(save_dir / "diff_render_batch.png"))
            plt.close()
            self.output_vis(save_dir)

        res = renders - renders.detach() + real_render
        if self.corruptions is not None: 
            return self.corruptions(res)
        return res

    def output_vis(self, out_dir):
        def save_as(arr, fname, mode='RGBA'):
            Image.fromarray((arr.cpu().numpy() * 255).astype(np.uint8), mode).save(str(fname))

        with ch.no_grad():
            uv_data = self.uv_map_sh_mem[:1]
            render_data = self.render_sh_mem[:1]
            tex_to_render = self.textures[[0],...]
            renders, _ = self.diff_render_with_tex(tex_to_render.cpu(), uv_data.cpu(), 
                        ch.zeros(1, 3, self.image_size, self.image_size).cpu())

            save_as(uv_data[0], out_dir / "uv_map_ex.png")
            save_as(render_data[0], out_dir / "render_ex.png")
            save_as(renders[0].permute(1, 2, 0), out_dir / "diffrender_ex.png", "RGB")
            for i, tex in enumerate(self.textures.data.detach().cpu().numpy()):
                tex_im = Image.fromarray((tex.transpose(1, 2, 0) * 255).astype(np.uint8))
                tex_fname = str(out_dir / f"tex_{i}.png")
                tex_im.save(tex_fname)