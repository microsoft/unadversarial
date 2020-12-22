import numpy as np
from PIL import Image
import torch as ch
import torch.nn as nn
import torchvision.transforms as tvt
from . import make_center_mask
from kornia.augmentation import RandomAffine
from kornia.geometry import warp_affine
import os
from . import generic_booster as gb

def pad_tensor(tensor, final_dim):
    tensor_size = tensor.shape[-1]
    padded_tensor = ch.zeros(size=(tensor.shape[0], 3, final_dim, final_dim))
    center_y, center_x = final_dim//2, final_dim//2
    ystart = center_y - tensor_size//2
    xstart = center_x - tensor_size//2
    yend = ystart + tensor_size
    xend = xstart + tensor_size
    padded_tensor[:, :, ystart:yend, xstart:xend] = tensor
    return padded_tensor.to(tensor.device)


class BestImageBooster(gb.GenericBooster):
    def __init__(self, num_classes, image_dim, transforms, patch_size,
                 best_images_path, apply_transforms=True):
        super().__init__()
        to_tensor = tvt.ToTensor()
        to_PIL = tvt.ToPILImage()

        assert os.path.exists(best_images_path), f'Path "{best_images_path}" to best training images does not exist'
        best_images = ch.load(best_images_path).cpu()
        best_images = [to_tensor(to_PIL(best_image).resize((patch_size, patch_size))) for best_image in best_images]
        self.patches = ch.stack(best_images, 0)
        # self.patches = tvt.ToTensor()(tvt.functional.resize(best_images, patch_size))
        self.patches = pad_tensor(self.patches, image_dim).cuda()

        self.patch_size = patch_size
        self.image_dim = image_dim
        self.aff_transformer = RandomAffine(**transforms, return_transform=True, resample='nearest').cuda()
        self.apply_transforms = apply_transforms

    def forward(self, inp, target, save_dir=None, debug=True):
        bs, _, h, w = inp.shape
        masks = make_center_mask(self.image_dim, self.patch_size, bs)
        if debug: assert (masks.max() <= 1) and (masks.min() >= 0)
        patches = self.patches[target]

        if self.apply_transforms:
            masks, tx_fn = self.aff_transformer(masks)
            patches = warp_affine(patches, tx_fn[:,:2,:], dsize=(h, w))

        inp = (masks * patches) + (1 - masks) * inp
        return inp

    def output_vis(self, out_dir):
        pass
