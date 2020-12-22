from torch import nn
import torch as ch
from kornia.augmentation import RandomAffine
from robustness.tools.vis_tools import show_image_row
from kornia.geometry import warp_affine
from . import generic_booster as gb

class ClassConsistentBooster(gb.GenericBooster):
    def __init__(self, num_classes, image_dim, transforms, patch_size,
                 model=None, apply_transforms=True, init='rand'):
        super().__init__()
        if init == 'const':
           initialization = ch.zeros(num_classes, 3, image_dim, image_dim) + 0.5
        elif init == 'rand':
            initialization = ch.normal(mean=0.5, std=0.1, size=(num_classes, 3, image_dim, image_dim))
        else:
            raise ValueError('Unknown initialization for the booster.') 

        self.patches = nn.Parameter(initialization, requires_grad=True)
        self.patch_size = patch_size
        self.image_dim = image_dim
        self.aff_transformer = RandomAffine(**transforms, return_transform=True).cuda()
        self.apply_transforms = apply_transforms

    def step_booster(self, lr):
        step_direction = self.patches.grad.sign().detach()
        step = (-lr) * step_direction
        self.patches.data.add_(step)
        self.patches.data.clamp_(0, 1)

    def forward(self, inp, target, save_dir=None, debug=True):
        bs, _, h, w = inp.shape
        masks = make_center_mask(self.image_dim, self.patch_size, bs)
        if debug: assert (masks.max() <= 1) and (masks.min() >= 0)
        ind = ch.zeros_like(target) if self.patches.shape[0] == 1 else target
        patches = self.patches[ind]

        if self.apply_transforms:
            masks, tx_fn = self.aff_transformer(masks)
            patches = warp_affine(patches, tx_fn[:,:2,:], dsize=(h, w))

        inp = (masks * patches) + (1 - masks) * inp
        return inp
    
    def output_vis(self, out_dir):
        pass

def make_center_mask(mask_size, patch_size, bs):
    mask = ch.cuda.FloatTensor(size=(bs, 3, mask_size, mask_size)).zero_()
    center_y, center_x = mask_size//2, mask_size//2
    ystart = center_y - patch_size//2
    xstart = center_x - patch_size//2
    yend = ystart + patch_size
    xend = xstart + patch_size
    mask[:, :, ystart:yend, xstart:xend] = 1
    return mask