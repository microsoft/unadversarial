import cv2
try:
    from pyzbar import pyzbar
    import qrcode
except:
    pass
import numpy as np
from PIL import Image
import torch as ch
import torch.nn as nn
import torchvision.transforms as tvt
from . import make_center_mask
from kornia.augmentation import RandomAffine
from kornia.geometry import warp_affine
from . import generic_booster as gb

class QRCodeModel(ch.nn.Module):
    def __init__(self, num_classes, detector='pyzbar'):
        super().__init__()
        self.num_classes = num_classes
        self.possible_classes = list(range(num_classes))
        self.detector = 'pyzbar'

    def forward(self, x, *args, **kwargs):
        preds = ch.zeros(x.shape[0], self.num_classes).cuda()
        for i,im in enumerate(x.cpu()):
            qrcode_image = 255*np.uint8(im.permute(1, 2, 0).numpy())
            try:
                if self.detector == 'cv2':
                    y_pred, _, _ = self.qrCodeDetector.detectAndDecode(qrcode_image)
                elif self.detector == 'pyzbar':
                    y_pred = pyzbar.decode(qrcode_image)[0].data.decode("utf-8")
                if y_pred is not None:
                    y_pred = int(y_pred)
                    if y_pred not in self.possible_classes: raise Exception()
            except:
                y_pred = np.random.randint(0, self.num_classes)
            preds[i, y_pred] = 1 

        return preds, None # return a tuple to fit with the robustness lib
      

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


class QRCodeBooster(gb.GenericBooster):
    def __init__(self, num_classes, image_dim, transforms, patch_size,
                 apply_transforms=True, detector='pyzbar'):
        super().__init__()
        to_tensor = tvt.ToTensor()
        self.detector = detector
        if detector == 'cv2':
            self.qrCodeDetector = cv2.QRCodeDetector()
        
        self.patches = [to_tensor(qrcode.make(str(i), border=1).convert('RGB').resize((patch_size, patch_size), Image.NEAREST)) \
                            for i in range(num_classes)]
        self.patches = ch.stack(self.patches, 0)
        self.patches = pad_tensor(self.patches, image_dim).cuda()

        for i,p in enumerate(self.patches):
            qrcode_image = 255*np.uint8(p.permute(1, 2, 0).cpu().numpy())
            if detector == 'cv2':
                ## opencv qr detector is really crappy
                decoded_class, _, _ = self.qrCodeDetector.detectAndDecode(qrcode_image)
            elif detector == 'pyzbar':
                decoded_class = pyzbar.decode(qrcode_image)[0].data.decode("utf-8") 
            else:
                raise Exception('Unknown QRCode detector')
            assert decoded_class == str(i), f'The decoded QR code for class {i} is wrong.' \
                                        'Probably something weird happend during processing.'

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
