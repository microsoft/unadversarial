from torchvision.datasets import VisionDataset
import torch as ch

class SolidColors(VisionDataset):
    def __init__(self, root, *args, image_size=None, num_images=int(1e5), **kwargs):
        super().__init__(root)
        self.N = num_images
        assert image_size is not None
        self.image_size = image_size
        self.samples = []
    
    def __getitem__(self, ind):
        rgb = ch.rand(3)
        new_im = ch.zeros(3, self.image_size, self.image_size)
        for i in range(3): new_im[i,...] = rgb[i]
        return new_im, ch.randint(0, 1000, ())

    def __len__(self):
        return self.N
