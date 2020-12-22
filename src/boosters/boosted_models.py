from torch import nn
import torch as ch
from fast_corruptions import gpu_corruptions
from fast_corruptions.configs import FIXED_CONFIG

class BoostedModel(nn.Module):
    def __init__(self, model, booster, training_mode):
        super().__init__()
        assert not isinstance(model, nn.DataParallel)
        self.boosted_model = model
        self.booster = booster
        self.training_mode = training_mode
        self.dummy = ch.nn.Parameter(ch.ones(()), requires_grad=True)

    def train(self, mode=True):
        if self.training_mode != 'booster' or not mode:
            return super(BoostedModel, self).train(mode)

        self.training = mode
        if self.booster is not None:
            self.booster.train()
        self.boosted_model.eval()

    def forward(self, inp, target, use_boost=True, save_dir=None, *args, **kwargs):
        inp = inp + self.dummy * ch.zeros(())
        if use_boost and self.booster is not None:
            new_inp = self.booster(inp, target, save_dir=save_dir)
            new_inp = ch.clamp(new_inp, 0., 1.)
        else:
            new_inp = inp

        return self.boosted_model(new_inp, target=target, *args, **kwargs)

class DataAugmentedModel(nn.Module):
    def __init__(self, model, dataset_name, augmentations, severity=3):
        super().__init__()
        self.model = model
        self.augmentations = augmentations

        self.severity = severity
        self.cfg = FIXED_CONFIG[dataset_name]

    def apply(self, inp):
        augmented = inp
        for aug in self.augmentations:
            corrupter = getattr(gpu_corruptions, aug)
            cfg = self.cfg[aug][self.severity - 1]
            augmented = corrupter(augmented, *cfg)

        return augmented

    def forward(self, inp, *args, **kwargs):
        augmented = self.apply(inp)
        return self.model(augmented, *args, **kwargs)