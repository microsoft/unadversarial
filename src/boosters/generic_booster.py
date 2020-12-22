from torch import nn

class GenericBooster(nn.Module):
    def __init__(self):
        super().__init__()

    def step_booster(self, lr):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def output_vis(self, out_dir):
        pass