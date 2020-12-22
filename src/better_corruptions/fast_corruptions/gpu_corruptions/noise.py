import torch as ch

"""
Noise corruptions
"""
def gaussian_noise(inp, std):
    return ch.clamp(inp + ch.randn_like(inp) * std, 0, 1)

def shot_noise(inp, c):
    return ch.clamp(ch.poisson(inp * c) / c, 0, 1)

def impulse_noise(inp, replace_frac):
    should_replace = ch.bernoulli(ch.ones_like(inp) * replace_frac).bool()
    bw_image = ch.bernoulli(ch.ones_like(inp) * 0.5)
    return ch.clamp(ch.where(should_replace, bw_image, inp), 0, 1)