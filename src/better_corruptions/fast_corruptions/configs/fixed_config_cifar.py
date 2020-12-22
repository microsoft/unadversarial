'''
The only purpose of this script is comparing with the original ImageNet-C and
CIFAR-C corruptions and making sure that the images look similar and accuracies
look similar.
'''

# Format: old_name: (new_name, cfg_sev_1, cfg_sev_2, ... cfg_sev_5)
_snow_params = [
    (0.1, 0.2, 1,    0.6, 7,  0.95),
    (0.1, 0.2, 1,    0.5, 9,  0.9),
    (0.15,0.35, 1.75, 0.55,9,  0.9),
    (0.25,0.35, 2.25, 0.6, 13, 0.85),
    (0.3, 0.35, 1.25, 0.65,25, 0.6)
]

IMSIZE = 32
_elastic_params = [
    (IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
    (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
    (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
    (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
    (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)
]

CORRUPTIONS = {
    'gaussian_noise': ((0.04,), (0.06,), (.08,), (.09,), (.10,)),
    'shot_noise': ((500,), (250,), (100,), (75,), (50,)),
    'impulse_noise': ((.01,), (.02,), (.03,), (.05,), (.07,)),
    'glass_blur': ((0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)),
    'defocus_blur': ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)),
    'motion_blur': ((5, 0.6), (5, 1.0), (5, 1.5), (4, 1.5), (4, 2.)), # Changed
    'zoom_blur': ((1.06, 0.01), (1.11, 0.01), (1.16, 0.01), (1.21, 0.01), (1.26, 0.01)), 
    'fog': ((.3, 3), (.6, 3), (0.85, 2.5), (1.1,2), (1.6, 1.75)),
    'frost': ((1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)),
    'snow': tuple(_snow_params),
    'contrast': ((.75,), (.5,), (.4,), (.3,), (0.15,)),
    'brightness': ((.05,), (.1,), (.15,), (.2,), (.3,)),
    'jpeg_compression': ((80,), (65,), (58,), (50,), (40,)),
    'pixelate': ((0.95,), (0.9,), (0.85,), (0.75,), (0.65,)),
    'elastic_transform': tuple(_elastic_params)
}