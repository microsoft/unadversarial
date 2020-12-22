'''
The only purpose of this script is comparing with the original ImageNet-C and
CIFAR-C corruptions and making sure that the images look similar and accuracies
look similar.
'''

# # Format: old_name: (new_name, cfg_sev_1, cfg_sev_2, ... cfg_sev_5)
_snow_params = [
    (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
    (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
    (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
    (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
    (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)
]

_elastic_params = [
    (244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
    (244 * 2, 244 * 0.08, 244 * 0.2),
    (244 * 0.05, 244 * 0.01, 244 * 0.02),
    (244 * 0.07, 244 * 0.01, 244 * 0.02),
    (244 * 0.12, 244 * 0.01, 244 * 0.02)
]

CORRUPTIONS = {
    'gaussian_noise': ((.08,), (.12,), (0.18,), (0.26,), (0.38,)),
    'shot_noise': ((60,), (25,), (12,), (5,), (3,)),
    'impulse_noise': ((.03,), (.06,), (.09,), (0.17,), (0.27,)),
    'glass_blur': ((0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)),
    'defocus_blur': ((3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)),
    'motion_blur': ((10, 3), (15, 5), (15, 8), (15, 12), (20, 15)),
    'zoom_blur': ((1.11, 0.01), (1.16, 0.01), (1.21, 0.02), (1.26, 0.02), (1.31, 0.03)), 
    'fog': ((1.6, 2), (2.1, 2), (2.6, 1.7), (2.5, 1.5), (3., 1.4)),
    'frost': ((1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)),
    'snow': tuple(_snow_params),
    'contrast': ((0.4,), (.3,), (.2,), (.1,), (.05,)),
    'brightness': ((.1,), (.2,), (.3,), (.4,), (.5,)),
    'jpeg_compression': ((25,), (18,), (15,), (10,), (7,)),
    'pixelate': ((0.6,), (0.5,), (0.4,), (0.32,), (0.29,)),
    'elastic_transform': tuple(_elastic_params)
}