# Fast and Calibrated Image Corruptions

## Benchmarking 

The ``fast_corruptions.benchmark`` script runs a suite of corruptions, times
them, evaluates a pretrained model on them, and optionally compares them to the
original ImageNet-C corrupted dataset.

Usage:
```
cd src/
python -m fast_corruptions.benchmark --imagenet-path /PATH/TO/IMAGENET \
                                     --corruption [gaussian_blur|glass_blur|...|all] \
                                     --severity X 
                                     [--compare]
                                     [--out-dir PATH]
                                     [--precomputed PATH/TO/IMAGENET-C] 
```

### Notes
- If the ``--compare`` argument is given, the corruption will be compared to its
  ImageNet-C counterpart.
- The ``--out-dir`` argument is where example images are saved from each
  corruption. If not specified, no example images will be saved. 
- The ``--precomputed`` argument can be used to pass the path to a
  pre-downloaded ImageNet-C dataset, which will be used as the "old" corruptions
  instead of recomputing them from the original code.

## Programmatic Usage

To use the library programatically:
```python
from fast_corruptions import gpu_corruptions
from fast_corruptions.configs import fixed_config
import torch as ch

severity = 1
cfg = fixed_config.CORRUPTIONS
x = ch.rand(10, 3, 224, 224).cuda() # Fake batch of images
tx = gpu_corruptions.fog(x.clone(), *cfg["fog"][severity - 1])
``` 