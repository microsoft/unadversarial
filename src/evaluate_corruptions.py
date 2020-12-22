import pathlib
import sys

from torchvision.utils import save_image

curr_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(curr_path / 'better_corruptions'))

import argparse
import collections
import copy
import itertools
import os
from pathlib import Path

import cox.store
import cox.utils
import dill
import numpy as np
import robustness
import torch as ch
from fast_corruptions.configs import FIXED_CONFIG
from robustness import datasets, defaults, loaders, model_utils, train
from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10

from . import boosters
from . import constants
from .utils import LinearModel

BOOSTING_FP = 'boosting.ch'

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--boosting', choices=['none', 'class_consistent', 'qrcode', 'best_images'],
                    default='class_consistent',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--no-tqdm', type=int, default=1, choices=[0, 1],
                    help='Do not use tqdm.')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--workers', type=int, default=10)
parser.add_argument('--out-dir', type=str, default='/tmp')
parser.add_argument('--eval-only', type=int, default=0)
parser.add_argument('--exp-name', type=str, required=False)
parser.add_argument('--dataset', choices=['cifar', 'imagenet'],
                    default='imagenet')
parser.add_argument('--patch-size', type=int, default=70)
parser.add_argument('--pytorch-pretrained', action='store_true')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--adv-eval', type=int, default=0)
parser.add_argument('--model-path', type=str, required=True, 
                    help='Path to a checkpoint to load.')
parser.add_argument('--args-from-store', type=str, default=None,
                    help='arguments to read from existing store, comma separated.'
                        'e.g. data,dataset,arch,patch_size')
parser.add_argument('--zipped', action='store_true')
parser.add_argument('--apply-booster-transforms', type=int, default=1, choices=[0, 1],
                    help='Apply random transforms to the booster.')

### QRCode booster specific
parser.add_argument('--no-translation', type=int, default=0, choices=[0, 1],
                    help='Avoid translation of the patch. Useful for evaluating QRCodes' 
                    'since they are not robust to occlustions at all')
parser.add_argument('--qrcode-detector', type=str, choices=['cv2', 'pyzbar'], 
                    default='pyzbar', help='which qrcode detector to use.')

### Best images booster specific
parser.add_argument('--path-best-images', type=str, default='', 
                    help='Path to best images, required if --boosting with best_images')


"""
Example usage:

python evaluate_corruptions.py --arch resnet50 --dataset cifar --batch-size 64 --out-dir outdir 
        --exp-name tmp --patch-size 10 --model-path <path_to_boosted_model>
"""

def get_dataset_and_loaders(args):
    if args.dataset == 'cifar':
        ds = datasets.CIFAR('/tmp')
    elif args.dataset == 'imagenet':
        ds = datasets.ImageNet(args.data)
        if args.zipped:
            ds.custom_class = 'Zipped'
    else:
        raise NotImplementedError

    # TODO: with_index
    train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size,
                                               val_batch_size=args.batch_size,
                                               workers=args.workers,
                                               data_aug=True,
                                               only_val=True)
    return ds, (train_loader, val_loader)

def get_boosted_model(args, ds):
    if args.boosting == 'qrcode':
        assert args.arch == 'None', 'With QRCodes, there is no model. Please pass "None" to --arch'
        model = boosters.DataAugmentedModel(boosters.QRCodeModel(ds.num_classes, 
                                        detector=args.qrcode_detector), ds.ds_name, [])
        booster = boosters.QRCodeBooster(ds.num_classes, constants.DS_TO_DIM[args.dataset],
                                        constants.PATCH_TRANSFORMS,args.patch_size,
                                        apply_transforms=args.apply_booster_transforms, 
                                        detector=args.qrcode_detector)
        return boosters.BoostedModel(model, booster, 'qrcode').cuda()

    is_pt_model = args.arch in constants.NAME_TO_ARCH and args.dataset == 'imagenet'
    arch = constants.NAME_TO_ARCH[args.arch](args.pytorch_pretrained) if is_pt_model else args.arch

    if arch == 'linear':
        arch = LinearModel(constants.DS_TO_CLASSES[args.dataset], constants.DS_TO_DIM[args.dataset])


    # Check if the checkpoint is of a BoostedModel or a robustness lib AttackerModel 
    checkpoint = ch.load(args.model_path, pickle_module=dill)
    sd = checkpoint['model']
    is_checkpoint_boosted = 'module.booster.patches' in sd.keys()

    kwargs = {'arch': arch, 'dataset': ds, 
              'resume_path': args.model_path if not is_checkpoint_boosted else None,
              'add_custom_forward': is_pt_model or args.arch=='linear',
              'pytorch_pretrained': args.pytorch_pretrained}

    model, _ = model_utils.make_and_restore_model(**kwargs)

    # Wrap the model wtith DataAugmentedModel even if there are not corruptions. 
    # For consistency when loading from checkpoints
    model = boosters.DataAugmentedModel(model, ds.ds_name, [])

    # don't pass checkpoint to train_model do avoid resuming for epoch, optimizers etc.
    if args.boosting == 'class_consistent':
        boosting_path = Path(args.out_dir) / BOOSTING_FP
        if boosting_path.exists():
            booster = ch.load(boosting_path)
        else:
            dim = constants.DS_TO_DIM[args.dataset]
            booster = boosters.ClassConsistentBooster(ds.num_classes, dim,
                                                      constants.PATCH_TRANSFORMS,
                                                      args.patch_size,
                                                      model, apply_transforms=args.apply_booster_transforms)

        model = boosters.BoostedModel(model, booster, None)
    elif args.boosting == 'best_images':
        dim = constants.DS_TO_DIM[args.dataset]
        booster = boosters.BestImageBooster(ds.num_classes, dim,
                                            constants.PATCH_TRANSFORMS,
                                            args.patch_size,
                                            args.path_best_images,
                                            apply_transforms=args.apply_booster_transforms)

        model = boosters.BoostedModel(model, booster, None)
    elif args.boosting == 'none':
        # assert args.eval_only
        model = boosters.BoostedModel(model, None, None)
    else:
        raise ValueError(f'boosting not found: {args.boosting}')

    if is_checkpoint_boosted:
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint of BoostedModel'{}' (epoch {})".format(args.model_path, checkpoint['epoch']))

    return model.cuda()


def evaluate(args, store):
    ds, (train_loader, val_loader) = get_dataset_and_loaders(args)

    if 'corruptions_eval' not in store.keys:
        store.add_table('corruptions_eval', {'type': str, 'severity': int, 'corr_acc': float})
    store.save_dir = store.save_dir + '_eval'
    if not os.path.exists(store.save_dir):
        os.makedirs(store.save_dir)
    
    model = get_boosted_model(args, ds) 
    # model.booster = None

    def iteration_hook(model, i, loop_type, inp, target):
        if model.module.booster is None:
            return
        if i % 50 == 0:
            inp, target = inp.cuda(), target.cuda()
            example_boosted = model.module.booster(inp, target)
            bs_path = Path(store.save_dir) / f'boosted_{i}.jpg'
            save_image(example_boosted[:4], bs_path)
            example_adversaried = model.module.boosted_model.apply(example_boosted)
            inp_path = Path(store.save_dir) / f'inp_{i}.jpg'
            adv_path = Path(store.save_dir) / f'adv_{i}.jpg'
            save_image(inp[:4], inp_path)
            save_image(example_adversaried[:4], adv_path)
            if i == 0:
                print(f'Saved in {store.save_dir}')

    args.iteration_hook = iteration_hook

    with ch.no_grad():
        # evaluate on corrupted boosted dataset
        for corr in FIXED_CONFIG[ds.ds_name]:
            for severity in range(1,6):
                print('---------------------------------------------------')
                print(f"Dataset: {args.dataset} | Model: {args.arch} | Corruption: {corr} | Severity: {severity}")
                print('---------------------------------------------------')
                model.boosted_model.augmentations = [corr]
                model.boosted_model.severity = severity
                result = train.eval_model(args, model, val_loader, store=None)
                store['corruptions_eval'].append_row({'type': corr[:14], 'severity': severity, 'corr_acc': result['nat_prec1']})

        # evlautate on the clean boosted dataset
        print('---------------------------------------------------')
        print(f"Dataset: {args.dataset} | Model: {args.arch} | Corruption: None | Severity: N/A")
        print('---------------------------------------------------')
        model.boosted_model.augmentations = []
        result = train.eval_model(args, model, val_loader, store=None)
        store['corruptions_eval'].append_row({'type': 'clean', 'severity': -1, 'corr_acc': result['nat_prec1']})


if __name__ == "__main__":
    args = parser.parse_args()

    if args.adv_train and args.eps == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    assert not args.adv_train, 'not supported yet slatta dog'

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Useful for evaluation QRCodes since they are not robust to occlustions at all
    if args.no_translation: 
        constants.PATCH_TRANSFORMS['translate'] = (0., 0.)

    # Preprocess args
    args = defaults.check_and_fill_args(
        args, defaults.CONFIG_ARGS, datasets.DATASETS[args.dataset])
    args = defaults.check_and_fill_args(
        args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[args.dataset])

    store = cox.store.Store(args.out_dir, args.exp_name)
    if args.args_from_store:
        args_from_store = args.args_from_store.split(',')
        df = store['metadata'].df
        print(f'==>[Reading from existing store in {store.path}]')
        for a in args_from_store:
            if a not in df:
                raise Exception(f'Did not find {a} in the store {store.path}')
            setattr(args,a, df[a][0])
            print(f'==>[Read {a} = ({getattr(args, a)}) ')

    if 'metadata_eval' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata_eval', schema)
        store['metadata_eval'].append_row(args_dict)
    else:
        print('[Found existing metadata_eval in store. Skipping this part.]')

    evaluate(args, store)
