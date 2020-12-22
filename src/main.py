import pathlib
import sys

from torchvision.utils import save_image

curr_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(curr_path / 'better_corruptions'))

import argparse
import os
from pathlib import Path

import cox.store
import cox.utils
import dill
import json
import numpy as np
import torch as ch
from robustness import datasets, defaults, loaders, model_utils, train
from robustness.tools import breeds_helpers
from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10

from . import boosters, constants
from .utils import custom_datasets, LinearModel
from uuid import uuid4

# ch.set_default_tensor_type(ch.cuda.FloatTensor)

BOOSTING_FP = 'boosting.ch'

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--boosting', choices=['none', 'class_consistent', '3d'],
                    default='class_consistent',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--no-tqdm', type=int, default=1, choices=[0, 1],
                        help='Do not use tqdm.')
parser.add_argument('--exp-name', type=str, required=False)
parser.add_argument('--augmentations', type=str, default=None, help='e.g. fog,gaussian_noise')
parser.add_argument('--dataset', choices=['cifar', 'imagenet', 'entity13', 'living17', 'solids', 'city'],
                    default='imagenet')
parser.add_argument('--info-dir', type=str, help='Where to find (or download) info files for breeds')
parser.add_argument('--patch-size', type=int, default=70)
parser.add_argument('--training-mode', type=str, choices=['joint','model','booster'])
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--patch-lr', type=float, default=0.005)
parser.add_argument('--pytorch-pretrained', action='store_true')
parser.add_argument('--save-freq', type=int, default=50, 
                    help="How frequently we should save images")
parser.add_argument('--save-only-last', action='store_true',
                    help="Only keep the last visualizations instead of all")
parser.add_argument('--resume', action='store_true', 
                    help='Whether to resume training the DataAugmentedModel or not.' 
                        'Useful to continue training if job is pre-empted.'
                        '(Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default=None, 
                    help='Path to a checkpoint to load (useful for training a patch using a pretrained model).')
parser.add_argument('--zipped', action='store_true')
parser.add_argument('--apply-booster-transforms', type=int, default=1, choices=[0, 1],
                        help='Apply random transforms to the booster.')
parser.add_argument('--debug', action='store_true', help='Print debug stuff')
parser.add_argument('--json-config', help='Path to a JSON config file **that will override argparse args**')

## Arguments for 3D boosters:
parser.add_argument('--single-class', type=int, help="Whether to act "
            "in single-class mode. If given, will be used as a fixed "
            "target class (only optimize ONE texture across all images)")
parser.add_argument('--num-texcoord-renderers', default=1, type=int)
parser.add_argument('--forward-render', action='store_true',
    help="Use blender rendering on forward pass instead of matmul")
parser.add_argument('--add-corruptions', action='store_true',
    help="Add corruptions in the loop (see constants.py for details)")
# Render configuration
parser.add_argument('--render-samples', type=int, default=1)
parser.add_argument('--custom-file', help='If given, use object from file instead of Cube')
# Zoom (bigger = more zoomed out)
parser.add_argument('--min-zoom', type=int, default=20, help="Minimum zoom (i.e., most zoomed in)")
parser.add_argument('--max-zoom', type=int, default=40, help="Maximum zoom (i.e., most zoomed out)")
# Lighting
parser.add_argument('--min-light', type=float, default=0.5, help="Minimum lighting (darkest)")
parser.add_argument('--max-light', type=float, default=0.5, help="Maximum lighting (lightest)")

"""
Example usage:

python main.py --arch resnet50 --dataset cifar --batch-size 64 --out-dir outdir 
        --exp-name tmp --patch-size 10 --patch-lr 0.01 --training-mode joint
"""

def get_dataset_and_loaders(args):
    if args.dataset == 'solids':
        ds = datasets.ImageNet(args.data, 
            custom_class=custom_datasets.SolidColors,
            custom_class_args={'image_size': constants.DS_TO_DIM[args.dataset]})
    elif args.dataset == 'city':
        ds = datasets.ImageNet(args.data)
    elif args.dataset == 'cifar':
        ds = datasets.CIFAR('/tmp')
    elif args.dataset == 'imagenet':
        ds = datasets.ImageNet(args.data)
        if args.zipped:
            ds.custom_class = 'Zipped'
    elif args.dataset == 'entity13':
        split = breeds_helpers.make_entity13(args.info_dir)[1][0]
        ds = datasets.CustomImageNet(args.data, split)
    elif args.dataset == 'living17':
        split = breeds_helpers.make_living17(args.info_dir)[1][0]
        ds = datasets.CustomImageNet(args.data, split)
    else:
        raise NotImplementedError

    # TODO: with_index
    train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size,
                                               val_batch_size=args.batch_size,
                                               workers=args.workers,
                                               data_aug=True)
    return ds, (train_loader, val_loader)

def get_boosted_model(args, ds):
    is_pt_model = args.arch in constants.NAME_TO_ARCH and args.dataset == 'imagenet'
    arch = constants.NAME_TO_ARCH[args.arch](args.pytorch_pretrained) if is_pt_model else args.arch
    num_classes = 1 if args.single_class else ds.num_classes

    if arch == 'linear':
        arch = LinearModel(num_classes, constants.DS_TO_DIM[args.dataset])

    kwargs = {'arch': arch, 'dataset': ds, 'resume_path': args.model_path,
              'add_custom_forward': is_pt_model or args.arch=='linear',
              'pytorch_pretrained': args.pytorch_pretrained}

    model, _ = model_utils.make_and_restore_model(**kwargs)

    # Wrap the model wtith DataAugmentedModel even if there are not corruptions. 
    # For consistenct when loading from checkpoints
    model = boosters.DataAugmentedModel(model, ds.ds_name, 
                        args.augmentations.split(',') if args.augmentations else [])


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
                                                      model, 
                                                      apply_transforms=args.apply_booster_transforms)

        model = boosters.BoostedModel(model, booster, args.training_mode)
    elif args.boosting == '3d':
        boosting_path = Path(args.out_dir) / BOOSTING_FP
        if boosting_path.exists():
            booster = ch.load(boosting_path)
        else:
            dim = constants.DS_TO_DIM[args.dataset]
            render_options = {
                'min_zoom': args.min_zoom,
                'max_zoom': args.max_zoom,
                'min_light': args.min_light,
                'max_light': args.max_light,
                'samples': args.render_samples
            }
            corruptions = constants.THREE_D_CORRUPTIONS if args.add_corruptions else None
            booster = boosters.ThreeDBooster(num_classes=num_classes,
                                             tex_size=args.patch_size,
                                             image_size=dim,
                                             batch_size=args.batch_size,
                                             render_options=render_options,
                                             num_texcoords=args.num_texcoord_renderers,
                                             num_gpus=ch.cuda.device_count(),
                                             debug=args.debug,
                                             forward_render=args.forward_render,
                                             custom_file=args.custom_file,
                                             corruptions=corruptions)

        model = boosters.BoostedModel(model, booster, args.training_mode)
    elif args.boosting == 'none':
        # assert args.eval_only
        model = boosters.BoostedModel(model, None, args.training_mode)
    else:
        raise ValueError(f'boosting not found: {args.boosting}')

    return model.cuda()

def main_trainer(args, store):
    ds, (train_loader, val_loader) = get_dataset_and_loaders(args)
    if args.single_class is not None:
        print(f"Boosting towards a single class {args.single_class}")
        # Transform everything to have the same label
        class_tx = lambda x, y: (x, ch.ones_like(y) * args.single_class)
        train_loader = loaders.LambdaLoader(train_loader, class_tx)
        val_loader = loaders.LambdaLoader(val_loader, class_tx)

    model = get_boosted_model(args, ds) 

    # Resume traing the boosted model from a checkpoint
    resume_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    checkpoint = None
    if args.resume and os.path.isfile(resume_path):
        print('[Resuming training BoostedModel from a checkpoint...]')
        checkpoint = ch.load(resume_path, pickle_module=dill)

        sd = checkpoint['model']
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint of BoostedModel'{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    print(f"Dataset: {args.dataset} | Model: {args.arch}")

    if args.eval_only:
        print('==>[Evaluating the model]')
        return train.eval_model(args, model, val_loader, store=store)

    parameters = [model.dummy] # avoids empty param list to optimizer when optimizing the booster alone
    if args.training_mode in ['joint', 'model']:
        parameters = model.boosted_model.parameters()

    def iteration_hook(model, i, loop_type, inp, target):
        if loop_type == 'val' or model.module.booster is None:
            return
        if args.training_mode in ['booster', 'joint']:
            model.module.booster.step_booster(lr=args.patch_lr)
        if i % args.save_freq == 0:
            save_dir = Path(store.save_dir)
            #TODO: Move this part inside the 2D boosters. It is 
            # a bit tricky cause if we do that, we cannot save the "corrupted"
            # boosted images, but only the boosted images
            if args.boosting != '3d':
                inp, target = inp.cuda(), target.cuda()
                example_boosted = model.module.booster(inp, target)
                bs_path = save_dir / f'boosted_{i}.jpg'
                save_image(example_boosted[:4], bs_path)
                example_adversaried = model.module.boosted_model.apply(example_boosted)
                inp_path = save_dir / f'inp_{i}.jpg'
                adv_path = save_dir / f'adv_{i}.jpg'
                save_image(inp[:4], inp_path)
                save_image(example_adversaried[:4], adv_path)
            else:
                if not args.save_only_last:
                    save_dir = save_dir / f'iteration_{i}'
                    os.makedirs(save_dir)
                with ch.no_grad():
                    model(inp, target, save_dir=save_dir)
            if i == 0:
                print(f'Saved in {store.save_dir}')

    args.iteration_hook = iteration_hook

    return train.train_model(args, model, (train_loader, val_loader),
                             store=store, checkpoint=checkpoint, 
                             update_params=parameters)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.json_config is not None:
        print("Overriding args with JSON...")
        new_args = json.load(open(args.json_config))
        assert all(hasattr(args, k) for k in new_args), set(new_args.keys()) - set(vars(args).keys())
        for k in new_args: setattr(args, k, new_args[k]) 

    assert not args.adv_train, 'not supported yet slatta dog'
    assert args.training_mode is not None, "training_mode is required"

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    if args.exp_name == 'random':
        args.exp_name = str(uuid4())
        print(f"Experiment name: {args.exp_name}")
    assert args.exp_name != None

    # Preprocess args
    default_ds = args.dataset if args.dataset in datasets.DATASETS else "imagenet"
    args = defaults.check_and_fill_args(
        args, defaults.CONFIG_ARGS, datasets.DATASETS[default_ds])
    if not args.eval_only:
        args = defaults.check_and_fill_args(
            args, defaults.TRAINING_ARGS, datasets.DATASETS[default_ds])
    if False and (args.adv_train or args.adv_eval):
        args = defaults.check_and_fill_args(
            args, defaults.PGD_ARGS, datasets.DATASETS[default_ds])
    args = defaults.check_and_fill_args(
        args, defaults.MODEL_LOADER_ARGS, datasets.DATASETS[default_ds])

    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')

    print(args)
    main_trainer(args, store)
