from robustness import datasets, model_utils, loaders
from robustness.model_utils import make_and_restore_model as make_model
from robustness.tools.vis_tools import show_image_row
from matplotlib import pyplot as plt
from .ext_files import make_cifar_c, make_imagenet_c
from . import gpu_corruptions
from .gpu_corruptions.utils import torch_to_np, np_to_torch
from .configs import FIXED_CONFIG
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from timeit import Timer
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import dill

import torch as ch
import numpy as np

## Constants
IM_DIMS = {
    'imagenet': [3, 224, 224],
    'cifar': [3, 32, 32]
}

OLD_C_MAKERS = {
    'imagenet': make_imagenet_c,
    'cifar': make_cifar_c
}

## Multiprocessing stuff
def process_batch(in_tens, out_tens, in_q, out_q):
    t = out_tens.numpy()
    while True:
        req = in_q.get()
        if req == 'Done': break
        i, dataset, c_name, severity = req
        old_c = getattr(OLD_C_MAKERS[dataset], c_name)
        res = np_to_torch(np.stack([old_c(Image.fromarray(m), severity) \
                             for m in torch_to_np(in_tens[i][None,...])]))
        t[i] = res
        out_q.put(i)

def corruption_tx(c_func):
    def f(x, y):
        return c_func(x), y
    return f

def mp_corruption_tx(in_q, out_q, input_tens, output_tens, dataset, c_name, severity):
    def f(x, y):
        input_tens[:len(x)] = x
        [in_q.put((i, dataset, c_name, severity)) for i in range(len(x))]
        [out_q.get() for _ in range(len(x))]
        return output_tens[:len(x)].clone(), y
    return f

def start_server(num_threads, im_dims):
    ctx = mp.get_context("spawn")
    in_q, out_q = ctx.Queue(), ctx.Queue()
    input_tens = ch.zeros(*im_dims).share_memory_()
    output_tens = ch.zeros(*im_dims).share_memory_()
    proc_args = (input_tens, output_tens, in_q, out_q)
    ps = [ctx.Process(target=process_batch, args=proc_args) for _ in range(num_threads)]
    [p.start() for p in ps]
    return in_q, out_q, input_tens, output_tens

## Timing and logging
def time_and_return(f, x):
    t = Timer(lambda: f(x))
    return f(x), t.timeit(number=5) / 5.

def log_and_save(out_dir, name, ims, _time, acc):
    if out_dir is not None:
        show_image_row([ims.cpu()])
        plt.savefig(str(Path(out_dir) / f'{name}.png'))
        plt.close()

    print(f"Corruption: {name} | "
            f"Time (new): {_time:.4f}s for 10 images | "
            f"Model accuracy: {acc}")

## Setup tools
def model_and_dataset(args):
    ds = datasets.DATASETS[args.dataset](args.dataset_path)
    if args.dataset == 'cifar':
        model, _ = make_model(arch=args.arch, dataset=ds,
            resume_path='PATH_TO_MODEL')
    else:
        model, _ = make_model(arch=args.arch, dataset=ds, pytorch_pretrained=True)
    model.eval()
    return ds, model

def precomputed_loader(args, c_name):
    if args.dataset == 'imagenet':
        if_ds = ImageFolder(Path(args.precomputed) / c_name / str(args.severity), transform=ToTensor())
    elif args.dataset == 'cifar':
        ims = np_to_torch(np.load(str(Path(args.precomputed) / f'{c_name}.npy')))
        labs = ch.tensor(np.load(str(Path(args.precomputed) / 'labels.npy'))).long()
        ims, labs = [x[(args.severity-1)*10000:args.severity*10000] for x in [ims, labs]]
        if_ds = TensorDataset(ims, labs)

    loader = DataLoader(if_ds, batch_size=args.batch_size, shuffle=True, num_workers=20)
    return loader

## Evaluation
def eval_loader(model, corruption_fn, ds, loader, max_ims=None):
    tot_corr, tot = 0., 0.
    tqdm_tot = max_ims or len(loader)
    # it = tqdm(enumerate(loader), total=tqdm_tot)
    it = enumerate(loader)
    for i, (ims, labs) in it:
        ims = ims.cuda()
        tot_corr += model(ims)[0].argmax(1).eq(labs.cuda()).sum()
        tot += len(labs)
        if (max_ims is not None) and (i == max_ims):
            break
    return tot_corr / tot

def main(args):
    dataset, model = model_and_dataset(args)
    if args.threads > 1: 
        mp_args = start_server(args.threads, [args.batch_size] + IM_DIMS[args.dataset])
        print("Server started")

    _, vl = dataset.make_loaders(batch_size=10, workers=0, only_val=True)
    _, big_vl = dataset.make_loaders(batch_size=args.batch_size, 
                                     workers=20, only_val=True)
    fixed_im, _ = next(iter(vl))

    if args.out_dir is not None:
        show_image_row([fixed_im])
        plt.savefig(str(Path(args.out_dir) / 'base_figs.png'))

    CFG = FIXED_CONFIG[args.dataset]
    corruption_list = CFG if args.corruption == 'all' else [args.corruption]
    for c_name in corruption_list:
        severities = [1, 2, 3, 4, 5] if args.severity == 'all' else [int(args.severity)]
        print('=' * 40)
        for severity in severities:
            print('-' * 40)
            print(f"Corruption: {c_name} | Severity: {severity}")
            new_inp = fixed_im.clone().cuda()
            old_inp = fixed_im.clone()
            c_params = FIXED_CONFIG[args.dataset][c_name]
            if c_params is None: 
                print(f"Skipping corruption {c_name}...")
                continue

            ### NEW CORRUPTIONS
            new_fn = lambda x: getattr(gpu_corruptions, c_name)(x.cuda(), *c_params[severity - 1])
            new_res, new_time = time_and_return(new_fn, new_inp)
            c_loader = loaders.LambdaLoader(big_vl, corruption_tx(new_fn))
            new_acc = eval_loader(model, new_fn, dataset, c_loader, max_ims=args.subset)
            log_and_save(args.out_dir, f'{c_name}_new', new_res.cpu(), new_time, new_acc)

            if not args.compare: continue

            ### OLD CORRUPTIONS
            c_maker = {'imagenet': make_imagenet_c, 'cifar': make_cifar_c}[args.dataset]
            old_c = getattr(c_maker, c_name)
            old_fn = lambda x: np_to_torch(np.stack([old_c(Image.fromarray(m), severity) for m in torch_to_np(x)]))
            old_res, old_time = time_and_return(old_fn, old_inp)
            if args.precomputed is not None:
                loader = precomputed_loader(args, c_name)
            elif args.threads > 1:
                loader = loaders.LambdaLoader(big_vl, mp_corruption_tx(*mp_args, args.dataset, c_name, severity))
            elif args.threads == 1:
                loader = loaders.LambdaLoader(big_vl, corruption_tx(old_fn))

            old_acc = eval_loader(model, old_fn, dataset, loader, max_ims=args.subset) 
            log_and_save(args.out_dir, f'{c_name}_old', old_res, old_time, old_acc)
    [mp_args[0].put('Done') for _ in range(args.threads)]

if __name__ == '__main__':
    corruption_choices = list(FIXED_CONFIG['imagenet'].keys()) + ['all']
    parser = ArgumentParser()
    parser.add_argument('--arch', default='resnet50', help="Arch to evaluate")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--dataset', choices=['cifar', 'imagenet'])
    parser.add_argument('--resume-path', help='path to model checkpoint')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--corruption', choices=corruption_choices, default='all')
    parser.add_argument('--severity', required=True)
    parser.add_argument('--precomputed', help='path to ImageNet-C')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--compare', action='store_true', 
                        help='whether to compare to ImageNet-C/CIFAR-C')
    parser.add_argument('--subset', type=int, help='Number of iterations (batches) to evaluate')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
 
    with ch.no_grad():
        main(args)