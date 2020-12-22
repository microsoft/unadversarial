from argparse import ArgumentParser
import subprocess
from pathlib import Path

IMAGENET_TRAIN = 'PATH_TO_TRAIN'
IMAGENET_VAL = 'PATH_TO_VAL'

parser = ArgumentParser()
parser.add_argument('--with-imagenet', action='store_true')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--gpus', required=True)
args = parser.parse_args()

def mount(_args, src, dst):
    if isinstance(src, Path): src = src.absolute()
    return _args + ['--mount', f'type=bind,source={src},target={dst}']

run_args = ['docker', 'run', '-it', '--network=host', '--ipc=host', '--rm']
curr_path = Path(__file__).parent.absolute()
run_args = mount(run_args, curr_path / '../src', '/src')

out_path = Path(args.out_dir).absolute()
run_args = mount(run_args, out_path, '/out')

if args.with_imagenet:
    run_args = mount(run_args, IMAGENET_TRAIN, '/imagenet/train')
    run_args = mount(run_args, IMAGENET_VAL, '/imagenet/test')

run_args = mount(run_args, curr_path / 'run_imagenet.sh', '/run_imagenet.sh')

run_args.extend(['--runtime=nvidia', '-e', f'NVIDIA_VISIBLE_DEVICES={args.gpus}'])
subprocess.run(run_args + ['TAG', 'zsh'])
