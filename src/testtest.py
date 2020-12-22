from robustness import model_utils, datasets
import torch as ch
from tqdm.auto import tqdm 
from torchvision import models

# ds = datasets.CIFAR('/tmp/')
ds = datasets.ImageNet('/home/hasalman/datasets/IMAGENET/imagenet')

for ARCH in ['mnasnet',
            # 'shufflenet',
            # 'vgg16_bn',
            # 'densenet',
            # 'resnext50_32x4d',
            # 'mobilenet',
            ]:
    for EPS in [0, 3]:
        print(f'ARCH: {ARCH}     |     EPS: {EPS}')
        # model_path = '../notebooks/cifar_l2_0_25.pt'
        # model_path = '../notebooks/cifar_l2_1_0.pt'
        # model_path = '../notebooks/cifar_nat.pt'
        # model_path = 'msnet.ckpt'
        # model_path = 'checkpoint.pt.best'
        model_path = f'/home/hasalman/azure-madrylab/imagenet_experiments/opensource_models/{ARCH}_l2_eps{EPS}.ckpt'
        # model_path = None

        pytorch_models = {
            'alexnet': models.alexnet,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'squeezenet': models.squeezenet1_0,
            'densenet': models.densenet161,
            # 'inception': models.inception_v3,
            # 'googlenet': models.googlenet,
            'shufflenet': models.shufflenet_v2_x1_0,
            'mobilenet': models.mobilenet_v2,
            'resnext50_32x4d': models.resnext50_32x4d,
            'mnasnet': models.mnasnet1_0,
        }

        model, checkpoint = model_utils.make_and_restore_model(arch=pytorch_models[ARCH](True), dataset=ds, 
                                                            resume_path=model_path, add_custom_forward=True)
        # model, checkpoint = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path, add_custom_forward=False)
        train_loader, val_loader = ds.make_loaders(batch_size=64, workers=4)

        correct = 0
        model.eval()
        with ch.no_grad():
            for X,y in tqdm(val_loader):
                X,y = X.cuda(), y.cuda()
                out = model(X, with_image=False)
                _, pred = out.topk(1,1)
                correct += (pred.squeeze()==y).detach().cpu().sum()
        print(f'The clean accuracy is {1.*correct/len(val_loader.dataset)*100.}%')