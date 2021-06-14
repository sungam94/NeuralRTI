import os

import colour
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import pytorch-ranger
# optimizer
from torch.optim import SGD, Adam
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau

from src.utils.optimizers import RAdam
from src.utils.visualization import *
from src.utils.warmup_scheduler import GradualWarmupScheduler


def get_optimizer(hparams, models, lr=None,):
    if lr is None:
        lr = hparams.lr
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = RAdam(parameters, lr=lr, eps=eps,
                          weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = Ranger(parameters, lr=lr, eps=eps,
                           weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer,
                                milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)

    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=hparams.num_epochs,
                                      eta_min=eps)

    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer,
                             lambda epoch: (1 - epoch / hparams.num_epochs) ** hparams.poly_exp)

    elif hparams.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      verbose=True,
                                      patience=hparams.patience,
                                      factor=hparams.factor,
                                      min_lr=hparams.min_lr,
                                      eps=hparams.eps)

    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=hparams.warmup_multiplier,
                                           total_epoch=hparams.warmup_epochs,
                                           after_scheduler=scheduler)

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def load_lp_file(data_dir, lp_filename='dirs.lp', img_wh=None):
    with open(data_dir + '/' + lp_filename) as f:
        data = f.read()
    lines = data.split('\n')
    num_lights = int(lines[0])
    image_paths = []
    all_dirs = []
    all_rgb = []

    transform = tv.transforms.ToTensor()

    image_transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]
    )

    for i, l in enumerate(lines):
        s = l.split(" ")
        if len(s) == 4:
            # read image files
            image_path = os.path.join(data_dir, s[0])
            image_paths += [image_path]
            # read light directions
            dir_T = torch.FloatTensor([list(map(float, s[1:4]))])
            all_dirs += [dir_T]

            # load image
            img = Image.open(image_path).convert('RGB')
            if img_wh is not None:
                img = img.resize(img_wh, Image.LANCZOS)
            else:
                img_wh = img.size

            # img - torch.from_numpy(img)
            img = transform(img)  # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0]
            img = img.permute(1, 2, 0)  # (H, W, C)
            # img = colour.RGB_to_YCbCr(img).astype(np.float32)

            # img = np.reshape(img, (img_wh[0] * img_wh[1], 3))  # (h*w, 3)
            # all_rgb += [torch.from_numpy(img)]
            all_rgb += [img]
    all_rgb = torch.stack(all_rgb)  # (n_img, C, H, W)
    all_dirs = torch.cat(all_dirs, 0)  # num_img*h*w, 3)
    return all_dirs, all_rgb, img_wh


def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()


if __name__ == '__main__':
    data_path = "/home/mk301/RTI/loewenkopf"
    print(load_lp_file(data_path))
