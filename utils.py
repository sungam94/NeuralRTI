import os
import numpy as np
import torch
from PIL import Image
import torchvision as tv
import matplotlib.pyplot as plt


def load_lp_file(data_dir, lp_filename='dirs.lp', img_wh = None):
    with open(data_dir + '/' + lp_filename) as f:
        data = f.read()
    lines = data.split('\n')
    num_lights = int(lines[0])
    image_paths = []
    all_dirs = []
    all_rgb = []

    transform = tv.transforms.ToTensor()

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

            img = transform(img)  # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0]
            # img = img.permute(1, 2, 0)  # (H, W, C)
            # img = np.reshape(img, (img_wh[0] * img_wh[1], 3))  # (h*w, 3)
            all_rgb += [img]
    all_rgb = torch.stack(all_rgb) # (n_img, C, H, W)
    all_dirs = torch.cat(all_dirs, 0)  # num_img*h*w, 3)
    return all_dirs, all_rgb, img_wh

def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()

if __name__=='__main__':
    data_path = "/home/mk301/RTI/loewenkopf"
    print(load_lp_file(data_path))