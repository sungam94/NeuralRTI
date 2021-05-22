import os
from scipy.misc import face
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from torchvision import transforms as tv

import pytorch_lightning as pl
# from pytorch_lightning.utilities.cli import LightningCLI
# from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

from PIL import Image
from typing import Optional
import utils

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
        parser.add_argument('--lp_name', type=str, default='dirs.lp')
        parser.add_argument('--data.batch_size', type=int, default=1)


class LpDataset(Dataset):
    reload = 1
    train_idx = None
    val_idx = None
    img_wh = None
    num_lights = None
    num_img = None
    num_train_img = None
    num_val_img = None

    def __init__(self, split='train', ratio=(200, 40)):
        print('init dataset')
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.split = split

        if LpDataset.reload:
            self.ratio = int(ratio[1] / ratio[1])
            # load all images from disk
            LpDataset.all_dir, LpDataset.all_rgb, LpDataset.img_wh = utils.load_lp_file(self.data_dir)
            LpDataset.num_light = len(LpDataset.all_dir)
            nl = LpDataset.num_light
            num_val = int(nl / (self.ratio+1))
            LpDataset.num_val_img = num_val
            val_idx = np.random.choice(nl, num_val, replace=False)
            LpDataset.val_idx = np.sort(val_idx)
            train_idx = list(set(range(nl)) - set(LpDataset.val_idx))
            LpDataset.num_train_img = len(train_idx)
            LpDataset.train_idx = train_idx
            LpDataset.reload = 0

        if self.split == 'train':
            self.train_rgb = [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            # self.train_rgb = torch.stack(self.train_rgb)
            self.train_rgb = torch.stack(self.train_rgb)
            n, c, h, w = self.train_rgb.size()
            self.train_rgb = self.train_rgb.permute(0, 1, 2, 3) # 120 3 85 128
            s = self.train_rgb.stride()
            self.train_rgb = torch.reshape(self.train_rgb, (w * h, n*c))
            ss = self.train_rgb.stride()
            # self.train_rgb = torch.stack(self.train_rgb)
            # self.train_rgb = torch.cat(self.train_rgb, -1)  # (h*w, num_img*3)
            self.train_dirs = [LpDataset.all_dir[i] for i in LpDataset.train_idx]
            self.train_dirs = torch.stack(self.train_dirs)
            print(f'train rgb size: {self.train_rgb.size()}')
            print(f'train rgb stride: {self.train_rgb.stride()}')
        elif self.split == 'val':
            self.eval_rgb = [LpDataset.all_rgb[i] for i in LpDataset.val_idx]
            self.eval_rgb = torch.stack(self.eval_rgb)
            n, c, h, w = self.eval_rgb.size()
            self.eval_rgb = self.eval_rgb.permute(0, 3, 2, 1) # leave in c h w for memory layout??
            self.eval_rgb = torch.reshape(self.eval_rgb, (w * h, n * c))
            s= self.eval_rgb.stride()
            self.eval_dirs = [LpDataset.all_dir[i] for i in LpDataset.val_idx]
            self.eval_dirs = torch.stack(self.eval_dirs)
            assert self.eval_dirs.size() == (LpDataset.num_val_img, 3)
            assert self.eval_rgb.size() == (self.img_wh[0] * self.img_wh[1], LpDataset.num_val_img * 3)
            print(f'eval rgb size: {self.eval_rgb.size()}')
            print(f'eval rgb stride: {self.eval_rgb.stride()}')

    def __getitem__(self, idx):
        if self.split == 'train':
            rnd_l_idx = torch.randint(0, LpDataset.num_train_img, (1,))
            gt = self.train_rgb[idx, (rnd_l_idx*3):(rnd_l_idx*3+3)],
            return self.train_dirs[rnd_l_idx], self.train_rgb[idx], gt, idx
        elif self.split == 'val':
            return self.eval_dirs[idx], self.eval_rgb, idx

    def __len__(self):
        if self.split == 'train':
            return (LpDataset.img_wh[0] * LpDataset.img_wh[1])
        elif self.split == 'val':
            return self.num_val_img

    def define_transforms(self):
        # Not sure why this is not in __init__
        self.transform = tv.ToTensor()


class LpDataModule(pl.LightningDataModule):
    def __init__(self,
                 # data_dir: str = "/home/mk301/RTI/loewenkopf",
                 batch_size: int = 1,
                 ratio: tuple = (1, 1)
                 ):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = batch_size
        print(f'lp_dm batch size: {batch_size}')
        self.setup()

        # ######## TESTS ########
        # w = LpDataset.img_wh[0]
        # h = LpDataset.img_wh[1]
        # print(f"image size: {w, h}")
        # # dir, a = self.data_train[0]
        # a = LpDataset.all_rgb[0]
        # a = a.permute(1,2,0)
        # plt.imshow(a)
        # plt.show()
        # return


    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(split='train')
        self.data_val = LpDataset(split='val')
        self.data_test = self.data_train

    def train_dataloader(self):
        print(f'train abtch size: {self.batch_size}')
        return DataLoader(self.data_train, batch_size=1024)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=1, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class NeuralRtiModule(pl.LightningModule):
    def __init__(self, x):
        super().__init__()
        self.save_hyperparameters()
        n_lights = 120
        n_coeff = 9
        # the first and last layers contain 3N units, N are number of input lights
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_coeff)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_coeff + 2, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, 3)
        )

    def forward(self, x, dir):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        embedding = torch.cat([embedding, dir], -1)
        decoded = self.decoder(embedding)
        assert len(decoded) == 3
        return decoded

    def training_step(self, batch, batch_idx):
        dir_batch, rgb_batch, gt_batch, pixel_idx = batch

        rgb_out = []
        for l_rgb, dir in zip(rgb_batch, dir_batch):
            l_rgb = torch.squeeze(l_rgb)
            dir = torch.squeeze(dir)[:2]
            rgb_out += [self(l_rgb, dir)]

        rgb_out = torch.stack(rgb_out)
        gt = torch.stack(gt_batch)
        # img = torch.reshape(rgb_out, (LpDataset.img_wh[1], LpDataset.img_wh[0], 3))
        # gt = torch.reshape(iv, (LpDataset.img_wh[1], LpDataset.img_wh[0], (iv.size()[1] // 3), 3))
        # gt = gt.permute(2, 0, 1, 3)
        # print(torch.max(img))
        loss = F.mse_loss(rgb_out, gt)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True)

        # tqdm_dict = {'train_loss': loss}
            # outputs = {
            #     'loss': loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # }
            # self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            # return outputs



    def validation_step(self, batch, batch_idx):
        # validation is run over whole Image Volume
        dir_batch, rgb_batch, light_idx = batch
        # rgb_batch is batch times image volume iv
        for iv, dir, idx in zip(rgb_batch, dir_batch, light_idx):
            iv = torch.squeeze(iv)
            dir = torch.squeeze(dir)[:2]
            rgb_out = []
            for rgb in iv:
                rgb_out += [self(rgb, dir)]
            rgb_out = torch.stack(rgb_out)
            rgb_out = rgb_out.cpu().detach()

            img =rgb_out
            print(f'\nrgb_out size {img.size()}')
            print(f'rgb_out stride {img.stride()}')

            img = torch.reshape(rgb_out, (LpDataset.img_wh[1], LpDataset.img_wh[0], 3))
            # iv = iv.cpu().detach()
            print(f'gt size {rgb_batch[0].size()}')
            print(f'gt stride {rgb_batch[0].stride()}')
            gt = torch.reshape(rgb_batch[0], (LpDataset.img_wh[1], LpDataset.img_wh[0], (iv.shape[1]//3), 3))

            gt = gt.permute(2, 0, 1, 3)
            print(f'gt size {gt.size()}')
            print(f'gt stride {gt.stride()}')

            gt = gt[idx]
            print(f'gt size 2 {gt.size()}')
            print(f'gt stride 2 {gt.stride()}')

            print(f'img size 3 {img.size()}')
            print(f'img stride 3 {img.stride()}')

            loss = F.mse_loss(img, gt)
            self.log('valid_loss', loss, on_step=True, prog_bar=True, logger=True)
            plt.imshow((img * 255).numpy().astype(np.uint8))
            plt.show()
            plt.imshow((gt * 255).numpy().astype(np.uint8))
            plt.show()
            # tqdm_dict = {'train_loss': loss}
            # outputs = {
            #     'loss': loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # }
            # self.log('loss', loss, on_step=True, prog_bar=True, logger=True)
            # return outputs


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main(args):
    # net_module = NeuralRtiModule()
    system = NeuralRtiModule(args)

    # checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
    #                                                             '{epoch:d}'),
    #                                       monitor='val/loss',
    #                                       mode='min',
    #                                       save_top_k=5,)

    # logger = TestTubeLogger(
    #     save_dir="logs",
    #     name=hparams.exp_name,
    #     debug=True,
    #     create_git_tag=False
    # )
    trainer = pl.Trainer(
                        max_epochs=args.num_epochs,
                      # checkpoint_callback=checkpoint_callback,
                      # resume_from_checkpoint=hparams.ckpt_path,
                      # logger=logger,
                      progress_bar_refresh_rate=1,
                      # gpus=hparams.num_gpus,
                      # distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=2,
                      benchmark=True,
                      # profiler=hparams.num_gpus==1
    )
    dm = LpDataModule(
        batch_size = 1,
        ratio = (1, 1)
    )
    trainer.fit(system, dm)

    # cli = MyLightningCLI(NeuralRtiModule, LpDataModule, seed_everything_default=1234)
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    import sys

    print(sys.argv)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
    parser.add_argument('--lp_name', type=str, default='dirs.lp')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=1)
    # parser.add_argument('--check_val_every_n_epoch', type=int, default= 1)

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # dl = LpDataset()
    # print(dl[0])
    cli_main(args)
