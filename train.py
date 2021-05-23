import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from torchvision import transforms as tv
import torch.autograd.profiler as profiler
import pytorch_lightning as pl

from typing import Optional
import utils


class LpDataset(Dataset):
    reload = 1
    train_idx = None
    test_idx = None
    img_wh = None
    num_train_img = None
    num_test_img = None
    train_dirs = None
    n_l_dirs = None

    def __init__(self, split='train', ratio=(1, 1)):
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.split = split

        if LpDataset.reload:
            ration_sum = sum(ratio)

            LpDataset.ratio = int(ratio[0] / ratio[1])
            # load all images from disk
            LpDataset.all_dir, LpDataset.all_rgb, LpDataset.img_wh = utils.load_lp_file(self.data_dir)
            LpDataset.n_l_dirs = len(LpDataset.all_dir)
            nl = LpDataset.n_l_dirs
            part_size = nl // ration_sum
            LpDataset.num_test_img = part_size * ratio[1]
            LpDataset.test_idx = np.random.choice(nl, LpDataset.num_test_img, replace=False)
            LpDataset.test_idx = np.sort(LpDataset.test_idx)
            LpDataset.train_idx = list(set(range(nl)) - set(LpDataset.test_idx))
            LpDataset.num_train_img = len(LpDataset.train_idx)
            LpDataset.train_dirs = torch.stack([LpDataset.all_dir[i] for i in LpDataset.train_idx])
            LpDataset.reload = 0

        if self.split == 'train':
            self.train_rgb = [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            self.train_rgb = torch.stack(self.train_rgb)
            n, c, h, w = self.train_rgb.size()
            self.train_rgb = self.train_rgb.permute(0, 1, 2, 3)
            self.train_rgb = torch.reshape(self.train_rgb, (w * h, n*c))
        elif self.split == 'eval':
            self.eval_rgb = torch.stack([LpDataset.all_rgb[i] for i in LpDataset.train_idx])
            n, c, h, w = self.eval_rgb.size()
            self.eval_rgb = self.eval_rgb.permute(0, 3, 2, 1)
            self.eval_rgb = torch.reshape(self.eval_rgb, (n, w * h, c))
            self.eval_dirs = [LpDataset.all_dir[i] for i in LpDataset.train_idx]
            self.eval_dirs = torch.stack(self.eval_dirs)
            assert self.eval_dirs.size() == (LpDataset.num_train_img, 3)
            assert self.eval_rgb.size() == (LpDataset.num_train_img, self.img_wh[0] * self.img_wh[1], 3)
        elif self.split == 'test':
            self.test_rgb = torch.stack([LpDataset.all_rgb[i] for i in LpDataset.test_idx])
            n, c, h, w = self.test_rgb.size()
            self.test_rgb = self.test_rgb.permute(0, 3, 2, 1)  # leave in c h w for memory layout??
            self.test_rgb = torch.reshape(self.test_rgb, (n, w * h, c))
            self.test_dirs = [LpDataset.all_dir[i] for i in LpDataset.test_idx]
            self.test_dirs = torch.stack(self.test_dirs)
            assert self.test_dirs.size() == (LpDataset.num_test_img, 3)
            assert self.test_rgb.size() == (LpDataset.num_test_img, self.img_wh[0] * self.img_wh[1], 3)

    def __getitem__(self, idx):
        if self.split == 'train':
            rnd_l_idx = torch.randint(0, LpDataset.num_train_img, (1,))
            gt = self.train_rgb[idx, (rnd_l_idx*3):(rnd_l_idx*3+3)],
            return self.train_dirs, self.train_rgb[idx], gt
        elif self.split == 'eval':
            idx = torch.randint(0, LpDataset.num_train_img, (1,))
            return self.eval_dirs[idx], self.eval_rgb, idx
        elif self.split == 'test':
            return self.test_dirs[idx], self.test_rgb, idx

    def __len__(self):
        if self.split == 'train':
            return LpDataset.img_wh[0] * LpDataset.img_wh[1]
        elif self.split == 'eval':
            return 1
        elif self.split == 'test':
            return LpDataset.num_test_img

    # def define_transforms(self):
    #     # Not sure why this is not in __init__
    #     self.transform = tv.ToTensor()
    #

class LpDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 1,
                 ratio: tuple = (1, 1)
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.ratio = ratio
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(split='train', ratio=self.ratio)
        self.data_val = LpDataset(split='eval', ratio=self.ratio)
        self.data_test = LpDataset(split='test', ratio=self.ratio)

    def train_dataloader(self):
        return DataLoader(self.data_train, pin_memory=True, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, pin_memory=True, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=1)


class NeuralRtiModule(pl.LightningModule):
    def __init__(self, n_lights):
        super().__init__()
        self.save_hyperparameters()
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

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding


    def training_step(self, batch, batch_idx):
        dir_batch, rgb_batch, gt_batch  = batch
        for l_rgb, l_dirs in zip(rgb_batch, dir_batch):
            rgb_pred = []
            l_rgb = torch.squeeze(l_rgb)
            embedding = self(l_rgb)
            for l_dir in l_dirs:
                l_dir = torch.squeeze(l_dir)[:2]
                latent_code = torch.cat([embedding, l_dir], -1)
                decoded = self.decoder(latent_code)
                rgb_pred += [decoded]
            rgb_pred = torch.stack(rgb_pred)
            gt = torch.reshape(l_rgb, rgb_pred.size())
            loss = F.mse_loss(rgb_pred, gt)
            self.log('loss', loss, on_step=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # validation is run over whole Image Volume
        dir_batch, rgb_batch, light_idx = batch
        for iv, l_dir, idx in zip(rgb_batch, dir_batch, light_idx):
            iv = torch.squeeze(iv)
            gt_slice = iv[idx]
            gt = torch.reshape(gt_slice, (LpDataset.img_wh[0], LpDataset.img_wh[1], 3))
            gt = gt.permute(1, 0, 2).cpu().detach()
            iv = iv.permute(1,0,2)
            iv = torch.reshape(iv, (LpDataset.img_wh[1] * LpDataset.img_wh[0], LpDataset.num_train_img * 3))
            l_dir = torch.squeeze(l_dir)[:2]
            rgb_out = []
            for rgb in iv:
                embedding = self(rgb)
                embedding = torch.cat([embedding, l_dir], -1)
                decoded = self.decoder(embedding)
                rgb_out += [decoded]
            img = torch.stack(rgb_out)
            img = torch.reshape(img, (LpDataset.img_wh[0], LpDataset.img_wh[1], 3))
            img = img.permute(1, 0, 2).cpu().detach()

            loss = F.mse_loss(img, gt)
            self.log('valid_loss', loss, on_step=True, prog_bar=True, logger=True)
            img = (img*255).numpy().astype(np.uint8)
            gt = (gt*255).numpy().astype(np.uint8)
            utils.plot_image_grid([img, gt], ncols=1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


def train(args):
    # net_module = NeuralRtiModule()
    print(args)

    dm = LpDataModule(
        batch_size = args.batch_size,
        ratio = args.test_ratio
    )

    system = NeuralRtiModule(dm.data_train.num_train_img)

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
        auto_scale_batch_size=True,
        # auto_lr_find=True,
        # checkpoint_callback=checkpoint_callback,
        # resume_from_checkpoint=hparams.ckpt_path,
                      # logger=logger,
                      gpus=args.num_gpus,
                      num_sanity_val_steps=2,
                      benchmark=True,
                      profiler="simple",
    )

    trainer.fit(system, dm)

    # cli = MyLightningCLI(NeuralRtiModule, LpDataModule, seed_everything_default=1234)
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)


# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
#         parser.add_argument('--lp_name', type=str, default='dirs.lp')
#         parser.add_argument('--data.batch_size', type=int, default=1)

if __name__ == '__main__':
    print(sys.argv)
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
    parser.add_argument('--lp_name', type=str, default='dirs.lp')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--test_ratio', type=tuple, default=(8, 2))
    # parser.add_argument('--check_val_every_n_epoch', type=int, default= 1)

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    # dl = LpDataset()
    # print(dl[0])
    train(args)
