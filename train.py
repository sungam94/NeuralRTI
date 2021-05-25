import os.path
import sys, os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms
import torch.autograd.profiler as profiler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
import metrics
import utils
import matplotlib.pyplot as plt

# os.environ["WANDB_MODE"] = "dryrun"

class LpDataset(Dataset):
    reload = 1
    train_idx = None
    test_idx = None
    img_wh = None
    num_train_img = None
    num_test_img = None
    trn_val_dirs = None
    n_l_dirs = None
    h, w, n, c = None, None, None, None

    def __init__(self, split="train", ratio=(1, 1)):
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.split = split

        if LpDataset.reload:
            ration_sum = sum(ratio)
            LpDataset.ratio = int(ratio[0] / ratio[1])
            # load all images from disk
            LpDataset.all_dir, LpDataset.all_rgb, LpDataset.img_wh = utils.load_lp_file(
                self.data_dir
            )
            LpDataset.n_l_dirs = len(LpDataset.all_dir)
            nl = LpDataset.n_l_dirs
            part_size = nl // ration_sum
            LpDataset.num_test_img = part_size * ratio[1]
            LpDataset.test_idx = np.random.choice(
                nl, LpDataset.num_test_img, replace=False
            )
            LpDataset.test_idx = np.sort(LpDataset.test_idx)
            LpDataset.train_idx = list(set(range(nl)) - set(LpDataset.test_idx))
            LpDataset.num_train_img = len(LpDataset.train_idx)
            LpDataset.trn_val_dirs = torch.stack(
                [LpDataset.all_dir[i] for i in LpDataset.train_idx]
            )
            LpDataset.reload = 0

        if self.split == "train":
            self.train_rgb = [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            self.train_rgb = torch.stack(self.train_rgb)
            self.t_n, self.t_c, self.t_h, self.t_w = self.train_rgb.size()
            self.train_rgb = self.train_rgb.permute(0, 3, 2, 1)
            self.train_rgb = self.train_rgb.contiguous()
            self.train_rgb = torch.reshape(
                self.train_rgb, (self.t_n, self.t_h * self.t_w, self.t_c)
            )
            assert self.train_rgb.size() == (self.t_n,  self.t_h * self.t_w, self.t_c)
        elif self.split == "eval":
            self.eval_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            )
            n, c, h, w = self.eval_rgb.size()
            self.eval_rgb = self.eval_rgb.permute(0, 3, 2, 1)
            self.eval_rgb = torch.reshape(self.eval_rgb, (n, w * h, c))
            assert self.eval_rgb.size() == (
                LpDataset.num_train_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )
        elif self.split == "test":
            self.test_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.test_idx]
            )
            n, c, h, w = self.test_rgb.size()
            self.test_rgb = self.test_rgb.permute(
                0, 3, 2, 1
            )  # leave in c h w for memory layout??
            self.test_rgb = torch.reshape(self.test_rgb, (n, w * h, c))
            self.test_dirs = [LpDataset.all_dir[i] for i in LpDataset.test_idx]
            self.test_dirs = torch.stack(self.test_dirs)
            assert self.test_dirs.size() == (LpDataset.num_test_img, 3)
            assert self.test_rgb.size() == (
                LpDataset.num_test_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )

    def __getitem__(self, idx):
        if self.split == "train":
            # p_idx = idx // self.t_n
            l_dir = self.trn_val_dirs
            iv = self.train_rgb[:, idx, :]
            return l_dir, iv

        elif self.split == "eval":
            idx = torch.randint(0, LpDataset.num_train_img, (1,))
            return self.trn_val_dirs[idx], self.eval_rgb, idx
        elif self.split == "test":
            return self.test_dirs[idx], self.test_rgb, idx

    def __len__(self):
        if self.split == "train":
            return LpDataset.img_wh[0] * LpDataset.img_wh[1]
        elif self.split == "eval":
            return 1
        elif self.split == "test":
            return LpDataset.num_test_img


class LpDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1, ratio: tuple = (1, 1)):
        super().__init__()
        self.batch_size = batch_size
        self.ratio = ratio
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(split="train", ratio=self.ratio)
        self.data_val = LpDataset(split="eval", ratio=self.ratio)
        self.data_test = LpDataset(split="test", ratio=self.ratio)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.data_val, pin_memory=True, num_workers=0, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=1)


class NeuralRtiModule(pl.LightningModule):
    def __init__(self, args, n_lights):
        super().__init__()
        self.idx_col = []
        self.args = args
        self.save_hyperparameters()
        self.n_lights = n_lights
        n_coeff = 9
        # the first and last layers contain 3N units, N are number of input lights
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_coeff),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.ELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.ELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.ELU(),
            nn.Linear(n_coeff + 2, 3),
        )

    def forward(self, l_rgb, l_dirs):
        # in lightning, forward defines the prediction/inference actions
        preds = []

        for l_dir in l_dirs:
            embedding = self.encoder(l_rgb)
            l_dir = l_dir[:2]
            latent_code = torch.cat([embedding, l_dir], -1)
            rgb = self.decoder(latent_code)
            preds += [rgb]
        return torch.stack(preds)


    def training_step(self, batch, batch_idx):
        train_loss = []
        dir_batch, rgb_batch = batch
        for l_rgb, l_dirs in zip(rgb_batch, dir_batch):
            l_rgb = l_rgb.flatten()
            rgb_pred = self.forward(l_rgb, l_dirs)
            loss = F.mse_loss(rgb_pred.flatten(), l_rgb, reduction="mean")
            # loss = F.mse_loss(rgb_pred.flatten(), l_rgb, reduction="sum")
            train_loss += [loss]
            self.log("lr", utils.get_learning_rate(self.optimizer))
            self.log("global_step", self.global_step)
            self.log('loss', loss)
            self.log("train_loss", loss, on_step=True, prog_bar=True, logger=False)
        return {"loss": torch.stack(train_loss)}
        # return {"loss": torch.mean(torch.stack(train_loss))}

    # def training_step_end(self, outs):
    #     # log accuracy on each step_end, for compatibility with data-parallel
    #     self.idx_col.append(outs['indices'])
    #     if ((self.global_step % 500) == 0):
    #         idx = np.asarray(self.idx_col)
    #         self.idx_col = []
    #         l_data = idx[:,0]
    #         l_table = wandb.Table(data=l_data, columns=['l_idx'])
    #         wandb.log({'l_idx': wandb.plot.histogram(l_table, "l_idx",
    #               title=f"{self.n_lights} - light direction indices")})
    #         p_data = idx[:, 1]
    #         p_table = wandb.Table(data=p_data, columns=['p_idx'])
    #         wandb.log({'p_idx': wandb.plot.histogram(p_table, "p_idx",
    #                                                   title=f"pixel indices")})
    #         data = idx[:,2]
    #         table = wandb.Table(data=data, columns=['iv_idx'])
    #         wandb.log({'iv_idx': wandb.plot.histogram(table, "iv_idx",
    #               title=f"image volume indices")})



    def validation_step(self, batch, batch_idx):
        val_loss = []
        val_psnr = []
        dir_batch, rgb_batch, idx = batch
        for iv, l_dir in zip(rgb_batch, dir_batch):
            gt_slice = iv[idx]  # at light index
            gt_slice = torch.squeeze(gt_slice)
            gt_slice = torch.reshape(gt_slice, (64, 43, 3))
            gt_slice = gt_slice.permute(1, 0, 2).contiguous()
            gt_slice = torch.reshape(gt_slice, (43, 64, 3))
            gt_slice = gt_slice.cpu().detach()
            iv = iv.permute(1, 0, 2).contiguous()
            iv = torch.reshape(
                iv,
                (LpDataset.img_wh[1] * LpDataset.img_wh[0], LpDataset.num_train_img * 3)    )
            rgb_out = []

            for iv_rgb in iv:
                rgb = self.forward(iv_rgb, l_dir)
                rgb_out += [rgb]

            img = torch.stack(rgb_out)
            img = torch.reshape(img, (LpDataset.img_wh[0], LpDataset.img_wh[1], 3))
            img = img.permute(1, 0, 2).cpu().detach()

            psnr = metrics.psnr(img, gt_slice)
            loss = F.mse_loss(img, gt_slice, reduction='mean')
            val_loss += [loss]
            val_psnr += [psnr]

            img_np = (img * 255).numpy().astype(np.uint8)
            gt_np = (gt_slice * 255).numpy().astype(np.uint8)
            utils.plot_image_grid([img_np, gt_np], ncols=1)

            log_imgs = torch.cat([img, gt_slice], dim=0)
            log_imgs = log_imgs.permute(2, 0, 1).contiguous()
            caption = "Top: Prediction, Bottom: Ground Truth"
            self.log("global_step", self.global_step)
            self.log("val_loss", loss)
            self.log("val_psnr", psnr)
            self.log("pred/gt", [wandb.Image(log_imgs, caption=caption)])
        # return {
        #     "val_loss": torch.mean(torch.stack(val_loss)),
        #     "val_psnr": torch.mean(torch.stack(val_psnr)),
        # }

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, on_step=True)

    def configure_optimizers(self):
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.optimizer = utils.get_optimizer(self.args, [self])
        scheduler = utils.get_scheduler(self.args, self.optimizer)
        return [self.optimizer], [scheduler]


def train(args):
    # net_module = NeuralRtiModule()
    root_dir = args.root_dir
    log_dir = os.path.join(root_dir, "/logs")
    ckpts_dir = os.path.join(root_dir, "/ckpts")

    wandb_logger = WandbLogger()
    wandb.login(key=args.key)
    wandb.init(config=args)
    print(args)

    dm = LpDataModule(batch_size=args.batch_size, ratio=args.test_ratio)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpts_dir,
        filename=os.path.join(
            f"{args.root_dir}", f"{args.exp_name}", "ckpts", "nRTI_{epoch:02d}"
        ),
        save_top_k=5,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    rti_model = NeuralRtiModule(args, dm.data_train.num_train_img)
    wandb.watch(rti_model, log="all")
    trainer = pl.Trainer(
        default_root_dir=args.root_dir,
        resume_from_checkpoint=args.ckpt_path,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=args.num_epochs,
        # auto_scale_batch_size=True,
        # auto_lr_find=True,
        val_check_interval=200.0,
        # log_every_n_steps=10,
        logger=wandb_logger,
        gpus=args.num_gpus,
        num_sanity_val_steps=2,
        # benchmark=True,
        # profiler="simple",
    )
    trainer.fit(rti_model, dm)
    wandb.finish()

    # lr_finder = trainer.tuner.lr_find(rti_model)
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()


if __name__ == "__main__":
    print(sys.argv)
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/home/mag/RTI/loewenkopf")
    parser.add_argument("--root_dir", type=str, default="/home/mag/RTI/")
    parser.add_argument("--exp_name", type=str, default="loewenkopf_relu")
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--lp_name", type=str, default="dirs.lp")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--test_ratio", type=tuple, default=(10, 2))
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="radam")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.995)
    parser.add_argument("--decay_step", type=float, default=1)
    parser.add_argument("--decay_gamma", type=float, default=0.995)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    # parser.add_argument('--check_val_every_n_epoch', type=int, default= 1)

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    # dl = LpDataset()
    # print(dl[0])
    train(args)
