import os
import gc
import os.path
import sys
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

import src.utils.metrics as metrics
import src.utils.my_utils as utils
import wandb


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

    def __init__(self, args, split="train", ratio=(1, 1)):
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.split = split
        assert split in ['train', 'eval', 'test']
        if LpDataset.reload:
            ration_sum = sum(ratio)
            LpDataset.ratio = int(ratio[0] / ratio[1])
            LpDataset.all_dir, LpDataset.all_rgb, LpDataset.img_wh = utils.load_lp_file(
                self.data_dir,
                img_wh=args.val_size
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
            self.train_imgVol = [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            self.train_imgVol = torch.stack(self.train_imgVol)
            self.t_n, self.t_h, self.t_w, self.t_c = self.train_imgVol.size()
            s = self.train_imgVol.stride()
            self.train_imgVol = self.train_imgVol.permute(1, 2, 0, 3).contiguous()
            s2 = self.train_imgVol.stride()

            self.train_imgVol = self.train_imgVol.view(self.t_h * self.t_w,
                                                        self.t_n * self.t_c)
            self.train_gt = self.train_imgVol.view(self.t_h * self.t_w *
                                                        self.t_n, self.t_c)
            s3 = self.train_imgVol.stride()

            assert self.train_imgVol.size() == (self.t_w * self.t_h, self.t_n * self.t_c)

        elif self.split == "eval":
            self.eval_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            )
            n, h, w, c = self.eval_rgb.size()

            self.eval_rgb = self.eval_rgb.view(n, h * w, c).contiguous()

            assert self.trn_val_dirs.size() == (LpDataset.num_train_img, 3)
            assert self.eval_rgb.size() == (
                LpDataset.num_train_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )

        elif self.split == "test":
            self.test_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.test_idx]
            )
            n, h, w, c = self.test_rgb.size()
            # self.test_rgb = self.test_rgb.permute(0, 3, 2, 1)
            self.test_rgb = torch.reshape(self.test_rgb, (n, w * h, c))

            self.test_dirs = [LpDataset.all_dir[i] for i in LpDataset.test_idx]
            self.test_dirs = torch.stack(self.test_dirs)

            # picked by light direction
            assert self.test_dirs.size() == (LpDataset.num_test_img, 3)
            assert self.test_rgb.size() == (
                LpDataset.num_test_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )

    def __getitem__(self, idx):
        if self.split == "train":
            t = idx % (self.t_n)
            p = idx % (self.t_w * self.t_h)

            l_dir = self.trn_val_dirs[t, :2]
            ray = self.train_imgVol[p, :]
            gt = self.train_gt[idx]

            return l_dir, ray, gt

        elif self.split == "eval":
            l_idx = torch.randint(0, len(LpDataset.trn_val_dirs), (1,))
            return self.trn_val_dirs[l_idx], self.eval_rgb, l_idx

        elif self.split == "test":
            idx = torch.randint(0, LpDataset.num_test_img, (1,))
            return self.test_dirs[idx], self.test_rgb, idx

    def __len__(self):
        if self.split == "train":
            return LpDataset.img_wh[0] * LpDataset.img_wh[1] * self.num_train_img
        elif self.split == "eval":
            return 1
        elif self.split == "test":
            return 1


class LpDataModule(pl.LightningDataModule):
    def __init__(self, args,  batch_size: int = 1, ratio: tuple = (1, 1)):
        super().__init__()
        self.args = args

        self.batch_size = batch_size
        self.ratio = ratio
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(self.args, split="train", ratio=self.ratio)
        self.data_val = LpDataset(self.args, split="eval", ratio=self.ratio)
        self.data_test = LpDataset(self.args, split="test", ratio=self.ratio)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            num_workers=args.num_workers,
            # pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          # pin_memory=True,
                          num_workers=args.num_workers,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=1)


class NeuralRtiEncoder(torch.nn.Module):
    def __init__(self, n_coeff, n_lights):
        super().__init__()
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            nn.Linear(n_units, n_units),
            # nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            # nn.Dropout(),
            nn.SELU(),
            nn.Linear(n_units, n_units),
            # nn.Dropout(),
            # nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            nn.SELU(),
            nn.Linear(n_units, n_coeff),
            # nn.BatchNorm1d(n_coeff, momentum=0.01, eps=0.001),
            nn.SELU(),
            # nn.BatchNorm1d(n_coeff, momentum=0.01, eps=0.001),
        )

    def forward(self,  ray):
        embedding = self.encoder(ray)
        return embedding


class NeuralRtiDecoder(torch.nn.Module):
    def __init__(self, n_coeff):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, 3),
        )

    def forward(self, embedding, dir ):
        latent_code = torch.cat([embedding, dir], dim=-1)
        rgb = self.decoder(latent_code)
        return rgb


class NeuralRtiModule(pl.LightningModule):
    def __init__(self, args, n_lights):
        super().__init__()
        self.hparams.update(args.__dict__)
        self.save_hyperparameters()

        self.n_lights = n_lights
        n_coeff = self.hparams.n_coeff

        self.encoder = NeuralRtiEncoder(n_coeff, n_lights)
        self.decoder = NeuralRtiDecoder(n_coeff)

        # self.automatic_optimization = False

    def forward(self, ray, dir):
        rgb = self.decoder(self.encoder(ray), dir)
        return rgb

    def training_step(self, batch, batch_idx):
        # opt_enc, opt_dec = self.optimizers()
        dirs_batch, ray_batch, gt_batch = batch
        rgb_pred = self(ray_batch, dirs_batch)

        loss = F.mse_loss(rgb_pred, gt_batch, reduction=self.hparams.reduction)
        # loss = F.l1_loss(rgb_pred, ray_batch, reduction=self.hparams.reduction)
        # loss = loss / batch_size

        # if optimizer_idx == 0:
        #     embedding = self.encoder(ray_batch)
        #     # with torch.no_grad():
        #
        #     rgb_pred = self.decoder(embedding, dirs_batch)
        #     loss = F.mse_loss(rgb_pred, gt_batch, reduction="sum")
        #     # sch_enc.step(loss)
        #     #
        #     # opt_enc.step()
        #     # opt_dec.step()
        #     # opt_dec.zero_grad()
        #     # opt_enc.zero_grad()
        #     # self.manual_backward(loss)
        #
        #     # self.manual_backward(loss, opt_enc)
        #     # self.manual_backward(loss, opt_dec
        #
        # if optimizer_idx == 1:
        #     with torch.no_grad():
        #         embedding = self.encoder(ray_batch)
        #     rgb_pred = self.decoder(embedding, dirs_batch)
        #     loss = F.mse_loss(rgb_pred, gt_batch, reduction="sum")
        #     # sch_dec.step(loss)
        #
        #     # opt_dec.step()
        #     # opt_dec.zero_grad()
        #     # self.manual_backward(loss)
        #
        #     # self.manual_backward(loss, opt_enc)
        #     # self.manual_backward(loss, opt_dec)

        self.log("global_step", self.global_step)
        self.log('train_loss', loss)
        self.log("lr", utils.get_learning_rate(self.trainer.optimizers[0]), on_step=True, prog_bar=True, logger=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        val_loss = []
        val_psnr = []
        dir_batch, rgb_batch, idx = batch
        dir_batch = dir_batch[..., :2]
        for imgV, l_dir in zip(rgb_batch, dir_batch):
            gt_img = imgV[idx]
            gt_img = torch.squeeze(gt_img)  # w*h, 3
            gt_img = gt_img.view((LpDataset.img_wh[1], LpDataset.img_wh[0], 3))
            gt_img = gt_img.cpu().detach()

            imgV = imgV.permute(1, 0, 2)  # .contiguous()
            imgV = torch.reshape(
                imgV,
                (LpDataset.img_wh[1] * LpDataset.img_wh[0],
                 LpDataset.num_train_img * 3
                 )).contiguous()

            rgb_out = []
            for ray in imgV:
                ray = ray.unsqueeze(0)
                rgb = self.forward(ray, l_dir)  # shouldn't squeeze, but keep batch dim of IV
                rgb_out += [rgb]

            pred_img = torch.stack(rgb_out)
            pred_img = pred_img.view(LpDataset.img_wh[1], LpDataset.img_wh[0], 3)
            pred_img = pred_img.cpu().detach()

            loss = F.mse_loss(pred_img, gt_img, reduction='mean')
            # loss = loss / hparams.batch_size
            val_loss += [loss.detach()]
            psnr = metrics.psnr(pred_img, gt_img)
            val_psnr += [psnr.detach()]

            # pred_img = colour.YCbCr_to_RGB(pred_img)
            # gt_img = colour.YCbCr_to_RGB(gt_img)

            pred_img = np.clip(pred_img, 0.0, 1.0)

            # img_np = (pred_img * 255).numpy().astype(np.uint8)
            # gt_np = (gt_img * 255).numpy().astype(np.uint8)
            # utils.plot_image_grid([img_np, gt_np], ncols=1)

            log_imgs = torch.cat([pred_img, gt_img], dim=0)
            log_imgs = log_imgs.permute(2, 0, 1).contiguous()
            caption = "Top: Prediction, Bottom: Ground Truth"

            self.log("global_step", self.global_step)
            self.log("val_loss", loss)
            self.log("val_psnr", psnr)
            self.log("pred/gt", [wandb.Image(log_imgs, caption=caption)])
        gc.collect()
        return {
            "val_loss": torch.mean(torch.stack(val_loss)),
            "val_psnr": torch.mean(torch.stack(val_psnr)),
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, on_step=True)


    def configure_optimizers(self):
        params = [self.encoder, self.decoder]
        optimizer = utils.get_optimizer(self.hparams, params, lr=self.hparams.lr_enc)
        scheduler = utils.get_scheduler(self.hparams, optimizer)

        optimizer_dec = utils.get_optimizer(self.hparams, [self.decoder], lr=self.hparams.lr_dec)
        scheduler_dec = utils.get_scheduler(self.hparams, optimizer_dec)

        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                    mode='min',
        #                                                    factor=0.2,
        #                                                    patience=500,
        #                                                    verbose=True)
        #
        # optimizer_dec = utils.get_optimizer(self.hparams, [self.decoder])
        # scheduler_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dec,
        #                                                        mode='min',
        #                                                        factor=0.9,
        #                                                        patience=500,
        #                                                        verbose=True)

        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'mode': 'min',
                    'interval': 'epoch',
                    'frequency': self.hparams.sch_freq,
                }
            },
            # {
            #     'optimizer': optimizer_dec,
            #     'lr_scheduler': {
            #         'scheduler': scheduler_dec,
            #         'monitor': 'val_loss',
            #         'interval': 'epoch',
            #         'frequency': self.hparams.sch_freq,
            #     }
            # }
        )
        # return optimizer


def train(args):
    # net_module = NeuralRtiModule()
    root_dir = args.root_dir
    log_dir = os.path.join(root_dir, "./logs")
    ckpts_dir = os.path.join(root_dir, "./ckpts")

    wandb_logger = WandbLogger()
    wandb.login(key=args.key)
    wandb.init(config=args)

    # tb_logger = TensorBoardLogger(save_dir=log_dir)

    dm = LpDataModule(args, batch_size=args.batch_size, ratio=args.test_ratio)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpts_dir,
        filename=os.path.join(
            f"{args.root_dir}", f"{args.exp_name}", "ckpts", "nRTI_{epoch:02d}"
        ),
        save_top_k=5,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    rti_model = NeuralRtiModule(args, dm.data_train.num_train_img)
    wandb.watch(rti_model, log_freq=1500, log="all")

    trainer = pl.Trainer(
        default_root_dir=args.root_dir,
        resume_from_checkpoint=args.ckpt_path,
        callbacks=[lr_monitor],
        max_epochs=args.num_epochs,
        # auto_scale_batch_size=True,
        val_check_interval=1.0,  # 1/float times per epoch or every int steps
        # check_val_every_n_epoch=1,
        logger=[wandb_logger],
        log_every_n_steps=1,
        gpus=args.num_gpus,
        num_sanity_val_steps=2,
        # auto_lr_find=True,
        # benchmark=True,
        # profiler="simple",
    )
    # trainer.tune(rti_model, dm)
    trainer.fit(rti_model, dm)
    wandb.finish()
    return
    # lr_finder = trainer.tuner.lr_find(rti_model,
    #                                   num_training=2000,
    #                                   min_lr=0.0000003,
    #                                   max_lr=0.000001)
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # # fig.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # fig.show()


# print(lr_finder.results[lr_finder._optimal_idx])


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "dryrun"

    print(sys.argv)
    parser = ArgumentParser()
    try:
        if os.environ["WANDB_MODE"] == "dryrun":
            parser.add_argument("--num_workers", type=int, default=0)
    except:
        parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--data_dir", type=str, default="./data")
    # parser.add_argument("--data_dir", type=str, default="/home/mag/RTI/exampledataset")
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--exp_name", type=str, default="loewenkopf_elu")
    # parser.add_argument("--exp_name", type=str, default="example")
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--lp_name", type=str, default="dirs.lp")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_size", type=tuple, default=None)
    # parser.add_argument("--val_size", type=tuple, default=(128, 128))


    parser.add_argument("--test_ratio", type=tuple, default=(10, 2))
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="radam")
    # parser.add_argument("--lr", type=float, default=0.000035)
    # parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--lr_enc", type=float, default=1e-4)
    parser.add_argument("--lr_dec", type=float, default=1e-4)

    parser.add_argument("--weight_decay", type=float, default=0.995)  # optimizer
    parser.add_argument("--decay_gamma", type=float, default=0.995)
    parser.add_argument("--decay_step", type=float, default=1)

    parser.add_argument("--lr_scheduler", type=str, default="plateau")
    parser.add_argument("--patience", type=float, default=3)
    parser.add_argument("--factor", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--sch_freq", type=float, default=1)

    parser.add_argument("--n_coeff", type=int, default=9)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    parser.add_argument("--reduction", type=str, default='mean')

    # parser.add_argument("--img_w", type=int, default=256)
    # parser.add_argument("--img_h", type=int, default=171)

    args = parser.parse_args()
    train(args)
