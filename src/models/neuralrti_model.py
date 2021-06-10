from abc import ABC

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from src.utils import my_utils, metrics

from src.datamodules.datasets.lp_rti_dataset import LpDataset
from src.models.modules.rti_autoencoder import NeuralRtiEncoder, NeuralRtiDecoder


class NeuralRtiModule(LightningModule, ABC):
    def __init__(self, n_lights, **kwargs):
        super().__init__()
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
        dirs_batch = dirs_batch[..., :2].squeeze()
        batch_size = ray_batch.size()[0]
        num_lights = ray_batch.size()[1]
        ray_batch = ray_batch.view((batch_size, 3 * num_lights))
        ray_batch = ray_batch.squeeze()
        gt_batch = gt_batch.squeeze()

        rgb_pred = self.forward(ray_batch, dirs_batch)

        loss = torch.nn.functional.mse_loss(rgb_pred, gt_batch, reduction=self.hparams.reduction)

        self.log('batch_loss', loss)
        self.log("global_step", self.trainer.global_step)
        self.log("lr", my_utils.get_learning_rate(self.trainer.optimizers[0]))
        return {"loss": loss}

    # def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:



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

            loss = torch.nn.functional.mse_loss(pred_img, gt_img, reduction='mean')
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

        return {
            "val_loss": torch.mean(torch.stack(val_loss)),
            "val_psnr": torch.mean(torch.stack(val_psnr)),
        }


    def configure_optimizers(self):
        params = [self.encoder, self.decoder]
        optimizer = my_utils.get_optimizer(self.hparams, params, lr=self.hparams.lr)
        scheduler = my_utils.get_scheduler(self.hparams, optimizer)

        # optimizer_dec = utils.get_optimizer(self.hparams, [self.decoder], lr=self.hparams.lr_dec)
        # scheduler_dec = utils.get_scheduler(self.hparams, optimizer_dec)

        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.hparams.monitor,
                    'interval': self.hparams.interval,
                    'frequency': self.hparams.sch_freq,
                }
            },
        )