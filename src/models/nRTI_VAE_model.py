from abc import ABC

import numpy as np
import torch
from torch import nn
import wandb
from pytorch_lightning import LightningModule
from src.utils import my_utils, metrics

from src.datamodules.datasets.lp_rti_dataset import LpDataset
from src.models.modules.rti_autoencoder import NeuralRtiEncoderVae, NeuralRtiDecoderVae


class NrtiVaeModule(LightningModule, ABC):
    def __init__(self, n_lights, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.n_lights = n_lights
        n_coeff = self.hparams.n_coeff
        self.kl_coeff = self.hparams.kl_coeff

        self.encoder = NeuralRtiEncoderVae(n_coeff, n_lights)
        self.decoder = NeuralRtiDecoderVae(n_coeff)

        self.fc_mu  = nn.Linear(n_coeff, n_coeff)
        self.fc_var = nn.Linear(n_coeff, n_coeff)
        # self.automatic_optimization = False

    def forward(self, ray, dir):
        rgb = self.decoder(self.encoder(ray), dir)
        return rgb

    def _run_step(self, ray_batch, dirs_batch):
        x = self.encoder(ray_batch)
        # x = torch.cat([x, dirs_batch], dim=-1)

        mu = self.fc_mu(x.view(-1, 2, 9)[:, 0])
        log_var = self.fc_var(x.view(-1, 2, 9)[:, 1])
        p, q, z = self.sample(mu, log_var)
        embedding = torch.cat([z, dirs_batch], dim=-1)

        return z, self.decoder(embedding), p, q

    def sample(self, mu, log_var):
        # std = torch.exp(log_var / 2)
        std = torch.exp(log_var)

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        dirs_batch, ray_batch, gt_batch = batch
        dirs_batch = dirs_batch[..., :2].squeeze()
        batch_size = ray_batch.size()[0]
        num_lights = ray_batch.size()[1]
        ray_batch = ray_batch.view((batch_size, 3 * num_lights))
        ray_batch = ray_batch.squeeze()
        gt_batch = gt_batch.squeeze()

        z, rgb_pred, p, q = self._run_step(ray_batch, dirs_batch)

        recon_loss = torch.nn.functional.mse_loss(rgb_pred, gt_batch, reduction=self.hparams.reduction)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
            "lr": my_utils.get_learning_rate(self.trainer.optimizers[0])
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss


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
            z_col = []
            for ray in imgV:
                ray = ray.unsqueeze(0)

                x = self.encoder(ray)

                # x = torch.cat([x, l_dir], dim=-1)


                # mu = self.fc_mu(x)
                # log_var = self.fc_var(x)
                mu = self.fc_mu(x.view(-1, 2, 9)[:, 0])
                log_var = self.fc_var(x.view(-1, 2, 9)[:, 1])
                p, q, z = self.sample(mu, log_var)

                z = torch.cat([z, l_dir], dim=-1)

                rgb = self.decoder(z)  # shouldn't squeeze, but keep batch dim of IV
                rgb_out += [rgb]
                # z_col += [z]
                # wandb.log({"eval_embeddings": wandb.Histogram(z.cpu().detach())})

            # z_col = torch.stack(z_col)

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