import torch
import torch.nn as nn
import torchvision

class L1_Loss(torch.nn.Module):
    def __init__(self, weight):
        super(L1_Loss, self).__init__()
        self.weight = weight
        self.loss_fn = nn.functional.l1_loss

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss = self.weight * loss
        return loss

class VQVAE_Loss(nn.Module):
    def __init__(self, aniso_dim=-2):
        super().__init__()
        self.aniso_dim = aniso_dim

        self.L2_Loss = nn.MSELoss()
        
        self.loss_logger = dict()
        self.mip_logger = dict()

    def get_mip(self, img:torch.Tensor, aniso_dim):
        dim_list = [-1, -2, -3]
        dim_list.remove(aniso_dim)
        aniso_mip = torch.max(img, dim=aniso_dim).values
        iso_mip1 = torch.max(img, dim=dim_list[0]).values
        iso_mip2 = torch.max(img, dim=dim_list[1]).values
        return aniso_mip, iso_mip1, iso_mip2
    
    def forward(self, z, model_out):
        z_e, z_q, z_rec = model_out
        aniso_mip_gt, halfIso_mip1_gt, halfIso_mip2_gt = self.get_mip(z, self.aniso_dim)
        aniso_mip_rec, halfIso_mip1_rec, halfIso_mip2_rec = self.get_mip(z_rec, self.aniso_dim)

        # mip logger
        B, C, H, W = aniso_mip_gt.shape
        logger_aniso_mip = torch.stack((aniso_mip_gt, aniso_mip_rec), dim=1).view(-1, C, H, W)
        logger_halfIso_mip1 = torch.stack((halfIso_mip1_gt, halfIso_mip1_rec), dim=1).view(-1, C, H, W)
        logger_halfIso_mip2 = torch.stack((halfIso_mip2_gt, halfIso_mip2_rec), dim=1).view(-1, C, H, W)
        self.nrow = 2
        self.mip_logger['aniso_mip'] = torchvision.utils.make_grid(logger_aniso_mip, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip1'] = torchvision.utils.make_grid(logger_halfIso_mip1, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip2'] = torchvision.utils.make_grid(logger_halfIso_mip2, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)

        # loss logger
        loss_vqvae_embed = self.L2_Loss(z_e.detach(), z_q) + \
                           self.L2_Loss(z_e, z_q.detach())
        self.loss_logger['loss/loss_vqvae_embed'] = loss_vqvae_embed.item()

        loss_vqvae_rec = self.L2_Loss(z, z_rec)
        self.loss_logger['loss/loss_vqvae_rec'] = loss_vqvae_rec.item()

        loss = loss_vqvae_embed + loss_vqvae_rec
        self.loss_logger['loss/loss_total'] = loss.item()

        return loss, self.loss_logger, self.mip_logger

def get_loss(args, model=None):
    return VQVAE_Loss(args.aniso_dim)
