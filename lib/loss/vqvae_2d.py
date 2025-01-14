import torch
import torch.nn as nn
import torchvision

class VQVAE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L2_Loss = nn.MSELoss()
        
        self.loss_logger = dict()
        self.mip_logger = dict()
    
    def forward(self, z, model_out):
        z_e, z_q, z_rec = model_out

        # mip logger
        B, C, H, W = z_rec.shape
        logger_aniso_mip = torch.stack((z, z_rec), dim=1).view(-1, C, H, W)
        self.nrow = 2
        self.mip_logger['image'] = torchvision.utils.make_grid(logger_aniso_mip, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)

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
    return VQVAE_Loss()
