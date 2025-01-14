import torch
import torch.nn as nn
import torchvision

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.getcwd())
from lib.utils.utils import get_slant_mip
from lib.arch.vqvae_sr import VQVAE_SR

class GAN_Loss(nn.Module):
    def __init__(self, weight, target_real_label=1.0, target_fake_label=0.0):
        super(GAN_Loss, self).__init__()
        self.weight = weight
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_fn = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(prediction.device)

    def __call__(self, prediction, target_is_real, is_disc):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss_fn(prediction, target_tensor)
        if not is_disc:
            loss = self.weight * loss
        return loss

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
    def __init__(self, model:VQVAE_SR,
                 aniso_dim=-2, iso_dim=-1, angel=-45, 
                 G_train_it=1, D_train_it=1, 
                 full_mip=True):
        super().__init__()
        self.model = model

        self.aniso_dim = aniso_dim
        self.iso_dim = iso_dim
        self.angel = angel
        self.G_train_it = G_train_it
        self.D_train_it = D_train_it
        self.full_mip = full_mip

        self.GAN_Loss = GAN_Loss(weight=1.0)
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
    
    def set_requires_grad(self, nets:list[torch.nn.Module], requires_grad=False):
        if not isinstance(nets, list):
                nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cal_GAN_loss(self, net:torch.nn.Module, input:torch.Tensor, target:bool, is_disc:bool):
        output = net(input)
        loss = self.GAN_Loss(output, target, is_disc=is_disc)
        return loss
    
    def cal_loss_G(self):
        # vqvae loss
        loss_vqvae_embed = self.L2_Loss(self.z_e.detach(), self.z_q) + \
                           self.L2_Loss(self.z_e, self.z_q.detach())
        self.loss_logger['loss_G/loss_vqvae_embed'] = loss_vqvae_embed.item()

        loss_vqvae_rec = self.L2_Loss(self.z, self.z_rec)
        self.loss_logger['loss_G/loss_vqvae_rec'] = loss_vqvae_rec.item()

        # cal MIP loss
        loss_aniso_mip = self.cal_GAN_loss(self.model.D_MIP, self.aniso_mip_sr, True, is_disc=False)
        loss_halfIso_mip1 = self.cal_GAN_loss(self.model.D_MIP, self.halfIso_mip1_sr, True, is_disc=False)
        loss_halfIso_mip2 = self.cal_GAN_loss(self.model.D_MIP, self.halfIso_mip2_sr, True, is_disc=False)
        self.loss_logger['loss_G/loss_aniso_mip'] = loss_aniso_mip.item()
        self.loss_logger['loss_G/loss_halfIso_mip1'] = loss_halfIso_mip1.item()
        self.loss_logger['loss_G/loss_halfIso_mip2'] = loss_halfIso_mip2.item()

        # cal Cube loss
        loss_rec = self.cal_GAN_loss(self.model.D_Rec, self.z_sr, True, is_disc=False)
        self.loss_logger['loss_G/loss_rec'] = loss_rec.item()

        loss_G = loss_vqvae_embed + loss_vqvae_rec + \
                 loss_aniso_mip + loss_halfIso_mip1 + loss_halfIso_mip2 + \
                 loss_rec
        self.loss_logger['loss/loss_G'] = loss_G.item()
        return loss_G

    def cal_loss_D(self):
        # real MIP
        loss_real_pred_by_D_aniso = self.cal_GAN_loss(self.model.D_MIP, self.ref_iso_mip, True, is_disc=True)
        loss_real_pred_by_D_iso1 = self.cal_GAN_loss(self.model.D_MIP, self.ref_iso_mip, True, is_disc=True)
        loss_real_pred_by_D_iso2 = self.cal_GAN_loss(self.model.D_MIP, self.ref_iso_mip, True, is_disc=True)
        self.loss_logger['loss_D/loss_real_pred_by_D_aniso'] = loss_real_pred_by_D_aniso.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_iso1'] = loss_real_pred_by_D_iso1.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_iso2'] = loss_real_pred_by_D_iso2.item()

        # real Cube
        loss_real_pred_by_D_rec = self.cal_GAN_loss(self.model.D_Rec, self.z, True, is_disc=True)
        self.loss_logger['loss_D/loss_real_pred_by_D_rec'] = loss_real_pred_by_D_rec.item()

        # fake MIP
        loss_fake_pred_by_D_aniso = self.cal_GAN_loss(self.model.D_MIP, self.aniso_mip_sr.detach(), False, is_disc=True)
        loss_fake_pred_by_D_iso1 = self.cal_GAN_loss(self.model.D_MIP, self.halfIso_mip1_sr.detach(), False, is_disc=True)
        loss_fake_pred_by_D_iso2 = self.cal_GAN_loss(self.model.D_MIP, self.halfIso_mip2_sr.detach(), False, is_disc=True)
        self.loss_logger['loss_D/loss_fake_pred_by_D_aniso'] = loss_fake_pred_by_D_aniso.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_iso1'] = loss_fake_pred_by_D_iso1.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_iso2'] = loss_fake_pred_by_D_iso2.item()

        # fake Cube
        loss_fake_pred_by_D_rec = self.cal_GAN_loss(self.model.D_Rec, self.z_rec.detach(), False, is_disc=True)
        self.loss_logger['loss_D/loss_fake_pred_by_D_rec'] = loss_fake_pred_by_D_rec.item()

        loss_D_real = loss_real_pred_by_D_aniso + loss_real_pred_by_D_iso1 + loss_real_pred_by_D_iso2 + \
                      loss_real_pred_by_D_rec
        loss_D_fake = loss_fake_pred_by_D_aniso + loss_fake_pred_by_D_iso1 + loss_fake_pred_by_D_iso2 + \
                      loss_fake_pred_by_D_rec
        loss_D = loss_D_real + loss_D_fake
        self.loss_logger['loss/loss_D'] = loss_D.item()
        return loss_D

    def forward(self, z, model_out, it):
        if self.full_mip:
            self.z, self.ref_iso_mip, self.z_e, self.z_q, self.z_rec, self.z_sr = model_out
        else:
            self.z = z
            self.z_e, self.z_q, self.z_rec, self.z_sr = model_out
            self.ref_iso_mip = get_slant_mip(self.z, angel=self.angel, iso_dim=self.iso_dim)

        aniso_mip_gt, halfIso_mip1_gt, halfIso_mip2_gt = self.get_mip(self.z, self.aniso_dim)
        aniso_mip_rec, halfIso_mip1_rec, halfIso_mip2_rec = self.get_mip(self.z_rec, self.aniso_dim)
        self.aniso_mip_sr, self.halfIso_mip1_sr, self.halfIso_mip2_sr = self.get_mip(self.z_sr, self.aniso_dim)

        # mip logger
        B, C, H, W = aniso_mip_gt.shape
        logger_aniso_mip = torch.stack((aniso_mip_gt, aniso_mip_rec, self.aniso_mip_sr), dim=1).view(-1, C, H, W)
        logger_halfIso_mip1 = torch.stack((halfIso_mip1_gt, halfIso_mip1_rec, self.halfIso_mip1_sr), dim=1).view(-1, C, H, W)
        logger_halfIso_mip2 = torch.stack((halfIso_mip2_gt, halfIso_mip2_rec, self.halfIso_mip2_sr), dim=1).view(-1, C, H, W)
        self.nrow = 3
        self.mip_logger['aniso_mip'] = torchvision.utils.make_grid(logger_aniso_mip, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip1'] = torchvision.utils.make_grid(logger_halfIso_mip1, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip2'] = torchvision.utils.make_grid(logger_halfIso_mip2, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)

        self.mip_logger['ref_iso_mip'] = torchvision.utils.make_grid(self.ref_iso_mip, nrow=self.nrow, normalize=True, scale_each=True)

        # === G forward ===
        if (it+1)%self.G_train_it == 0:
            self.set_requires_grad([self.model.D_MIP, self.model.D_Rec], False)
            loss_G = self.cal_loss_G()
        else:
            loss_G = torch.Tensor([0.0])

        # === D forward ===
        if (it+1)%self.D_train_it == 0:
            self.set_requires_grad([self.model.D_MIP, self.model.D_Rec], True)
            loss_D = self.cal_loss_D()
        else:
            loss_D = torch.Tensor([0.0])

        return loss_G, loss_D, dict(sorted(self.loss_logger.items())), self.mip_logger

def get_loss(args, model=None):
    loss = VQVAE_Loss(model=model.module, aniso_dim=args.aniso_dim, iso_dim=args.iso_dim, angel=args.angel,
                      G_train_it=args.G_train_it, D_train_it=args.D_train_it,
                      full_mip=args.full_mip)
    return loss

if __name__ == '__main__':
    import argparse
    from lib.arch.vqvae_sr import get_model
    args = argparse.Namespace
    args.in_channels = 1
    args.out_channels = 1
    args.features = [32,64,128]
    args.embedding_num = 64
    args.embedding_dim = 128

    args.aniso_dim = -2
    args.iso_dim = -1
    args.angel = -45
    args.norm_type = None
    args.data_norm_type = 'min_max'
    args.full_mip = True
    args.load_size = 64
    args.feed_size = 64
    
    args.G_train_it = 1
    args.D_train_it = 1

    device = torch.device('cuda:0')
    model = get_model(args)
    model.to(device)

    B = 1
    size = 128

    real_A = torch.rand(B,1,size,size,size, dtype=torch.float32).to(device)
    model_out = model(real_A)

    loss_fn = get_loss(args, model)
    loss_G, loss_D, loss_logger, mip_logger = loss_fn(real_A, model_out, 1)
    print(loss_G.item(), loss_D.item(), loss_logger.keys(), mip_logger.keys())