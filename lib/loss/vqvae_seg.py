import torch
import torch.nn as nn
import torchvision

class L1_Loss(torch.nn.Module):
    def __init__(self, weight=1.):
        super(L1_Loss, self).__init__()
        self.weight = weight
        self.loss_fn = nn.functional.l1_loss

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss = self.weight * loss
        return loss

class ClDiceLoss(nn.Module):
    def __init__(self, weight=1.) -> None:
        super().__init__()
        self.weight = weight
    
    def soft_skeletonize(self, x, thresh_width=5):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            p1 = torch.nn.functional.max_pool3d(x * -1, (3, 1, 1), 1, (1, 0, 0)) * -1
            p2 = torch.nn.functional.max_pool3d(x * -1, (1, 3, 1), 1, (0, 1, 0)) * -1
            p3 = torch.nn.functional.max_pool3d(x * -1, (1, 1, 3), 1, (0, 0, 1)) * -1
            min_pool_x = torch.min(torch.min(p1, p2), p3)
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def positive_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)

        intersection = (clf * vf).sum(-1)
        return (intersection.sum(0) + 1e-12) / (clf.sum(-1).sum(0) + 1e-12)

    def soft_cldice(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice acc
        '''
        target_skeleton = self.soft_skeletonize(target)
        cl_pred = self.soft_skeletonize(pred)
        clrecall = self.positive_intersection(target_skeleton, pred)  # ClRecall
        recall = self.positive_intersection(target, pred)
        clacc = self.positive_intersection(cl_pred, target)
        acc = self.positive_intersection(pred, target)
        return clrecall[0], clacc[0], recall[0], acc[0]
    
    def forward(self, pred, target):
        clrecall, clacc, recall, acc = self.soft_cldice(pred, target)
        cldice = (2. * clrecall * clacc) / (clrecall + clacc)
        loss = 1 - cldice
        loss = self.weight * loss
        return loss

class VQVAE_Loss(nn.Module):
    def __init__(self, aniso_dim=-2):
        super().__init__()
        self.aniso_dim = aniso_dim

        self.L2_Loss = nn.MSELoss()
        self.clDice_Lo = ClDiceLoss()
        
        self.loss_logger = dict()
        self.mip_logger = dict()

    def get_mip(self, img:torch.Tensor, aniso_dim):
        dim_list = [-1, -2, -3]
        dim_list.remove(aniso_dim)
        aniso_mip = torch.max(img, dim=aniso_dim).values
        iso_mip1 = torch.max(img, dim=dim_list[0]).values
        iso_mip2 = torch.max(img, dim=dim_list[1]).values
        return aniso_mip, iso_mip1, iso_mip2
    
    def forward(self, z, gt_mask, model_out):
        z_e, z_q, z_rec, z_mask = model_out
        aniso_mip_gt, halfIso_mip1_gt, halfIso_mip2_gt = self.get_mip(z, self.aniso_dim)
        aniso_mip_rec, halfIso_mip1_rec, halfIso_mip2_rec = self.get_mip(z_rec, self.aniso_dim)
        aniso_mip_mask, halfIso_mip1_mask, halfIso_mip2_mask = self.get_mip(z_mask, self.aniso_dim)

        # mip logger
        B, C, H, W = aniso_mip_gt.shape
        logger_aniso_mip = torch.stack((aniso_mip_gt, aniso_mip_rec, aniso_mip_mask), dim=1).view(-1, C, H, W)
        logger_halfIso_mip1 = torch.stack((halfIso_mip1_gt, halfIso_mip1_rec, halfIso_mip1_mask), dim=1).view(-1, C, H, W)
        logger_halfIso_mip2 = torch.stack((halfIso_mip2_gt, halfIso_mip2_rec, halfIso_mip2_mask), dim=1).view(-1, C, H, W)
        self.nrow = 3
        self.mip_logger['aniso_mip'] = torchvision.utils.make_grid(logger_aniso_mip, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip1'] = torchvision.utils.make_grid(logger_halfIso_mip1, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)
        self.mip_logger['halfIso_mip2'] = torchvision.utils.make_grid(logger_halfIso_mip2, nrow=self.nrow, normalize=True, scale_each=True, padding=2, pad_value=1)

        # loss logger
        loss_vqvae_embed = self.L2_Loss(z_e.detach(), z_q) + \
                           self.L2_Loss(z_e, z_q.detach())
        self.loss_logger['loss/loss_vqvae_embed'] = loss_vqvae_embed.item()

        loss_vqvae_rec = self.L2_Loss(z, z_rec)
        self.loss_logger['loss/loss_vqvae_rec'] = loss_vqvae_rec.item()

        loss_vqvae_mask = self.clDice_Lo(z_mask, gt_mask)
        self.loss_logger['loss/loss_vqvae_mask'] = loss_vqvae_mask.item()

        loss = loss_vqvae_embed + loss_vqvae_rec + loss_vqvae_mask
        self.loss_logger['loss/loss_total'] = loss.item()

        return loss, self.loss_logger, self.mip_logger

def get_loss(args, model=None):
    return VQVAE_Loss(args.aniso_dim)

if __name__ == '__main__':
    import argparse
    import sys, os
    sys.path.append(os.getcwd())
    from lib.arch.vqvae_seg import get_model
    args = argparse.Namespace
    args.in_channels = 1
    args.out_channels = 1
    args.features = [32,64,128]
    args.embedding_num = 64
    args.embedding_dim = 128
    args.norm_type = None

    args.aniso_dim = -2
    args.data_norm_type = 'min_max'
    args.load_size = 64

    device = torch.device('cuda:0')
    model = get_model(args)
    model.to(device)

    B = 1
    size = 128
    img = torch.rand(B,1,size,size,size, dtype=torch.float32).to(device)
    mask = torch.rand(B,1,size,size,size, dtype=torch.float32).to(device)
    model_out = model(img)

    loss_fn = get_loss(args, model)
    loss, loss_logger, mip_logger = loss_fn(img, mask, model_out)
    print(loss.item(), loss_logger.keys(), mip_logger.keys())
