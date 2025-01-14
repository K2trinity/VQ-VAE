import torch
import torch.nn as nn
import random
import functools

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())

from lib.utils.utils import get_slant_mip, center_crop
from lib.dataset.tif_dataset import norm_fn

# === norm ====
def get_norm_layer(norm_type:str, dim=2):
    if dim == 2:
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    elif dim == 3:
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    else:
        raise Exception('Invalid dim.')

    if norm_type == 'batch':
        norm_layer = functools.partial(BatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm, affine=False, track_running_stats=False)
    elif norm_type is None or norm_type == 'identity':
        norm_layer = nn.Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# === residual conv ====
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type=None, dim=3):
        super(ResConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        conv_layers = []
        conv_layers.append(Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        conv_layers.append(norm_layer(out_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        conv_layers.append(norm_layer(out_channels))
        self.conv = nn.Sequential(*conv_layers)

        if in_channels != out_channels:
            self.conv1x1 = Conv(in_channels, out_channels, kernel_size=1, bias=use_bias)
        else:
            self.conv1x1 = None
        self.final_act = nn.ReLU()

    def forward(self, x):
        input = x
        x = self.conv(x)
        if self.conv1x1:
            input = self.conv1x1(input)
        x = x + input
        x = self.final_act(x)
        return x


class DownScale(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, factor:int=2, *, dim=3) -> None:
        super(DownScale, self).__init__()
        if dim == 2:
            self.down = nn.MaxPool2d(kernel_size=factor)
        elif dim == 3:
            self.down = nn.MaxPool3d(kernel_size=factor)
        else:
            raise Exception('Invalid dim.')
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class UpScale(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, factor=2, *, dim=3) -> None:
        super(UpScale, self).__init__()
        if dim == 2:
            Conv = nn.Conv2d
            upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        elif dim == 3:
            Conv = nn.Conv3d
            upsample = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        else:
            raise Exception('Invalid dim.')
        
        self.up = nn.Sequential(
            upsample,
            Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, features=[64, 128, 256], *, norm_type='batch', dim=3):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        
        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')

        self.pre_conv = nn.Sequential(
            Conv(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        in_ch = features[0]
        for feature in features:
            self.encoder.append(ResConv(in_ch, feature, norm_type=norm_type, dim=dim))
            self.encoder.append(DownScale(feature, feature, 2, dim=dim))
            in_ch = feature

        self.post_conv = Conv(in_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pre_conv(x)
        for i, down in enumerate(self.encoder):
             x = down(x)
        x = self.post_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, features=[256, 128, 64], *, norm_type='batch', final_act=None, dim=3):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
            
        # Decoder
        self.pre_conv = nn.Sequential(
            Conv(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        in_ch = features[0]
        for feature in features:
            self.decoder.append(UpScale(in_ch, feature, 2, dim=dim))
            self.decoder.append(ResConv(feature, feature, norm_type=norm_type, dim=dim))
            in_ch = feature
        
        if final_act and final_act == 'sigmoid':
            self.post_conv = nn.Sequential(
                Conv(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        elif final_act is None:
            self.post_conv = Conv(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError('Invalid final activation')

    def forward(self, x):
        for i, up in enumerate(self.decoder):
            x = up(x)
        x = self.post_conv(x)
        return x
  

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, embed_num, embed_dim, *, dim=3):
        super(VectorQuantizer, self).__init__()
        self.embedding_num = embed_num
        self.embedding_dim = embed_dim

        if dim == 2:
            self.permute_dims = (0, 2, 3, 1)
            self.permute_dims_back = (0, 3, 1, 2)
        elif dim == 3:
            self.permute_dims = (0, 2, 3, 4, 1)
            self.permute_dims_back = (0, 4, 1, 2, 3)

        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/self.embedding_num, 1.0/self.embedding_num)

    def forward(self, z:torch.Tensor):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,D,H,W)
            2. flatten input to (B*D*H*W,C)

        """
        # reshape z -> (batch, depth, height, width, channel) and flatten
        z = z.permute(*self.permute_dims).contiguous()
        
        embed_flattened = z.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(embed_flattened**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) \
            - 2*torch.matmul(embed_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.embedding_num).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # reshape back to match original input shape
        z_q = z_q.permute(*self.permute_dims_back).contiguous()

        return z_q


class VQVAE(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=1, features:list=[64,128,256], embedding_num:int=128, embedding_dim:int=256, norm_type=None, dim=3):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(in_channels=in_channels, out_channels=embedding_dim, features=features, norm_type=norm_type, dim=dim)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(embedding_num, embedding_dim, dim=dim)

        # decode the discrete latent representation
        self.decoder = Decoder(in_channels=embedding_dim, out_channels=out_channels, features=features[::-1], norm_type=norm_type, dim=dim)

    def forward(self, z):
        z_e = self.encoder(z)

        z_q = self.vector_quantization(z_e)

        z_q_grad = z_e + (z_q - z_e).detach()
        z_rec = self.decoder(z_q_grad)
        
        return z_e, z_q, z_rec

class VQVAE_Seg(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=1, features:list=[64,128,256], embedding_num:int=128, embedding_dim:int=256, norm_type=None):
        super(VQVAE_Seg, self).__init__()
        self.vqvae = VQVAE(in_channels=in_channels, out_channels=in_channels, features=features, embedding_num=embedding_num, embedding_dim=embedding_dim, norm_type=norm_type, dim=3)
        self.seg_decoder = Decoder(in_channels=embedding_dim, out_channels=out_channels, features=features[::-1], norm_type=norm_type, final_act='sigmoid', dim=3)

    def load_pretrain(self, ckpt_path, gpu):
        device = torch.device(f'cuda:{gpu}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.vqvae.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

    def forward(self, z):
        z_e, z_q, z_rec = self.vqvae(z)
        z_mask = self.seg_decoder(z_q.detach())
        return z_e, z_q, z_rec, z_mask


def get_model(args):
    model = VQVAE_Seg(in_channels=args.in_channels, out_channels=args.out_channels, features=args.features, 
                     embedding_num=args.embedding_num, embedding_dim=args.embedding_dim, norm_type=args.norm_type)
    return model


if __name__ == '__main__':
    from torchinfo import summary
    import argparse

    device = torch.device('cuda:0')
    size = 64
    z = torch.rand(1,1,size,size,size).to(device)

    args = argparse.Namespace
    args.in_channels = 1
    args.out_channels = 1
    args.features = [32,64,128]
    args.embedding_num = 64
    args.embedding_dim = 128
    args.norm_type = None
    args.load_size = 64
    args.data_norm_type = 'min_max'
    args.pretrain_ckpt = '/home/ryuuyou/E5/project/VQ-VAE/out/weights/vqvae_tiny/Epoch_1000.pth'

    model = get_model(args)
    model.to(device)
    model.load_pretrain(args.pretrain_ckpt, 0)
    summary(model, (1,1,size,size,size))
    z_e, z_q, z_rec, z_mask = model(z)