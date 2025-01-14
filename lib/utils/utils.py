import torch
import math
import random
import numpy as np
import os

def affine_sample(img:torch.Tensor, angel=-45):
    b,c,d,h,w = img.shape
    angel = math.radians(angel)
    theta = torch.tensor(
                [[math.cos(angel), 0, math.sin(angel), 0],
                [0, 1, 0, 0],
                [-math.sin(angel), 0, math.cos(angel), 0]], 
                dtype=torch.float, device=img.device)
    theta = theta.repeat(b,*[1,]*len(theta.shape))
    grid = torch.nn.functional.affine_grid(theta, img.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(img, grid=grid, align_corners=False)
    return out

def get_slant_mip(img:torch.Tensor, angel=-45, iso_dim=-1, augment=True):
    img_affine = affine_sample(img, angel=angel)
    mip = torch.max(img_affine, dim=iso_dim).values
    if augment and random.random() < 0.5:
        mip = torch.flip(mip, dims=[-1])
    if augment and random.random() < 0.5:
        mip = torch.flip(mip, dims=[-2])
    if augment and random.random() < 0.5:
        mip = mip.transpose(-1,-2)
    return mip

def center_crop(img:torch.Tensor, crop_size=64, *, dim=3):
    assert crop_size%2==0, 'invalid crop size'
    c = np.asarray(img.shape[-dim:])//2
    r = np.asarray([crop_size//2]*dim)
    if dim == 3:
        ds,hs,ws = c-r
        de,he,we = c+r
        img_crop = img[..., ds:de, hs:he, ws:we]
    elif dim == 2:
        hs,ws = c-r
        he,we = c+r
        img_crop = img[..., hs:he, ws:we]
    return img_crop

def check_dir(path, *, mode:str='r'):
    if os.path.exists(path):
            print(f'the directory already exists: ["{path}"]')
    else:
        if mode == 'r':
            os.makedirs(path)
            print(f'the directory has been created: ["{path}"]')
        elif mode == 'a':
            os.mkdir(path)
            print(f'the directory has been created: ["{path}"]')
        else:
            print(f'the directory doesn\'t exist: ["{path}"]')
