import numpy as np
import os
import random
import torch
from torch.utils import data
import tifffile as tiff
from typing import Union

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())


def random_crop(img:np.ndarray, patch_size, start_d=None, start_h=None, start_w=None):
    d, h, w= img.shape[-3:]
    # randomly choose top and left coordinates
    if start_d is None:
        start_d = random.randint(0, d - patch_size)
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)
    img = img[..., start_d:start_d+patch_size, start_h:start_h+patch_size, start_w:start_w+patch_size]
    return img

# def norm_min_max(img:np.ndarray, percentiles=[0,1], return_info=False):
#     flattened_arr = np.sort(img.flatten())
#     clip_low = int(percentiles[0] * len(flattened_arr))
#     clip_high = int(percentiles[1] * len(flattened_arr))
#     clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

#     min_value = np.min(clipped_arr)
#     max_value = np.max(clipped_arr) 
#     img = (clipped_arr-min_value)/(max_value-min_value)
#     if return_info:
#         return img, min_value, max_value
#     return img

def norm_min_max(img:Union[np.ndarray,torch.Tensor], min_value=None, max_value=None, return_info=False, *args, **kwargs):
    if not min_value:
        min_value = img.min()
    if not max_value:
        max_value = img.max()
    img = (img - min_value) / (max_value - min_value)
    if return_info:
        return img, min_value, max_value
    return img

def norm_abs(img:Union[np.ndarray,torch.Tensor], abs_value=65535, *args, **kwargs):
    assert abs_value >= 0, 'invalid abs_value'
    img = img/abs_value
    return img

def norm_median(img:Union[np.ndarray,torch.Tensor], median=None, abs_value=65535, return_info=False, *args, **kwargs):
    assert abs_value >= 0, 'invalid abs_value'
    if median is None and isinstance(img, np.ndarray):
        median = np.median(img)
    elif median is None and isinstance(img, torch.Tensor):
        median = torch.median(img)
    # img = (img-median)/abs_value
    img = img - median
    if return_info:
        return img, median
    return img

def norm_identity(img:Union[np.ndarray,torch.Tensor], *args, **kwargs):
    return img

def norm_fn(type):
    if type=='min_max':
        return norm_min_max
    elif type=='abs':
        return norm_abs
    elif type=='median':
        return norm_median
    elif type is None:
        return norm_identity
    else:
        raise NotImplementedError(f'{type} normalization is not found')


def augment(img:torch.Tensor, dim_keep=-2, flip=True, transpose=True) -> torch.Tensor:
    dim_alter = [-3,-2,-1]
    dim_alter.remove(dim_keep)
    if flip and random.random() < 0.5:
        img = torch.flip(img, dims=[dim_alter[0]])
        img = torch.flip(img, dims=[dim_alter[1]])
    if flip and random.random() < 0.5:
        img = torch.flip(img, dims=[dim_keep])
    if transpose and random.random() < 0.5:
        img = img.transpose(dim_alter[0], dim_alter[1])
    return img


class tif_dataset(data.Dataset):
    def __init__(self, data_path:str, data_norm_type:str='min_max', augment:bool=True, aniso_dim:int=-2, load_size:int=None, full_mip:bool=False):
        super(tif_dataset, self).__init__()
        img_name_list = os.listdir(data_path)
        self.img_path_list = []
        for img_name in img_name_list:
            self.img_path_list.append(os.path.join(data_path, img_name))
        
        self.data_norm_type = data_norm_type
        self.augment = augment
        self.aniso_dim = aniso_dim
        self.load_size = load_size
        self.full_mip = full_mip

    def getitem_np(self, index):
        img_path = self.img_path_list[index]
        img = tiff.imread(img_path).astype(np.float32)
        if self.load_size and self.load_size < img.shape[-1]:
            img = random_crop(img, patch_size=self.load_size)
        if not self.full_mip:
            img = norm_fn(self.data_norm_type)(img)
        return img
        
    def __getitem__(self, index):
        img_np = self.getitem_np(index)
        img_tensor = torch.from_numpy(img_np)
        if self.augment:
            img_tensor = augment(img_tensor, self.aniso_dim)
        img_tensor = img_tensor[None]
        return img_tensor
    
    def __len__(self):
        return len(self.img_path_list)
    

def get_dataset(args):
    train_dataset = tif_dataset(data_path=args.data,
                                data_norm_type=args.data_norm_type,
                                augment=args.augment,
                                aniso_dim=args.aniso_dim,
                                load_size=args.load_size,
                                full_mip=args.full_mip)
    return train_dataset

if __name__ == '__main__':
    from ryu_pytools import arr_info, plot_mip, plot_some
    import napari
    from lib.utils.utils import get_slant_mip

    data_path = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/128_8k'
    ds = tif_dataset(data_path, load_size=64)
    print(len(ds))
    cube = ds.__getitem__(np.random.randint(len(ds)))
    mip = get_slant_mip(cube[None], iso_dim=-1)

    cube_np = cube[0].cpu().numpy()
    mip_np = mip[0,0].cpu().numpy()
    arr_info(cube_np)
    arr_info(mip_np)

    # plot_mip(cube_np, figsize=(15,5))
    # plot_some([mip_np])
    # pass
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(cube_np)
    viewer.add_image(mip_np)
    napari.run()