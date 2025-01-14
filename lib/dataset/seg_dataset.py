import numpy as np
import os
import random
import torch
from torch.utils import data
import tifffile
from typing import Union

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

def random_crop(img:np.ndarray, patch_size, start_d=None, start_h=None, start_w=None, return_start=False):
    d, h, w= img.shape[-3:]
    # randomly choose top and left coordinates
    if start_d is None:
        start_d = random.randint(0, d - patch_size)
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)
    img = img[..., start_d:start_d+patch_size, start_h:start_h+patch_size, start_w:start_w+patch_size]
    if return_start:
        return img, start_d, start_h, start_w
    return img

def norm_min_max(img:np.ndarray, percentiles=[0.01, 0.9999], return_info=False):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    if return_info:
        return img, min_value, max_value
    return img

# def norm_min_max(img:Union[np.ndarray,torch.Tensor], min_value=None, max_value=None, return_info=False, *args, **kwargs):
#     if not min_value:
#         min_value = img.min()
#     if not max_value:
#         max_value = img.max()
#     img = (img - min_value) / (max_value - min_value)
#     if return_info:
#         return img, min_value, max_value
#     return img

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

def random_rotate(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = random.randint(0, 3)  # Randomly choose number of 90 degree rotations
    dims = random.sample([-3, -2, -1], 2)  # Randomly choose two dimensions to rotate
    img = torch.rot90(img, k, dims=dims)
    mask = torch.rot90(mask, k, dims=dims)
    return img, mask

class tif_dataset(data.Dataset):
    def __init__(self, data_path, data_norm_type='min_max', augment=True, load_size=None):
        super(tif_dataset, self).__init__()
        img_dir = os.path.join(data_path, 'img')
        mask_dir = os.path.join(data_path, 'mask')

        img_name_list = sorted(os.listdir(img_dir))*10
        self.img_path_list = []
        for img_name in img_name_list:
            self.img_path_list.append(os.path.join(img_dir, img_name))

        mask_name_list = sorted(os.listdir(mask_dir))*10
        self.mask_path_list = []
        for mask_name in mask_name_list:
            self.mask_path_list.append(os.path.join(mask_dir, mask_name))
        
        self.data_norm_type = data_norm_type
        self.augment = augment
        self.load_size = load_size

    def getitem_np(self, index):
        img_path = self.img_path_list[index]
        mask_path = self.mask_path_list[index]

        img = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.float32)

        if self.load_size and self.load_size < img.shape[-1]:
            img, start_d, start_h, start_w = random_crop(img, patch_size=self.load_size, return_start=True)
            mask = random_crop(mask, patch_size=self.load_size, start_d=start_d, start_h=start_h, start_w=start_w)

        img = norm_fn(self.data_norm_type)(img)
        return img, mask
        
    def __getitem__(self, index):
        img_np, mask_np = self.getitem_np(index)
        img_tensor = torch.from_numpy(img_np)
        mask_tensor = torch.from_numpy(mask_np)
        if self.augment:
            img_tensor, mask_tensor = random_rotate(img_tensor, mask_tensor)
        img_tensor = img_tensor[None]
        mask_tensor = mask_tensor[None]
        return img_tensor, mask_tensor
    
    def __len__(self):
        return len(self.img_path_list)

def get_dataset(args):
    train_dataset = tif_dataset(data_path=args.data,
                                augment=args.augment,
                                load_size=args.load_size)
    return train_dataset

if __name__ == '__main__':
    from ryu_pytools import arr_info
    import napari

    data_path = '/home/ryuuyou/E5/project/data/RM009_instance_30'
    ds = tif_dataset(data_path, data_norm_type='min_max', load_size=64)
    print(len(ds))
    img, mask = ds.__getitem__(np.random.randint(len(ds)))
    img_np = img[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()
    arr_info(img_np)
    arr_info(mask_np)
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img_np)
    viewer.add_image(mask_np)
    napari.run()