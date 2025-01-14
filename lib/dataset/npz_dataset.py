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

class npz_dataset(data.Dataset):
    def __init__(self, data_path, data_norm_type='min_max', load_size=None):
        super(npz_dataset, self).__init__()
        file = np.load(data_path)
        self.IMAGES = file['Y']
        
        self.data_norm_type = data_norm_type
        self.load_size = load_size

    def getitem_np(self, index):
        img = self.IMAGES[index]
        if self.load_size and self.load_size < img.shape[-1]:
            img = random_crop(img, patch_size=self.load_size)
        img = norm_fn(self.data_norm_type)(img)
        return img
        
    def __getitem__(self, index):
        img_np = self.getitem_np(index)
        img_tensor = torch.from_numpy(img_np)
        img_tensor = img_tensor
        return img_tensor
    
    def __len__(self):
        return len(self.IMAGES)

def random_crop(img:np.ndarray, patch_size, start_h=None, start_w=None):
    h, w= img.shape[-2:]
    # randomly choose top and left coordinates
    if start_h is None:
        start_h = random.randint(0, h - patch_size)
    if start_w is None:
        start_w = random.randint(0, w - patch_size)
    img = img[..., start_h:start_h+patch_size, start_w:start_w+patch_size]
    return img

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

def get_dataset(args):
    load_size = args.load_size if hasattr(args, 'load_size') else None
    train_dataset = npz_dataset(data_path=args.data,
                                data_norm_type=args.data_norm_type,
                                load_size=load_size)
    return train_dataset

if __name__ == '__main__':
    from ryu_pytools import arr_info
    import napari

    data_path = '/home/ryuuyou/E5/project/data/CARE/Isotropic_Liver/train_data/data_label.npz'
    ds = npz_dataset(data_path, load_size=64)
    print(len(ds))
    cube = ds.__getitem__(np.random.randint(len(ds)))
    arr_info(cube)
    cube_np = cube[0].cpu().numpy()
    arr_info(cube_np)