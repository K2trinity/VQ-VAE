import os, sys
sys.path.append(os.getcwd())
from lib.arch.vqvae import get_model
from lib.dataset.npz_dataset import norm_fn
from ruamel.yaml import YAML
import argparse
import torch
import numpy as np
import tifffile as tiff
from ryu_pytools import plot_some, arr_info, tensor_to_ndarr

def read_cfg(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        yaml = YAML(typ='safe', pure=True)
        yml = yaml.load(f)
    cmd = [c[1:] for c in sys.argv if c[0]=='-']
    for k,v in yml.items():
        if k not in cmd:
            args.__dict__[k] = v
    return args

def process_oneImage(model:torch.nn.Module, img:np.ndarray, device:torch.device):
    img = norm_fn('min_max')(img)
    img = torch.from_numpy(img)[None, None].to(device)
    img_e, img_q, img_rec = model(img)
    return tensor_to_ndarr(img_rec[0,0])

def main(cfg_path:str, ckpt_path:str, img_path:str, show:bool=True):
    args = read_cfg(cfg_path)
    device = torch.device('cuda:0')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    model = get_model(args).to(device)
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
    model.eval()

    img = tiff.imread(img_path).astype(np.float32)
    
    with torch.no_grad():
        img_rec = process_oneImage(model, img, device)

    if show:
        import napari
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(img, name='img')
        viewer.add_image(img_rec, name='img_rec')
        napari.run()
    return

if __name__ == '__main__':
    cfg_path = 'config/vqvae_tiny.yaml'
    ckpt_path = 'out/weights/vqvae_tiny/Epoch_1000.pth'
    img_dir = '/home/ryuuyou/E5/project/data/RESIN_datasets/neuron/64_1k'
    # img_dir = '/home/ryuuyou/E5/project/data/RM009_rotated_1k/img'
    img_name = os.listdir(img_dir)
    index = np.random.randint(0, len(img_name))
    img_path = os.path.join(img_dir, img_name[index])

    # cfg_path = 'config/nissl.yaml'
    # ckpt_path = 'out/weights/nissl/Epoch_1000.pth'
    # img_dir = '/home/ryuuyou/E5/project/data/RESIN_datasets/NISSL/Train_datasets_2k'
    # img_name = os.listdir(img_dir)
    # index = np.random.randint(0, len(img_name))
    # img_path = os.path.join(img_dir, img_name[index])

    img_path = '/home/ryuuyou/E5/project/data/RM009_instance_30/mask/mask_21.tif'
    print(img_path)
    main(cfg_path, ckpt_path, img_path, show=True)