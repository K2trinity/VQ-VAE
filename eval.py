# ckpt = torch.load('out/weights/example/Epoch_100.pth', map_location='cpu')
# model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})