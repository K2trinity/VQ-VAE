import torch
import torch.optim as Opt
import itertools

class Adam_Opt(Opt.Adam):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adam_Opt, self).__init__(model.parameters(), lr=lr_start)

class SGD_Opt(Opt.SGD):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(SGD_Opt, self).__init__(model.parameters(), lr=lr_start)

class Adagrad_Opt(Opt.Adagrad):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adagrad_Opt, self).__init__(model.parameters(), lr=lr_start)

class VQVAE_SR_Opt():
    def __init__(self, model, args) -> None:
        lr_G = args.lr_G
        lr_D = args.lr_D
        self.optimG = Opt.Adam(itertools.chain(model.G.parameters()), lr=lr_G)
        self.optimD = Opt.Adam(itertools.chain(model.D_MIP.parameters(), 
                                               model.D_Rec.parameters()), lr=lr_D)

def get_optimizer(args, model):
    opt_fns = {
        'adam': Adam_Opt,
        'sgd': SGD_Opt,
        'adagrad': Adagrad_Opt,
        'vqvae': Adam_Opt,
        'vqvae_sr': VQVAE_SR_Opt,
    }
    opt_fn = opt_fns.get(args.optimizer, "Invalid Optimizer")
    opt = opt_fn(model.module, args)
    return opt