# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import cosine_scheduler
from lib.utils.distributed import MetricLogger
from glob import glob
import math
import pprint

import sys, os

class Trainer:

    def __init__(self, args, dataloader, model, loss, optimizer):
        pprint.pprint(vars(args))

        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

        if hasattr(args, 'pretrain_ckpt') and args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
            self.model.module.load_pretrain(args.pretrain_ckpt, args.gpu)
            print(f'\nload pre-trained model from "{args.pretrain_ckpt}"\n')

        # === TB writers === #
        if self.args.main:	
            self.writer = get_writer(args)		
            checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)

    def train_one_epoch(self, epoch, lr_schedule):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)

        lr_G_schedule, lr_D_schedule = lr_schedule
        for it, gt in enumerate(metric_logger.log_every(self.dataloader, 10, header)):
            # === Global Iteration === #
            it = len(self.dataloader) * epoch + it

            for i, param_group in enumerate(self.optimizer.optimG.param_groups):
                param_group["lr"] = lr_G_schedule[it]
            for i, param_group in enumerate(self.optimizer.optimD.param_groups):
                param_group["lr"] = lr_D_schedule[it]

            # === Inputs === #
            gt = gt.cuda(non_blocking=True)

            # === Forward pass === #
            with torch.cuda.amp.autocast(self.args.fp16):
                model_out = self.model(gt)
                loss_G, loss_D, loss_logger, mip_logger = self.loss(gt, model_out, it)

            # Sanity Check
            if not math.isfinite(loss_G.item()):
                print("Loss is {}, stopping training".format(loss_G.item()), force=True)
                sys.exit(1)
            if not math.isfinite(loss_D.item()):
                print("Loss is {}, stopping training".format(loss_D.item()), force=True)
                sys.exit(1)
            
            # === Backward pass === #

            # === optimize G ===
            if (it+1) % self.args.G_train_it == 0: 
                self.optimizer.optimG.zero_grad()
                if self.args.fp16:
                    self.fp16_scaler.scale(loss_G).backward()
                    self.fp16_scaler.step(self.optimizer.optimG)
                    self.fp16_scaler.update()
                else:
                    loss_G.backward()
                    self.optimizer.optimG.step()

            # === optimize D ===
            if (it+1) % self.args.D_train_it == 0: 
                self.optimizer.optimD.zero_grad()
                if self.args.fp16:
                    self.fp16_scaler.scale(loss_D).backward()
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()
                else:
                    loss_D.backward()
                    self.optimizer.optimD.step()
            
            # === Logging === #
            torch.cuda.synchronize()
            metric_logger.update(**loss_logger)

            if self.args.main:
                # loss logger
                for key in loss_logger.keys():
                    self.writer.add_scalar(key, metric_logger.meters[key].value, it)

                # mip logger
                if it % 100 == 0:
                    for name, mip in mip_logger.items():
                        self.writer.add_image(name, mip, it)

                # lr logger
                self.writer.add_scalar('LearningRate/LR_G', self.optimizer.optimG.param_groups[0]["lr"], it)
                self.writer.add_scalar('LearningRate/LR_D', self.optimizer.optimD.param_groups[0]["lr"], it)


        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    def fit(self):
        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        lr_G_schedule = cosine_scheduler(
                        base_value = self.args.lr_G,
                        final_value = self.args.lr_end,
                        epochs = self.args.epochs,
                        niter_per_ep = len(self.dataloader),
                        warmup_epochs= self.args.lr_warmup,
        )
        lr_D_schedule = cosine_scheduler(
                        base_value = self.args.lr_D,
                        final_value = self.args.lr_end,
                        epochs = self.args.epochs,
                        niter_per_ep = len(self.dataloader),
                        warmup_epochs= self.args.lr_warmup,
        )
        lr_schedule = [lr_G_schedule, lr_D_schedule]

        # === training loop === #
        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, lr_schedule)

            # === save model === #
            if self.args.main and (epoch+1)%self.args.save_every == 0:
                self.save(epoch)

    def load_if_available(self):
        ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

        if len(ckpts) > 0:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.optimG.load_state_dict(ckpt['optimizer_G'])
            self.optimizer.optimD.load_state_dict(ckpt['optimizer_D'])
            if self.args.fp16: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
            print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            print("Starting from scratch")


    def save(self, epoch):
        if self.args.fp16:
            state = dict(epoch=epoch+1, 
                         model=self.model.state_dict(), 
                         optimizer_G=self.optimizer.optimG.state_dict(),
                         optimizer_D=self.optimizer.optimD.state_dict(),
                         fp16_scaler = self.fp16_scaler.state_dict(),
                         args = self.args
                        )
        else:
            state = dict(epoch=epoch+1, 
                         model=self.model.state_dict(), 
                         optimizer_G=self.optimizer.optimG.state_dict(),
                         optimizer_D=self.optimizer.optimD.state_dict(),
                         args = self.args
                        )

        torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args.out, self.args.model, str(epoch+1).zfill(4) ))
