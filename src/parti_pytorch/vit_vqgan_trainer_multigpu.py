from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree
from PIL import Image

import time
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from einops import rearrange

from parti_pytorch.vit_vqgan import VitVQGanVAE
from parti_pytorch.optimizer import get_optimizer

from ema_pytorch import EMA

from accelerate import Accelerator

from accelerate import DistributedDataParallelKwargs

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# classes

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# main trainer class

class VQGanVAETrainerMGPU(nn.Module):
    def __init__(
        self,
        vae_config,
        *,
        num_train_steps,
        batch_size,
        folder,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.0005,
        random_split_seed = 42,
        ema_beta = 0.995,
        ema_update_after_step = 500,
        ema_update_every = 10,
        apply_grad_penalty_every = 4,
        amp = False
    ):
        super().__init__()
        image_size = vae_config['image_size']

        # self.accelerator = Accelerator(mixed_precision="fp16",kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        self.accelerator = Accelerator(mixed_precision="no",kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        self.device = self.accelerator.device
        
                
        self.vae =  VitVQGanVAE(
                dim = vae_config['dim'],               # dimensions
                image_size = vae_config['image_size'],        # target image size
                patch_size = vae_config['patch_size'],         # size of the patches in the image attending to each other
                num_layers = vae_config['num_layers'],           # number of layers
                vq_codebook_dim = vae_config['vq_codebook_dim'],
                vq_codebook_size = vae_config['vq_codebook_size'],
            ).to(self.device)
        self.ema_vae = EMA(self.vae, update_after_step = ema_update_after_step, update_every = ema_update_every).to(self.device)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(self.vae.parameters())
        discr_parameters = set(self.vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.discr_scaler = GradScaler(enabled = amp)

        # create dataset

        self.ds = ImageDataset(folder, image_size = image_size)

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.accelerator.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.accelerator.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            num_workers = 16,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            num_workers = 16,
            shuffle = True
        )

        self.vae, self.ema_vae, self.optim, self.dl, self.valid_dl, self.scaler, self.discr_scaler = self.accelerator.prepare(self.vae, self.ema_vae, self.optim, self.dl, self.valid_dl, self.scaler, self.discr_scaler)

        self.dl = cycle(self.dl)
        self.valid_dl = cycle(self.valid_dl)
        
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and self.accelerator.is_main_process:
            rmtree(str(self.results_folder))
        
        self.results_folder.mkdir(parents = True, exist_ok = True)

    def train_step(self):
        # device = next(self.vae.parameters()).device
        device = self.device
        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl)
            img = img.to(device)

            with autocast(enabled = self.amp):
                loss = self.vae(
                    img,
                    return_loss = True,
                    apply_grad_penalty = apply_grad_penalty
                )
                # self.scaler.scale(loss / self.grad_accum_every).backward()
                self.accelerator.backward(self.scaler.scale(loss / self.grad_accum_every))

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad()

        # update discriminator

        if exists(self.vae.module.discr):
            self.discr_optim.zero_grad()

            discr_loss = 0
            for _ in range(self.grad_accum_every):
                img = next(self.dl)
                img = img.to(device)

                with autocast(enabled = self.amp):
                    loss = self.vae(img, return_discr_loss = True)

                    # self.discr_scaler.scale(loss / self.grad_accum_every).backward()
                    self.accelerator.backward(self.discr_scaler.scale(loss / self.grad_accum_every))

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            self.discr_scaler.step(self.discr_optim)
            self.discr_scaler.update()

            # log
            cur_time = time.strftime("%Y-%m-%d %H:%M:%s", time.localtime(time.time()))
            self.accelerator.print(f"[{cur_time}] {steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")
    
        # update exponential moving averaged generator

        self.ema_vae.module.update()

        # sample results every so often

        if not (steps % self.save_results_every):
            for model, filename in ((self.ema_vae.module.ema_model, f'{steps}.ema'), (self.vae, str(steps))):
                model.eval()

                imgs = next(self.dl)
                imgs = imgs.to(device)

                recons = model(imgs)
                nrows = int(sqrt(self.batch_size))

                imgs_and_recons = torch.stack((imgs, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

            self.accelerator.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        if not (steps % self.save_model_every):
            self.accelerator.wait_for_everyone()
            state_dict = self.vae.module.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            ema_state_dict = self.ema_vae.module.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            self.accelerator.save(ema_state_dict, model_path)

            # self.accelerator.save()
            self.accelerator.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        # device = next(self.vae.parameters()).device
        device = self.device

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.accelerator.print('training complete')
