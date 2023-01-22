from lightningmodule.vit_module import vit_encoder
from lightningmodule.vit_tiny import VisionTransformer
from finetuning_scheduler import FinetuningScheduler
from Data import build_loader
import pytorch_lightning as pl
import torch.nn as nn
from torch import load
from callbacks import *
from learner import Learner
from model_optimizer import build_adamw
from lr_scheduler import build_scheduler
from math import ceil

config = {
    'epochs': 2,
    'n_inst': 200,
    'train_sz': 0.8,
    'train': {
        'batch_sz': 20
    },
    'scheduler': {
        'warmup_ep': 0,
        'decay_ep': 5,
        'ml_step': [],
        'name': 'cosine',
        'warmup_prefix': True,
        'min_lr': 0.00001,
        'warmup_lr' : 0.00008,
    },
}

cb = CallbackHandler([BatchCounter()])
model = VisionTransformer().to('cuda')
loss = nn.CrossEntropyLoss()

train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'])
n_iter = ceil((config['n_inst'] * config["train_sz"]) /config["train"]["batch_sz"])
opt = build_adamw(model, epsilon=0.0000001, betas=[0.9, 0.999], lr=0.00001, we_decay=0.05)
sched = build_scheduler(config, opt, n_iter)
#look for link in text to see simple_net, mnist_loss, opt, dl, valid_dl initializations
learner = Learner(model, loss, opt, sched, train_loader, val_loader, cb=cb) 
fit(2, learn=learner)

model = vit_encoder()

cbs = [FinetuningScheduler(ft_schedule="ft_schedule/vit_encoder_ft_schedule.yaml")]

triping = pl.Trainer(limit_train_batches=50, max_epochs=10, callbacks=cbs, gpus=1)
triping.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
