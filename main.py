from lightningmodule.vit_module import vit_encoder
from lightningmodule.vit_tiny import VisionTransformer
from lightningmodule.VSwinV2 import SwinTransformer3D
from finetuning_scheduler import FinetuningScheduler
from Data import build_loader
import pytorch_lightning as pl
import torch.nn as nn
from torch import save, optim
from callbacks import *
from learner import Learner
from model_optimizer import build_adamw
from lr_scheduler import build_scheduler
from math import ceil
from fastai.optimizer import OptimWrapper
from logger import logging
import os
from learner_utils import dump_json

config = {
    'model_name' : 'vit-tiny-patch16-224',
    'epochs_total': 7,
    'epochs_froozen': 3,
    'n_inst': 6000,
    'train_sz': 0.8,
    'seq_len': 0,
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
    'tags' : {
        'Model': 'vit-tiny-patch16-224',
        'Type': 'Debug',
        'Seeds': [0]
    },
    'eval_dir' : "Evaluation"
}
dump_json(config, os.path.join(config["eval_dir"], config["model_name"], "Hyperparameters.json"))

cb = CallbackHandler([BatchCounter()])
logger = logging.getLogger('vswin_logger')
#model = SwinTransformer3D(logger=logger).to('cuda')
model = VisionTransformer(drop_path_rate=0.2, drop_rate=.4, attn_drop_rate=0.2).to('cuda')
freeze_epochs=0,
frozen_stages=12
loss = nn.CrossEntropyLoss()
opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))

train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, bs=3)
n_iter = ceil((config['n_inst'] * config["train_sz"]) /config["train"]["batch_sz"])
opt = build_adamw(model, epsilon=0.0000001, betas=[0.9, 0.999], lr=0.001, we_decay=0.05)
#sched = build_scheduler(config, opt, n_iter)
learner = Learner(config, model, loss, train_loader, val_loader, opt_func) 

mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
mlflow.set_tags(config['tags'])
mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"], ), artifact_path=config["model_name"])
learner.fine_tune(config["epochs_total"],config["epochs_froozen"], n_iter*config["epochs_froozen"])
learner.test(test_loader)
mlflow.end_run()
