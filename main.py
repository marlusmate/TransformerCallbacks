from lightningmodule.swin import SwinTransformer
from lightningmodule.vit_tiny import VisionTransformer
from lightningmodule.vswin import SwinTransformer3D
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
from fastai.optimizer import OptimWrapper, Optimizer
from logger import logging
import os
from learner_utils import dump_json

config = {
    'model_name' : 'vswin-tiny-patch4-window7-224',
    'epochs_total': 1,
    'epochs_froozen': 1,
    'n_inst': 200,
    'train_sz': 0.8,
    'seq_len': 4,
    'batch_size': 20,
    'tags' : {
        'Model': 'vswin-tiny-patch4-window7-224',
        'Type': 'Debug',
        'Seeds': [0]
    },
    'eval_dir' : "Evaluation",
    'seed' : 0
}
dump_json(config, os.path.join(config["eval_dir"], config["model_name"], "Hyperparameters.json"))

cb = CallbackHandler([BatchCounter()])
logger = logging.getLogger('vswin_logger')
model = SwinTransformer3D(patch_size=(4,4,4), window_size=(2,7,7), logger=logger).to('cuda')
#model = VisionTransformer(num_classes=3, drop_path_rate=0.2, drop_rate=.4, attn_drop_rate=0.2).to('cuda')
#model = SwinTransformer(num_classes=1, load_weights='', drop_path_rate=0., drop_rate=0., attn_drop_rate=0.).to('cuda')
loss = nn.CrossEntropyLoss()
#loss = nn.MSELoss()
opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))
opf_func = Optimizer

train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, bs=config["batch_size"])
#n_iter = ceil((config['n_inst'] * config["train_sz"]) /config["train"]["batch_sz"])
#sched = build_scheduler(config, opt, n_iter)
learner = Learner(config, model, loss, train_loader, val_loader, opt_func) 

mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
mlflow.set_tags(config['tags'])
mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"], ), artifact_path=config["model_name"])
#learner.fit_one_cycle(epochs=config["epochs_total"], n_iter=train_loader.__len__(), lr_max=8e-5)
learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=2e-3)
learner.test(test_loader)
#learner.test_pv(test_loader)
mlflow.end_run()
