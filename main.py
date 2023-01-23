from lightningmodule.vit_tiny import VisionTransformer
from lightningmodule.VSwinV2 import SwinTransformer3D
from Data import build_loader
import torch.nn as nn
from torch import save, optim
from callbacks import *
from learner import Learner
from model_optimizer import build_adamw
from math import ceil
from fastai.optimizer import OptimWrapper
from logger import logging
import os
from learner_utils import dump_json

config = {
    'model_name' : 'swinv2-tiny-patch4-window7-224',
    'epochs_total' : 6,
    'epochs_froozen': 6,
    'n_inst': 2000,
    'train_sz': 0.8,
    'seq_len': 0,
    'batch_size': 20,
    'frozen_stages' : 4,
    'train': {
        
    },
    'tags' : {
        'Model': 'swinv2-tiny-patch4-window7-224',
        'Type': 'Debug',
        'Seeds': [0]
    },
    'eval_dir' : "Evaluation",
    'data_dir' : "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed"
}
dump_json(config, os.path.join(config["eval_dir"], config["model_name"], "Hyperparameters.json"))

cb = CallbackHandler([BatchCounter()])
logger = logging.getLogger('vswin_logger')
model = SwinTransformer3D(logger=logger, frozen_stages=config["frozen_stages"]).to('cuda')
#model = VisionTransformer(drop_path_rate=0.2, drop_rate=.4, attn_drop_rate=0.2).to('cuda')
freeze_epochs=0,
frozen_stages=12
loss = nn.CrossEntropyLoss()
opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))

train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=False,  bs=config["batch_size"])
n_iter = ceil((train_loader.__len__()/config["batch_size"]))
opt = build_adamw(model, epsilon=0.0000001, betas=[0.9, 0.999], lr=0.002, we_decay=0.05)
#sched = build_scheduler(config, opt, n_iter)
learner = Learner(config, model, loss, train_loader, val_loader, opt_func) 

mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
mlflow.set_tags(config['tags'])
mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"], ), artifact_path=config["model_name"])
learner.fine_tune(config["epochs_total"],config["epochs_froozen"], n_iter=n_iter)
learner.test(test_loader)
mlflow.end_run()
print("Finetuning, Testing, Logging completed")

learner.save_model("Models/"+config["model_name"]+"/model_finetuned")
# learner.save_learner("Models/"+config["model_name"]+"learner_finetuned")
print(config["model_name"]+" - Model, Leaner saved")
