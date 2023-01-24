from lightningmodule.vit_tiny import VisionTransformer
from lightningmodule.VSwinV2 import SwinTransformer3D
from lightningmodule.swinv2 import SwinTransformerV2
from lightningmodule.swin import SwinTransformer
from Data import build_loader
import torch.nn as nn
from torch import save, optim, device, cuda
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
    'epochs_total' : 10,
    'epochs_froozen': 1,
    'base_lr' : 1e-3,
    'num_samples': 2000,
    'train_size': 0.7,
    'seq_len': 0,
    'batch_size': 21,
    'frozen_stages' : 4,
    'patch_size' : (1,4,4),
    'window_size' : (1,8,8),
    'train': {
        
    },
    'tags' : {
        'Model': 'swinv2-tiny-patch4-window7-224',
        'Type': 'BaseTraun',
        'Seeds': [0]
    },
    'eval_dir' : "Evaluation",
    'data_dir' : "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed"
}
dump_json(config, os.path.join(config["eval_dir"], config["model_name"], "Hyperparameters.json"))

cb = CallbackHandler([BatchCounter()])
logger = logging.getLogger('vswin_logger')
train_device = device('cuda:0' if cuda.is_available() else 'cpu')
#model = SwinTransformer3D(logger=logger, frozen_stages=config["frozen_stages"], patch_size=config["patch_size"], window_size=config["window_size"]).to(train_device)
#model = VisionTransformer(drop_path_rate=0.2, drop_rate=.4, attn_drop_rate=0.2).to('cuda')
model = SwinTransformerV2().to(train_device)
#model = SwinTransformer().to(train_device)
loss = nn.CrossEntropyLoss()
opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))

train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['num_samples'], seq_len=config["seq_len"], 
    seq=False, bs=config["batch_size"], fldir=config["data_dir"], device=train_device
    )
n_iter = ceil((config["train_size"]*config["num_samples"]*3 /config["batch_size"]))
learner = Learner(config, model, loss, train_loader, val_loader, opt_func) 

mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
mlflow.set_tags(config['tags'])
mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"], ), artifact_path=config["model_name"])
learner.fine_tune(epochs=config["epochs_total"], freeze_epochs=config["epochs_froozen"], n_iter=train_loader.__len__(), base_lr=config["base_lr"])
learner.test(test_loader)

mlflow.end_run()
print("Finetuning, Testing, Logging completed")

learner.save_model("Models/"+config["model_name"]+"/model_finetuned")
# learner.save_learner("Models/"+config["model_name"]+"learner_finetuned")
print(config["model_name"]+" - Model, Leaner saved")

