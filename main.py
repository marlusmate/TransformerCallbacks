from Modelcode.swin import SwinTransformer
from Modelcode.vit import VisionTransformer
from Modelcode.vswin import SwinTransformer3D
from Modelcode.vswin_multimodal import SwinTransformer3D as MSwinTransformer3D
from Modelcode.vivit import VisionTransformer3D
from Data import build_loader, load_yaml
#import pytorch_lightning as pl
import torch.nn as nn
from torch import device, cuda,  optim
from callbacks import *
from learner import Learner
from fastai.optimizer import OptimWrapper, Optimizer
from logger import logging
import os

# Setup
config = load_yaml('config.yaml')
logger = logging.getLogger('vswin_logger')
train_device = device('cuda:0' if cuda.is_available() else 'cpu')
callback_dir = os.path.join("Models", config["model_name"])
# Init Model
model = MSwinTransformer3D(num_classes=2, load_weights=config["pretrained"], patch_size=config["patch_size"], window_size=config["window_size"], logger=logger, 
    drop_path_rate=config["drop_path_rate"], drop_rate=config["drop_rate"], attn_drop_rate=config["attn_drop_rate"]).to(train_device)
#model = VisionTransformer3D(num_classes=2,img_size=(4,224,224), patch_size=(2,16,16), weight_init=config["pretrained"],
    #drop_rate=config["drop_rate"], attn_drop_rate=config["attn_drop_rate"], drop_path_rate=config["drop_path_rate"]).to(train_device)
#model = VisionTransformer(num_classes=3, weight_init=config["pretrained"], drop_path_rate=0.1, drop_rate=.3, attn_drop_rate=0.1).to(train_device)
#model = SwinTransformer(num_classes=3, load_weights=config["pretrained"], drop_path_rate=0.1, drop_rate=0.3, attn_drop_rate=0.2).to(train_device)

# Loss, Optimizer, Dataloader
#loss = nn.CrossEntropyLoss()
loss = nn.HuberLoss()
opt_func = OptimWrapper(opt=optim.AdamW(model.parameters()))
train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, 
    bs=config["batch_size"], device=train_device, train_sz=config["train_sz"])

# Learner
learner = Learner(config, model, loss, train_loader, val_loader, opt_func,
    min_delta=config["min_delta_loss"], min_val_loss=config["min_val_loss"], callback_dir=callback_dir, patience=config["patience"]
) 


# Training
mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
mlflow.set_tags(config['tags'])
mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"]), artifact_path=config["model_name"])

learner.pv_learn(config["epochs_total"], params=config["PVs"], n_iter=train_loader.__len__(), loss_we=[0.5, 0.5])
#learner.fit_one_cycle(epochs=config["epochs_total"], n_iter=train_loader.__len__(), lr_max=config["base_lr"])
#learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=config["base_lr"])
#learner.test(test_loader)
learner.test_pv(test_loader)
mlflow.end_run()
learner.save_model()
