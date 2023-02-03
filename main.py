from Modelcode.swin import SwinTransformer
from Modelcode.vit import VisionTransformer
from Modelcode.vswin import SwinTransformer3D
from Modelcode.vswin_multimodal import SwinTransformer3D as MSwinTransformer3D
from Modelcode.vivit import VisionTransformer3D
from Data import build_loader, load_yaml
#import pytorch_lightning as pl
import torch.nn as nn
from torch import device, cuda, optim, load
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
if not config["transfer_learning"]:
    if 'vswin' in config["model_name"]:
        model = MSwinTransformer3D(
            num_classes=config["num_classes"], 
            load_weights=config["pretrained"], 
            patch_size=config["patch_size"], 
            window_size=config["window_size"], 
            logger=logger, 
            drop_path_rate=config["drop_path_rate"], 
            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"]        
        ).to(train_device)
    elif 'swin' in config["model_name"]:
        model = SwinTransformer(
            num_classes=config["num_classes"], 
            oad_weights=config["pretrained"], 
            drop_path_rate=config["drop_path_rate"], 
            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"]
        ).to(train_device)
    if 'vivit' in config["model_name"]:
        model = VisionTransformer3D(
            num_classes=config["num_classes"],
            img_size=(config["seq_len"],224,224), 
            patch_size=config["patch_size"], 
            weight_init=config["pretrained"],
            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"], 
            drop_path_rate=config["drop_path_rate"]
        ).to(train_device)
    elif 'vit' in config["model_name"]:
        model = VisionTransformer(
            num_classes=config["num_classes"],
            weight_init=config["pretrained"], 
            drop_path_rate=config["drop_path_rate"], 
            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"]
        ).to(train_device)
else:
    # Transfer Learning model
    pretrained_dir = "Models/"+config["model_name"]+"/model_callback_acc"
    model = load(pretrained_dir)
    model.reset_classifier(num_classes=3)
    model.to(train_device)

# Loss, Optimizer, Dataloader
if 'cross' in config["loss"]:
    loss = nn.CrossEntropyLoss()
elif 'huber' in config["loss"]:
    loss = nn.HuberLoss()
else:
    print("no loss function bruh")
opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))
train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, 
    bs=config["batch_size"], device=train_device, train_sz=config["train_sz"], fldir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain")

# Learner
learner = Learner(config, model, loss, train_loader, val_loader, opt_func,
    min_delta=config["min_delta_loss"], min_val_loss=config["min_val_loss"], callback_dir=callback_dir, patience=config["patience"]
) 


# Training
mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
with mlflow.start_run(run_name=config["model_name"]):
    mlflow.set_tags(config['tags'])
    mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"]), artifact_path=config["model_name"])

    if config["pv_learning"]:
        learner.pv_learn(config["epochs_total"], params=config["PVs"], n_iter=train_loader.__len__(), loss_we=[0.5, 0.5])
        learner.test_pv(test_loader)
    elif config["fine_tune"]:
        learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=config["base_lr"])
        learner.test(test_loader)
    else:
        learner.fit_one_cycle(epochs=config["epochs_total"], n_iter=train_loader.__len__(), lr_max=config["base_lr"])
        learner.test(test_loader)
    
mlflow.end_run()
learner.save_model()
