from Modelcode.swin import SwinTransformer
from Modelcode.vit import VisionTransformer
from Modelcode.vswin import SwinTransformer3D
from Modelcode.vswin_multimodal import SwinTransformer3D as MSwinTransformer3D
from Modelcode.vivit import VisionTransformer3D
from Data import build_loader, load_yaml, dump_json
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
            attn_drop_rate=config["attn_drop_rate"],
            final_actv=config["final_actv"],
        ).to(train_device)

elif config["testonly"]:
    model = load(config["pretraineddir"])
else:
    # Transfer Learning model
    pretrained_dir = config["pretraineddir"]
    model = load(pretrained_dir)
    if config["reset_head"]: model.reset_classifier(num_classes=3)
    elif config["overhead"]: model.head = nn.Sequential(
        model.head,
        nn.GELU(),
        nn.Linear(config["PVs"], 1024),
        nn.GELU(),
        nn.Linear(1024, config["num_classes"]),
        nn.Softmax()
    )
    model.to(train_device)

# Loss, Optimizer, Dataloader
if 'cross' in config["loss"]:
    loss = nn.CrossEntropyLoss()
elif 'huber' in config["loss"]:
    loss = nn.HuberLoss()
else:
    print("no loss function bruh")

train_loader, _, _, inst_dist1 = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, 
    bs=config["batch_size"], device=train_device, train_sz=0.99, fldir=config["fldir"])
test_loader, val_loader, _, inst_dist2 = build_loader(bs=1,train_sz=config["val_sz"], val_sz=0.99, fldir=config["test_dir"], n_inst=config["train_inst"], seq_len=config["seq_len"], seq=config["seq_len"]>0)
inst_dist = {'Training': inst_dist1['Training'], 'Validation': inst_dist2['Training'], 'Testing': inst_dist2['Validation']}
dump_json(inst_dist, dest=os.path.join(config["eval_dir"], config["model_name"])+'/InstanceDistribution.json')
print("IntsanceDistribution saved")

if config["opt"] == 'fastai':
    opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))
else:
    opt_func = optim.AdamW(model.parameters(), lr=config["base_lr"])

# Learner
learner = Learner(config, model, loss, train_loader, val_loader, opt_func, opt=None,
    min_delta=config["min_delta_loss"], min_val_loss=config["min_val_loss"], callback_dir=callback_dir, patience=config["patience"]
) 


# Training
mlflow.end_run()
mlflow.set_experiment("Markus_Transformer")
with mlflow.start_run(run_name=config["model_name"]):
    mlflow.set_tags(config['tags'])
    mlflow.log_artifact("config.yaml", artifact_path=config["model_name"])
    mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"])+'/InstanceDistribution.json', artifact_path=config["model_name"])
    if config["testonly"]:
        learner.test(test_loader)

    elif config["pv_learning"]:
        learner.pv_learn(config["epochs_total"], params=config["PVs"], n_iter=train_loader.__len__(), loss_we=[0.5, 0.5])
        learner.test_pv(test_loader)
    elif config["fine_tune"]:
        learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=config["base_lr"])
        learner.test(test_loader)
    else:
        learner.fit_one_cycle(epochs=config["epochs_total"], n_iter=train_loader.__len__(), lr_max=config["base_lr"])
        learner.test(test_loader)
    
mlflow.end_run()
#learner.save_model()
