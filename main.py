from Modelcode.swin import SwinTransformer
from Modelcode.vit import VisionTransformer
from Modelcode.vswin import SwinTransformer3D
from Modelcode.vswin_multimodal import SwinTransformer3D as MSwinTransformer3D
from Modelcode.vivit import VisionTransformer3D
from Data import build_loader, load_yaml, dump_json
#import pytorch_lightning as pl
import torch.nn as nn
from torch import device, cuda, optim, load, save
from callbacks import *
from learner import Learner
from fastai.optimizer import OptimWrapper, Optimizer
from logger import logging
from model_optimizer import build_adamw
import os

# Setup
config = load_yaml('config.yaml')
logger = logging.getLogger('vswin_logger')
train_device = device('cuda:0' if cuda.is_available() else 'cpu')
callback_dir = os.path.join("Models", config["model_name"])
# Init Model
if not config["transfer_learning"] and not config["testonly"]:
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
            load_weights=config["pretrained"], 
            drop_path_rate=config["drop_path_rate"], 

            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"],
            final_actv=config["final_actv"]
        ).to(train_device)
    if 'vivit' in config["model_name"]:
        model = VisionTransformer3D(
            num_classes=config["num_classes"],
            img_size=(config["seq_len"],224,224), 
            patch_size=config["patch_size"], 
            weight_init=config["pretrained"],
            drop_rate=config["drop_rate"], 
            attn_drop_rate=config["attn_drop_rate"], 
            drop_path_rate=config["drop_path_rate"],
            final_actv=config["final_actv"],
            global_pool=config["spatial_pool"]
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
    model.transfer_learning = False
elif config["testonly"]:
    model = load(config["pretraineddir"])
else:
    # Transfer Learning model
    pretrained_dir = config["pretraineddir"]
    model = load(pretrained_dir)
    model.transfer_learning = True
    if config["reset_head"]: model.reset_classifier(num_classes=3)
    elif config["overhead"]: 
        model.head = nn.Sequential(
            model.head,
            nn.GELU(),
            nn.Linear(config["PVs"], 1024),
            nn.GELU(),
            nn.Linear(1024, config["num_classes"]),
            nn.Softmax(dim=-1)
        )
        for param in model.head[0].parameters():
            param.requires_grad = False
        model.overhead=True
    model.to(train_device)

model.train_embed = config["TrainEmbed"]
model.train_spatial = config["TrainSpatial"]
model.train_temporal = config["TrainTemporal"]
train_loader, _, _, inst_dist1 = build_loader(n_inst=config['n_inst'], seq_len=config["seq_len"], seq=config["seq_len"]>0, 
    bs=config["batch_size"], device=train_device, train_sz=0.99, fldir=config["fldir"], n_inst_percentage=config["n_inst_percentage"],seed=config["Seeds"])
test_loader, val_loader, _, inst_dist2 = build_loader(bs=1,train_sz=config["val_sz"], val_sz=0.99, fldir=config["test_dir"], n_inst=config["train_inst"], seq_len=config["seq_len"], seq=config["seq_len"]>0, seed=config['Seeds'])
inst_dist = {'Training': inst_dist1['Training'], 'Validation': inst_dist2['Validation'], 'Testing': inst_dist2['Training']}
dump_json(inst_dist, dest=os.path.join(config["eval_dir"], config["model_name"])+'/InstanceDistribution.json')
print("IntsanceDistribution saved")

# Loss, Optimizer, Dataloader

if config["loss"] is None:
    loss = nn.MSELoss()
    we_target = tensor([1/config["num_classes"] for _ in range(config["num_classes"])])
elif 'cross' in config["loss"]:
    train_dist = inst_dist["Training"]
    we_target = tensor([(1/config["num_classes"])*(1-(cl_inst/(sum(train_dist)/config["num_classes"])-1)) for cl_inst in train_dist]).to(train_device)
    loss = nn.CrossEntropyLoss(weight=we_target)
elif 'huber' in config["loss"]:
    loss = nn.HuberLoss()
    we_target = tensor([1/config["num_classes"] for _ in range(config["num_classes"])])

if config["opt"] == 'fastai':
    opt_func = OptimWrapper(opt=optim.Adam(model.parameters()))
    opt=None
else:    
    opt = build_adamw(model, config["base_lr"], we_decay=config["we_decay"])
    opt_func=None
# Learner
learner = Learner(config, model, loss, train_loader, val_loader, opt_func, opt=opt,
    min_delta=config["min_delta_loss"], min_val_loss=config["min_val_loss"], callback_dir=callback_dir, patience=config["patience"]
) 


# Training
mlflow.end_run()
mlflow.set_experiment(config["model_name"])
with mlflow.start_run(run_name=config["TrainType"], nested=True) as train_type:
    with mlflow.start_run(run_name="InstancesUsed"+str(config["n_inst_percentage"]), nested=True) as instdist:

        mlflow.log_params(config)

        #mlflow.set_tags(config['tags'])
        mlflow.log_metrics(dict(zip(['0', '1', '2'],we_target.cpu().numpy())))
        print("Loss weights to counter imbalanced data set: ", we_target)
        mlflow.log_artifact("config.yaml", artifact_path=config["model_name"])
        mlflow.log_artifact(os.path.join(config["eval_dir"], config["model_name"])+'/InstanceDistribution.json', artifact_path=config["model_name"])
        if config["testonly"]:
            learner.test(test_loader) if not config["pv_learning"] else learner.test_pv(test_loader)

        elif config["pv_learning"]:
            learner.loss_we = [1]
            if not config["fine_tune"]:
                learner.pv_learn(config["epochs_total"], params=config["PVs"], n_iter=train_loader.__len__(), loss_we=[1]) 
            else:
                learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=config["base_lr"])
            learner.test_pv(test_loader)
        elif config["fine_tune"]:
            learner.fine_tune(config["epochs_total"],config["epochs_froozen"], train_loader.__len__(), base_lr=config["base_lr"])
            learner.test(test_loader)
        else:
            learner.fit_one_cycle(epochs=config["epochs_total"], n_iter=train_loader.__len__(), lr_max=config["base_lr"])
            learner.test(test_loader)
    
mlflow.end_run()
save(learner.model, 'Models/vivit_pretrained_finetuned')
