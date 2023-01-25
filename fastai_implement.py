from lightningmodule.swin import SwinTransformer
from fastai.learner import Learner
from fastai.optimizer import Optimizer
from Data import build_loader
from torch import save, device, cuda

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
    'data_dir' : "C:/Users/MarkOne/data/regimeclassification" #"/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed"
}

train_device = device('cuda:0' if cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader, inst_dist = build_loader(n_inst=config['num_samples'], seq_len=config["seq_len"], 
    seq=False, bs=config["batch_size"], fldir=config["data_dir"], device=train_device
    )

model = SwinTransformer

fastai_learner = Learner(dls=train_loader, model=model)

