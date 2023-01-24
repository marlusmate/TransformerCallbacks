from torch import nn
from lightningmodule.VSwinV2 import SwinTransformer3D

class vswin_module(SwinTransformer3D):
    def __init__(self, 
        train_epoch=None,
        pretrain_epoch=1,
        tune_epoch=1,
        melt_epoch=None,
        **kwargs):
        super().__init__()
        self.train_epoch = train_epoch
        self.pretrain_epoch = pretrain_epoch
        self.tune_epoch = tune_epoch
        self.melt_epoch = melt_epoch
        self.pretrain_head = nn.Sequential([nn.Linear(self.num_features, 3), nn.ReLU()])
        self.tune_head = nn.Linear(self.num_features, 3)

        