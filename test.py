import torch
from lightningmodule.VSwinV2 import SwinTransformer3D
from logger import logging
logger = logging.getLogger('vswin_logger')
temp = torch.load("Dictionaries/pytorch_model.bin")
model = SwinTransformer3D(logger=logger)
md = [model.state_dict()]