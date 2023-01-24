import torch
from lightningmodule.VSwinV2 import SwinTransformer3D
from lightningmodule.swinv2 import SwinTransformerV2
from logger import logging
logger = logging.getLogger('vswin_logger')
temp = torch.load("Dictionaries/swinv2-tiny-patch4-window7-224.bin")
model = SwinTransformer3D(logger=logger)
md = [model.state_dict()]