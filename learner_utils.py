from fastcore.foundation import L
from fastcore.basics import store_attr
from fastai.callback.core import Callback
import torch
from torch import tensor
import math
import numpy as np
from learner_fastai import Learner
import json
from Modelcode.swin import SwinTransformer
from Modelcode.vit import VisionTransformer
from Modelcode.vswin import SwinTransformer3D
from Modelcode.vswin_multimodal import SwinTransformer3D as MSwinTransformer3D
from Modelcode.vivit import VisionTransformer3D

"""
def buid_model(name, img_size=224):
    if 'vit-tiny-patch16-224' in name:
        model = VisionTransformer(
            
        )
    elif 'swin-tiny-patch4-window7-224' in name:
    elif 'vivit-tiny-patch16-224' in name:
    elif 'vswin-tiny-patch4-window_7_224' in name:
    return model
"""

def dump_json(fn, dest):
    with open(dest, 'w+') as f:
        json.dump(fn, f, indent=2)


class _Annealer:
    def __init__(self, f, start, end): store_attr('f,start,end')
    def __call__(self, pos): return self.f(self.start, self.end, pos)

def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

def SchedCos(start, end): return _Annealer(sched_cos, start, end)

def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.
    pcts = tensor([0] + L(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    pct_lim = len(pcts) - 2
    def _inner(pos):
        idx = min((pos >= pcts).nonzero().max(), pct_lim)
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos.item())
    return _inner

def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds([pct,1-pct], [SchedCos(start, middle), SchedCos(middle, end)])

class ParamScheduler(Callback):
    "Schedule hyper-parameters according to `scheds`"
    order,run_valid = 60,False

    def __init__(self, scheds): self.scheds = scheds
    def before_fit(self): self.hps = {p:[] for p in self.scheds.keys()}
    def before_batch(self): self._update_val(self.pct_train)

    def _update_val(self, pct):
        for n,f in self.scheds.items(): self.opt.set_hyper(n, f(pct))

    def after_batch(self):
        for p in self.scheds.keys(): self.hps[p].append(self.opt.hypers[-1][p])

    def after_fit(self):
        if hasattr(self.learn, 'recorder') and hasattr(self, 'hps'): self.recorder.hps = self.hps
