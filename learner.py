from torch import no_grad
from tqdm import tqdm
from learner_utils import combine_scheds, combined_cos
from callbacks import ParamScheduler
from torch import nn, optim
from fastai.optimizer import OptimWrapper
from fastcore.foundation import L
import numpy as np

def norm_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, nn.LayerNorm): return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res

class Learner:
    def __init__(self, model, loss_func, sched, train_dl, valid_dl, opt_func):
        self.model = model
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt = None
        self.sched = sched
        self.dls_train = train_dl
        self.dls_valid = valid_dl
        self.wd_bn_bias=False
        self.train_bn =True
        #self.cb = cb
        #self.cb.set_learn(self)

    def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

    def create_opt(self):
        #if isinstance(self.opt_func, partial):
            #if 'lr' in self.opt_func.keywords:
                #self.lr = self.opt_func.keywords['lr']
        if isinstance(self.opt_func, OptimWrapper):
            self.opt = self.opt_func
            self.opt.clear_state()
        else:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True

    def _freeze_stages(self):
        if self.model.frozen_stages >= 0:
            self.model.patch_embed.eval()
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False

        if self.model.frozen_stages >= 1:
            self.model.pos_drop.eval()
            for i in range(0, self.model.frozen_stages):
                m = self.model.blocks[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False  

    def _unfreeze_stages(self):
        if self.model.frozen_stages >= 0:
            self.model.patch_embed.eval()
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True

        if self.model.frozen_stages >= 1:
            self.model.pos_drop.eval()
            for i in range(0, self.model.frozen_stages):
                m = self.model.blocks[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = True

    def _unfreeze_block(self, i):
        if i == 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = True

        if i >= 1:
            self.pos_drop.eval()
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False        

    def metrics(self):
        acc = (self.pred.argmax(dim=1) == self.yb).float().mean()
        self.epoch_accuracy += acc / self.n_iter
        self.epoch_loss += self.loss / self.n_iter

    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _backward(self): self.loss_grad.backward()
    def _step(self): self.opt.step()

    def _do_grad_opt(self):
        self._backward()
        self._step()
        self.opt.zero_grad()

    def _do_one_batch(self):
        self.pred = self.model(self.xb)
        #self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, self.yb)
            self.loss = self.loss_grad.clone()

        #self('after_loss')
        print('Batch Loss: ', self.loss)
        if not self.training: self.metrics()
        if not self.training or not len(self.yb): return
        self._do_grad_opt()


    def one_batch(self, i, data):
        self.iter = i,
        self.xb= data[0]
        self.yb= data[1]
        self.cbs.before_batch()
        self._do_one_batch()
        self.cbs.after_batch()

    def _do_epoch_train(self):
        print("Train Epoch:")
        self.dl = self.dls_train
        self.training = True        
        self.all_batches()

    def _do_epoch(self):
        self.epoch_accuracy, self.epoch_loss = 0.,0.        
        self._do_epoch_train()
        self._do_epoch_validate(dl=self.dls_valid)
        print(" # Epoch Loss: ", self.epoch_loss, "\n # Epoch Accuracy: ", self.epoch_accuracy)

    def _do_epoch_validate(self, ds_idx=1, dl=None):
        print("Val Epoch:")
        if dl is None: dl = self.dls[ds_idx]
        self.dl = dl
        self.training = False
        with no_grad(): self.all_batches()

    def _do_fit(self):
        self.cbs.before_fit()
        for epoch in range(self.epochs):
            self.epoch = epoch
            self._do_epoch()
        self.cbs.after_fit()

    def fit(self, epochs, cbs):
        cbs.learn = self
        self.cbs = cbs
        #self.opt 
        self.epochs =epochs
        self._do_fit()
        

    def fit_one_cycle(self, epochs, n_iter, lr_max=None, div=25., div_final=1e5, pct_start=.25, moms=(0.95,0.85,0.95)):
        #if not self.cb.begin_fit(): return
        #self.opt.defaults['lr'] = self.lr_max if lr_max is None else lr_max
        if self.opt is None: self.create_opt()
        self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
        lr_max = np.array([h['lr'] for h in self.opt.hypers])
        scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
              'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))
              }
        cbs = ParamScheduler(scheds, n_iter, epochs)
        self.fit(epochs, cbs)

        

    def fine_tune(self, epochs, freeze_epochs, n_iter, base_lr=2e-3, lr_mult=100):
        #if not self.cb.begin_fit(): return
        #self.cb = self.cb.cbs[0]
        #self.cb = self.cb.cbs[0]
        self._freeze_stages()
        self.fit_one_cycle(freeze_epochs, n_iter, base_lr)
        base_lr /= 2
        self._unfreeze_block(-2)
        self.fit_one_cycle(epochs-freeze_epochs, n_iter, slice(base_lr/lr_mult, base_lr), pct_start=0.3, div=5)



