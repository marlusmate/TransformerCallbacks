# https://towardsdatascience.com/callbacks-in-neural-networks-b0b006df7626
from torch import no_grad, tensor
import mlflow

class Callback():
    def begin_fit(self):        return True

    def after_fit(self): 
        return True

    def begin_epoch(self, epoch):
        self.epoch = epoch
        return True

    def begin_validate(self): 
        return True

    def after_epoch(self): 
        return True

    def begin_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        return True

    def after_loss(self, loss):
        self.loss = loss
        return True

    def after_backward(self): 
        return True

    def after_step(self): 
        return True


class BatchCounter(Callback):
    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.batch_counter = 1
        return True

    def after_step(self):
        self.batch_counter += 1
        if self.batch_counter % 200 == 0: print(f'Batch {self.batch_counter} completed')
        return True

    def do_stop(self):
        # Abbruchkriteria?
        return False


class ParamScheduler(Callback):
    "Schedule hyper-parameters according to `scheds`"
    order,run_valid = 60,False

    def __init__(self, scheds, n_iter, n_epoch): 
        self.scheds = scheds
        self.n_iter = n_iter
        self.n_epoch = n_epoch

    def before_fit(self): 
        self.hps = {p:[] for p in self.scheds.keys()}
        self.epoch = 0
        self.loss = tensor(0.)
        self.train_iter = 0
        self.pct_train = 0.

    def before_batch(self): 
        self._update_val(self.pct_train)

    def _update_val(self, pct):
        for n,f in self.scheds.items(): self.learn.opt.set_hyper(n, f(pct))

    def after_batch(self):
        for p in self.scheds.keys(): self.hps[p].append(self.learn.opt.hypers[-1][p])
        if self.learn.training:
            self.pct_train += 1./(self.n_iter*self.n_epoch) 
            self.train_iter += 1 

    def after_fit(self):
        if hasattr(self.learn, 'recorder') and hasattr(self, 'hps'): self.recorder.hps = self.hps

    _docs = {"before_fit": "Initialize container for hyper-parameters",
             "before_batch": "Set the proper hyper-parameters in the optimizer",
             "after_batch": "Record hyper-parameters of this batch",
             "after_fit": "Save the hyper-parameters in the recorder if there is one"}

        #self.epoch_loss_pred += self.loss / self.n_iter


class CallbackHandler():
    def __init__(self, cbs=None):
        self.cbs = cbs if cbs else []

    def set_learn(self, learn):
        self.learn = learn
        for cb in self.cbs:
            cb.learn = self.learn

    def begin_fit(self):
        self.in_train = True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.begin_fit()
        return res




