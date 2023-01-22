# https://towardsdatascience.com/callbacks-in-neural-networks-b0b006df7626
from torch import no_grad
import mlflow

class Callback():
    def begin_fit(self):
        return True

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




