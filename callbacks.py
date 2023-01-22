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


def one_batch(xb, yb, learner):
    if not learner.cb.begin_batch(xb, yb): return
    loss = learner.cb.learn.loss_func(learner.cb.learn.model(xb), yb)
    if not learner.cb.after_loss(loss): return
    loss.backward()
    if learner.cb.after_backward(): learner.cb.learn.opt.step()
    if learner.cb.after_step(): learner.cb.learn.opt.zero_grad()

def one_val_batch(xb, yb, learner):
    if not learner.cb.begin_batch(xb, yb): return
    loss = learner.cb.learn.loss_func(learner.cb.learn.model(xb), yb)
    if not learner.cb.after_loss(loss): return

def train_batches(dl, learn):
    for xb, pv,  yb in dl:
        one_batch(xb, yb, learn)
        if learn.cb.do_stop(): return

def val_batches(dl, learn):
    for xb, pv,  yb in dl:
        one_val_batch(xb, yb, learn)
        if learn.cb.do_stop(): return

def fit(epochs, learn):
    if not learn.cb.begin_fit(): return
    learn.cb = learn.cb.cbs[0]
    for epoch in range(epochs):
        # HOTFTIX         
        if not learn.cb.begin_epoch(epoch): continue
        train_batches(learn.train_dl, learn)

        if learn.cb.begin_validate():
            with no_grad(): val_batches(learn.valid_dl, learn)
        if learn.cb.do_stop() or not learn.cb.after_epoch(): break
    learn.cb.after_fit()


