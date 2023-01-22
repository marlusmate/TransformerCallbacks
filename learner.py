from torch import no_grad
from tqdm import tqdm

class Learner:
    def __init__(self, model, loss_func, opt, sched, train_dl, valid_dl, cb):
        self.model = model
        self.loss_func = loss_func
        self.opt = opt
        self.sched = sched
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.cb = cb
        self.cb.set_learn(self)

    def one_batch(self, xb, yb):
        if not self.cb.begin_batch(xb, yb): return
        loss = self.cb.learn.loss_func(self.cb.learn.model(xb), yb)
        if not self.cb.after_loss(loss): return
        loss.backward()
        if self.cb.after_backward(): self.cb.learn.opt.step()
        if self.cb.after_step(): self.cb.learn.opt.zero_grad()

    def one_val_batch(self, xb, yb):
        if not self.cb.begin_batch(xb, yb): return
        loss = self.cb.learn.loss_func(self.cb.learn.model(xb), yb)
        if not self.cb.after_loss(loss): return

    def train_batches(self):
        for xb, pv,  yb in tqdm(self.train_dl):
            self.one_batch(xb, yb)
            if self.cb.do_stop(): return

    def val_batches(self):
        for xb, pv,  yb in self.valid_dl:
            self.one_val_batch(xb, yb)
            if self.cb.do_stop(): return

    def fit(self, epochs):
        if not self.cb.begin_fit(): return
        self.cb = self.cb.cbs[0]
        for epoch in range(epochs):
            # HOTFTIX         
            if not self.cb.begin_epoch(epoch): continue
            self.train_batches()

            if self.cb.begin_validate():
                with no_grad(): self.val_batches()
            if self.cb.do_stop() or not self.cb.after_epoch(): break
        self.cb.after_fit()

    #def fine_tune(epochs, self, freeze_epochs, base_lr, lr_mult):