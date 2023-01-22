# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/13a_learner.ipynb.

# %% ../nbs/13a_learner.ipynb 2
from __future__ import annotations
from fastai.data.all import *
from fastai.optimizer import *
from fastai.callback.core import *
import pickle,threading
from collections.abc import MutableSequence


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
        
# %% auto 0
__all__ = ['replacing_yield', 'mk_metric', 'save_model', 'load_model', 'SkipToEpoch', 'Learner', 'before_batch_cb',
           'load_learner', 'Metric', 'AvgMetric', 'AvgLoss', 'AvgSmoothLoss', 'ValueMetric', 'Recorder', 'CastToTensor',
           'CancelBackwardException', 'CancelStepException', 'CancelFitException', 'CancelEpochException',
           'CancelTrainException', 'CancelValidException', 'CancelBatchException']

# %% ../nbs/13a_learner.ipynb 4
_all_ = ['CancelBackwardException', 'CancelStepException','CancelFitException','CancelEpochException',
         'CancelTrainException','CancelValidException','CancelBatchException']

# %% ../nbs/13a_learner.ipynb 10
defaults.lr = 1e-3

# %% ../nbs/13a_learner.ipynb 11
def replacing_yield(o, attr, val):
    "Context manager to temporarily replace an attribute"
    old = getattr(o,attr)
    try:     yield setattr(o,attr,val)
    finally: setattr(o,attr,old)

# %% ../nbs/13a_learner.ipynb 13
def mk_metric(m):
    "Convert `m` to an `AvgMetric`, unless it's already a `Metric`"
    if isinstance(m,type): m = m()
    return m if isinstance(m, Metric) else AvgMetric(m)

# %% ../nbs/13a_learner.ipynb 15
def save_model(file, model, opt, with_opt=True, pickle_protocol=2, **torch_save_kwargs):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if rank_distrib(): return # don't save if child proc
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, file, pickle_protocol=pickle_protocol, **torch_save_kwargs)

# %% ../nbs/13a_learner.ipynb 17
def load_model(file, model, opt, with_opt=True, device=None, strict=True, **torch_load_kwargs):
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device, **torch_load_kwargs)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if hasopt and with_opt:
        try: opt.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved filed doesn't contain an optimizer state.")

# %% ../nbs/13a_learner.ipynb 19
def _try_concat(o):
    try:    return torch.cat(o)
    except: return sum([L(o_[i,:] for i in range_of(o_)) for o_ in o], L())

# %% ../nbs/13a_learner.ipynb 20
_before_epoch = [event.before_fit, event.before_epoch]
_after_epoch  = [event.after_epoch, event.after_fit]

# %% ../nbs/13a_learner.ipynb 21
class _ConstantFunc():
    "Returns a function that returns `o`"
    def __init__(self, o): self.o = o
    def __call__(self, *args, **kwargs): return self.o

# %% ../nbs/13a_learner.ipynb 22
class SkipToEpoch(Callback):
    "Skip training up to `epoch`"
    order = 70
    
    def __init__(self, epoch:int):
        self._skip_to = epoch

    def before_epoch(self):
        if self.epoch < self._skip_to:
            raise CancelEpochException

# %% ../nbs/13a_learner.ipynb 24
_loop = ['Start Fit', 'before_fit', 'Start Epoch Loop', 'before_epoch', 'Start Train', 'before_train',
         'Start Batch Loop', 'before_batch', 'after_pred', 'after_loss', 'before_backward', 'before_step',
         'after_step', 'after_cancel_batch', 'after_batch','End Batch Loop','End Train',
         'after_cancel_train', 'after_train', 'Start Valid', 'before_validate','Start Batch Loop',
         '**CBs same as train batch**', 'End Batch Loop', 'End Valid', 'after_cancel_validate',
         'after_validate', 'End Epoch Loop', 'after_cancel_epoch', 'after_epoch', 'End Fit',
         'after_cancel_fit', 'after_fit']

# %% ../nbs/13a_learner.ipynb 25
class Learner(GetAttr):
    _default='model'
    def __init__(self,
        dls:DataLoaders, # `DataLoaders` containing fastai or PyTorch `DataLoader`s
        model:callable, # PyTorch model for training or inference
        loss_func:callable|None=None, # Loss function. Defaults to `dls` loss
        opt_func:Optimizer|OptimWrapper=Adam, # Optimization function for training
        lr:float|slice=defaults.lr, # Default learning rate
        splitter:callable=trainable_params, # Split model into parameter groups. Defaults to one parameter group
        cbs:Callback|MutableSequence|None=None, # `Callback`s to add to `Learner`
        metrics:callable|MutableSequence|None=None, # `Metric`s to calculate on validation set
        path:str|Path|None=None, # Parent directory to save, load, and export models. Defaults to `dls` `path`
        model_dir:str|Path='models', # Subdirectory to save and load models
        wd:float|int|None=None, # Default weight decay
        wd_bn_bias:bool=False, # Apply weight decay to normalization and bias parameters
        train_bn:bool=True, # Train frozen normalization layers
        moms:tuple=(0.95,0.85,0.95), # Default momentum for schedulers
        default_cbs:bool=True # Include default `Callback`s
    ):
        path = Path(path) if path is not None else getattr(dls, 'path', Path('.'))
        if loss_func is None:
            loss_func = getattr(dls.train_ds, 'loss_func', None)
            assert loss_func is not None, "Could not infer loss function from the data, please pass a loss function."
        self.dls,self.model = dls,model
        store_attr(but='dls,model,cbs')
        self.training,self.create_mbar,self.logger,self.opt,self.cbs = False,True,print,None,L()
        if default_cbs: self.add_cbs(L(defaults.callbacks))
        self.add_cbs(cbs)
        self.lock = threading.Lock()
        self("after_create")

    @property
    def metrics(self): return self._metrics
    @metrics.setter
    def metrics(self,v): self._metrics = L(v).map(mk_metric)

    def _grab_cbs(self, cb_cls): return L(cb for cb in self.cbs if isinstance(cb, cb_cls))

    def add_cbs(self, cbs):
        L(cbs).map(self.add_cb)
        return self

    def remove_cbs(self, cbs):
        L(cbs).map(self.remove_cb)
        return self

    def add_cb(self, cb):
        if isinstance(cb, type): cb = cb()
        cb.learn = self
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return self

    def remove_cb(self, cb):
        if isinstance(cb, type): self.remove_cbs(self._grab_cbs(cb))
        else:
            cb.learn = None
            if hasattr(self, cb.name): delattr(self, cb.name)
            if cb in self.cbs: self.cbs.remove(cb)
        return self

    @contextmanager
    def added_cbs(self, cbs):
        self.add_cbs(cbs)
        try: yield
        finally: self.remove_cbs(cbs)

    @contextmanager
    def removed_cbs(self, cbs):
        self.remove_cbs(cbs)
        try: yield self
        finally: self.add_cbs(cbs)

    def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
    def __call__(self, event_name): L(event_name).map(self._call_one)

    def _call_one(self, event_name):
        if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
        for cb in self.cbs.sorted('order'): cb(event_name)

    def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

    def create_opt(self):
        if isinstance(self.opt_func, partial):
            if 'lr' in self.opt_func.keywords:
                self.lr = self.opt_func.keywords['lr']
        if isinstance(self.opt_func, OptimWrapper):
            self.opt = self.opt_func
            self.opt.clear_state()
        else:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True

    def _split(self, b):
        i = getattr(self.dls, 'n_inp', 1 if len(b)==1 else len(b)-1)
        self.xb,self.yb = b[:i],b[i:]

    def _with_events(self, f, event_type, ex, final=noop):
        try: self(f'before_{event_type}');  f()
        except ex: self(f'after_cancel_{event_type}')
        self(f'after_{event_type}');  final()

    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _backward(self): self.loss_grad.backward()
    def _step(self): self.opt.step()

    def _do_grad_opt(self):
        self._with_events(self._backward, 'backward', CancelBackwardException)
        self._with_events(self._step, 'step', CancelStepException)
        self.opt.zero_grad()

    def _do_one_batch(self):
        self.pred = self.model(*self.xb)
        self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self._do_grad_opt()

    def _set_device(self, b):
        model_device = next(self.model.parameters()).device
        dls_device = getattr(self.dls, 'device', default_device())
        if model_device == dls_device: return to_device(b, dls_device)
        else: return to_device(b, model_device)

    def one_batch(self, i, b):
        self.iter = i
        b = self._set_device(b)
        self._split(b)
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)

    def _do_epoch_train(self):
        self.dl = self.dls.train
        self._with_events(self.all_batches, 'train', CancelTrainException)

    def _do_epoch_validate(self, ds_idx=1, dl=None):
        if dl is None: dl = self.dls[ds_idx]
        self.dl = dl
        with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)

    def _do_epoch(self):
        self._do_epoch_train()
        self._do_epoch_validate()

    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch=epoch
            self._with_events(self._do_epoch, 'epoch', CancelEpochException)

    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):
        if start_epoch != 0:
            cbs = L(cbs) + SkipToEpoch(start_epoch)
        with self.added_cbs(cbs):
            if reset_opt or not self.opt: self.create_opt()
            if wd is None: wd = self.wd
            if wd is not None: self.opt.set_hypers(wd=wd)
            self.opt.set_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch
            self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

    def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None
    def __enter__(self): self(_before_epoch); return self
    def __exit__(self, exc_type, exc_value, tb): self(_after_epoch)

    def validation_context(self, cbs=None, inner=False):
        cms = [self.no_logging(),self.no_mbar(), self.lock]
        if cbs: cms.append(self.added_cbs(cbs))
        if not inner: cms.append(self)
        return ContextManagers(cms)

    def validate(self, ds_idx=1, dl=None, cbs=None):
        if dl is None: dl = self.dls[ds_idx]
        with self.validation_context(cbs=cbs): self._do_epoch_validate(ds_idx, dl)
        return getattr(self, 'final_record', None)

    @delegates(GatherPredsCallback.__init__)
    def get_preds(self,
        ds_idx:int=1, # `DataLoader` to use for predictions if `dl` is None. 0: train. 1: valid
        dl=None, # `DataLoader` to use for predictions, defaults to `ds_idx=1` if None
        with_input:bool=False, # Return inputs with predictions
        with_decoded:bool=False, # Return decoded predictions
        with_loss:bool=False, # Return per item loss with predictions
        act=None, # Apply activation to predictions, defaults to `self.loss_func`'s activation
        inner:bool=False, # If False, create progress bar, show logger, use temporary `cbs`
        reorder:bool=True, # Reorder predictions on dataset indicies, if applicable
        cbs:Callback|MutableSequence|None=None, # Temporary `Callback`s to apply during prediction
        **kwargs
    )-> tuple:
        if dl is None: dl = self.dls[ds_idx].new(shuffle=False, drop_last=False)
        else:
            try: len(dl)
            except TypeError as e:
                raise TypeError(f"`dl` is {type(dl)} and doesn't have len(dl)")
        if isinstance(dl, DataLoader):
            if dl.drop_last: dl = dl.new(shuffle=False, drop_last=False)
        if reorder and hasattr(dl, 'get_idxs'):
            idxs = dl.get_idxs()
            dl = dl.new(get_idxs = _ConstantFunc(idxs))
        cb = GatherPredsCallback(with_input=with_input, with_loss=with_loss, **kwargs)
        ctx_mgrs = self.validation_context(cbs=L(cbs)+[cb], inner=inner)
        if with_loss: ctx_mgrs.append(self.loss_not_reduced())
        with ContextManagers(ctx_mgrs):
            self._do_epoch_validate(dl=dl)
            if act is None: act = getcallable(self.loss_func, 'activation')
            res = cb.all_tensors()
            pred_i = 1 if with_input else 0
            if res[pred_i] is not None:
                res[pred_i] = act(res[pred_i])
                if with_decoded: res.insert(pred_i+2, getcallable(self.loss_func, 'decodes')(res[pred_i]))
            if reorder and hasattr(dl, 'get_idxs'): res = nested_reorder(res, tensor(idxs).argsort())
            return tuple(res)
        self._end_cleanup()

    def predict(self, item, rm_type_tfms=None, with_input=False):
        dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        i = getattr(self.dls, 'n_inp', -1)
        inp = (inp,) if i==1 else tuplify(inp)
        dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
        dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
        res = dec_targ,dec_preds[0],preds[0]
        if with_input: res = (dec_inp,) + res
        return res

    def show_results(self, ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
        if dl is None: dl = self.dls[ds_idx].new(shuffle=shuffle)
        b = dl.one_batch()
        _,_,preds = self.get_preds(dl=[b], with_decoded=True)
        dl.show_results(b, preds, max_n=max_n, **kwargs)

    def show_training_loop(self):
        indent = 0
        for s in _loop:
            if s.startswith('Start'): print(f'{" "*indent}{s}'); indent += 2
            elif s.startswith('End'): indent -= 2; print(f'{" "*indent}{s}')
            else: print(f'{" "*indent} - {s:15}:', self.ordered_cbs(s))

    @contextmanager
    def no_logging(self): return replacing_yield(self, 'logger', noop)
    @contextmanager
    def no_mbar(self):    return replacing_yield(self, 'create_mbar', False)

    @contextmanager
    def loss_not_reduced(self):
        if hasattr(self.loss_func, 'reduction'): return replacing_yield(self.loss_func, 'reduction', 'none')
        else: return replacing_yield(self, 'loss_func', partial(self.loss_func, reduction='none'))
    
    def to_detach(self,b,cpu=True,gather=True):
        return self.dl.to_detach(b,cpu,gather) if hasattr(getattr(self,'dl',None),'to_detach') else to_detach(b,cpu,gather)
    
    def __getstate__(self): return {k:v for k,v in self.__dict__.items() if k!='lock'}
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

Learner.x,Learner.y = add_props(lambda i,x: detuplify((x.xb,x.yb)[i]))

# %% ../nbs/13a_learner.ipynb 26
add_docs(Learner, "Group together a `model`, some `dls` and a `loss_func` to handle training",
    add_cbs="Add `cbs` to the list of `Callback` and register `self` as their learner",
    add_cb="Add `cb` to the list of `Callback` and register `self` as their learner",
    remove_cbs="Remove `cbs` from the list of `Callback` and deregister `self` as their learner",
    remove_cb="Add `cb` from the list of `Callback` and deregister `self` as their learner",
    added_cbs="Context manage that temporarily adds `cbs`",
    removed_cbs="Context manage that temporarily removes `cbs`",
    ordered_cbs="List of `Callback`s, in order, for an `event` in the training loop",
    create_opt="Create an optimizer with default hyper-parameters",
    one_batch="Train or evaluate `self.model` on batch `(xb,yb)`",
    all_batches="Train or evaluate `self.model` on all the batches of `self.dl`",
    fit="Fit `self.model` for `n_epoch` using `cbs`. Optionally `reset_opt`.",
    validate="Validate on `dl` with potential new `cbs`.",
    get_preds="Get the predictions and targets on the `ds_idx`-th dbunchset or `dl`, optionally `with_input` and `with_loss`",
    predict="Prediction on `item`, fully decoded, loss function decoded and probabilities",
    validation_context="A `ContextManagers` suitable for validation, with optional `cbs`",
    show_results="Show some predictions on `ds_idx`-th dataset or `dl`",
    show_training_loop="Show each step in the training loop",
    no_logging="Context manager to temporarily remove `logger`",
    no_mbar="Context manager to temporarily prevent the master progress bar from being created",
    loss_not_reduced="A context manager to evaluate `loss_func` with reduction set to none.",
    to_detach="Calls `to_detach` if `self.dl` provides a `.to_detach` function otherwise calls global `to_detach`",
    __call__="Call `event_name` for all `Callback`s in `self.cbs`"
)

# %% ../nbs/13a_learner.ipynb 33
if not hasattr(defaults, 'callbacks'): defaults.callbacks = [TrainEvalCallback]

# %% ../nbs/13a_learner.ipynb 88
def _before_batch_cb(f, self):
    xb,yb = f(self, self.xb, self.yb)
    self.learn.xb,self.learn.yb = xb,yb

# %% ../nbs/13a_learner.ipynb 89
def before_batch_cb(f):
    "Shortcut for creating a Callback on the `before_batch` event, which takes and returns `xb,yb`"
    return Callback(before_batch=partial(_before_batch_cb, f))

# %% ../nbs/13a_learner.ipynb 96
@patch
@delegates(save_model)
def save(self:Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    save_model(file, self.model, getattr(self,'opt',None), **kwargs)
    return file

# %% ../nbs/13a_learner.ipynb 98
@patch
@delegates(load_model)
def load(self:Learner, file, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`"
    if device is None and hasattr(self.dls, 'device'): device = self.dls.device
    if self.opt is None: self.create_opt()
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    load_model(file, self.model, self.opt, device=device, **kwargs)
    nested_attr(self, "accelerator.wait_for_everyone", noop)()
    return self

# %% ../nbs/13a_learner.ipynb 102
@patch
def export(self:Learner, fname='export.pkl', pickle_module=pickle, pickle_protocol=2):
    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib(): return # don't export if child proc
    self._end_cleanup()
    old_dbunch = self.dls
    self.dls = self.dls.new_empty()
    state = self.opt.state_dict() if self.opt is not None else None
    self.opt = None
    with warnings.catch_warnings():
        #To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(self, self.path/fname, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
    self.create_opt()
    if state is not None: self.opt.load_state_dict(state)
    self.dls = old_dbunch

# %% ../nbs/13a_learner.ipynb 104
def load_learner(fname, cpu=True, pickle_module=pickle):
    "Load a `Learner` object in `fname`, by default putting it on the `cpu`"
    distrib_barrier()
    map_loc = 'cpu' if cpu else default_device()
    try: res = torch.load(fname, map_location=map_loc, pickle_module=pickle_module)
    except AttributeError as e: 
        e.args = [f"Custom classes or functions exported with your `Learner` not available in namespace.\Re-declare/import before loading:\n\t{e.args[0]}"]
        raise
    if cpu: 
        res.dls.cpu()
        if hasattr(res, 'mixed_precision'): res = res.to_fp32()
        elif hasattr(res, 'non_native_mixed_precision'): res = res.to_non_native_fp32()
    return res

# %% ../nbs/13a_learner.ipynb 111
@docs
class Metric():
    "Blueprint for defining a metric"
    def reset(self): pass
    def accumulate(self, learn): pass
    @property
    def value(self): raise NotImplementedError

    @property
    def name(self): return class2attr(self, 'Metric')

    _docs = dict(
        reset="Reset inner state to prepare for new computation",
        name="Name of the `Metric`, camel-cased and with Metric removed",
        accumulate="Use `learn` to update the state with new results",
        value="The value of the metric")

# %% ../nbs/13a_learner.ipynb 118
class AvgMetric(Metric):
    "Average the values of `func` taking into account potential different batch sizes"
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__

# %% ../nbs/13a_learner.ipynb 122
class AvgLoss(Metric):
    "Average the losses taking into account potential different batch sizes"
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(learn.loss.mean())*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return "loss"

# %% ../nbs/13a_learner.ipynb 126
class AvgSmoothLoss(Metric):
    "Smooth average of the losses (exponentially weighted with `beta`)"
    def __init__(self, beta=0.98): self.beta = beta
    def reset(self):               self.count,self.val = 0,tensor(0.)
    def accumulate(self, learn):
        self.count += 1
        self.val = torch.lerp(to_detach(learn.loss.mean()), self.val, self.beta)
    @property
    def value(self): return self.val/(1-self.beta**self.count)

# %% ../nbs/13a_learner.ipynb 129
class ValueMetric(Metric):
    "Use to include a pre-calculated metric value (for instance calculated in a `Callback`) and returned by `func`"
    def __init__(self, func, metric_name=None): store_attr('func, metric_name')

    @property
    def value(self): return self.func()

    @property
    def name(self): return self.metric_name if self.metric_name else self.func.__name__

# %% ../nbs/13a_learner.ipynb 133
from fastprogress.fastprogress import format_time

# %% ../nbs/13a_learner.ipynb 134
def _maybe_item(t):
    t = t.value
    try: return t.item()
    except: return t

# %% ../nbs/13a_learner.ipynb 135
class Recorder(Callback):
    "Callback that registers statistics (lr, loss and metrics) during training"
    _stateattrs=('lrs','iters','losses','values')
    remove_on_fetch,order = True,50

    def __init__(self, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98):
        store_attr('add_time,train_metrics,valid_metrics')
        self.loss,self.smooth_loss = AvgLoss(),AvgSmoothLoss(beta=beta)

    def before_fit(self):
        "Prepare state for training"
        self.lrs,self.iters,self.losses,self.values = [],[],[],[]
        names = self.metrics.attrgot('name')
        if self.train_metrics and self.valid_metrics:
            names = L('loss') + names
            names = names.map('train_{}') + names.map('valid_{}')
        elif self.valid_metrics: names = L('train_loss', 'valid_loss') + names
        else: names = L('train_loss') + names
        if self.add_time: names.append('time')
        self.metric_names = 'epoch'+names
        self.smooth_loss.reset()

    def after_batch(self):
        "Update all metrics and records lr and smooth loss in training"
        if len(self.yb) == 0: return
        mets = self._train_mets if self.training else self._valid_mets
        for met in mets: met.accumulate(self.learn)
        if not self.training: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.smooth_loss.value)
        self.learn.smooth_loss = self.smooth_loss.value

    def before_epoch(self):
        "Set timer if `self.add_time=True`"
        self.cancel_train,self.cancel_valid = False,False
        if self.add_time: self.start_epoch = time.time()
        self.log = L(getattr(self, 'epoch', 0))

    def before_train   (self): self._train_mets[1:].map(Self.reset())
    def before_validate(self): self._valid_mets.map(Self.reset())
    def after_train   (self): self.log += self._train_mets.map(_maybe_item)
    def after_validate(self): self.log += self._valid_mets.map(_maybe_item)
    def after_cancel_train(self):    self.cancel_train = True
    def after_cancel_validate(self): self.cancel_valid = True

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learn.final_record = self.log[1:].copy()
        self.values.append(self.learn.final_record)
        if self.add_time: self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.smooth_loss.count)

    @property
    def _train_mets(self):
        if getattr(self, 'cancel_train', False): return L()
        return L(self.smooth_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_mets(self):
        if getattr(self, 'cancel_valid', False): return L()
        return (L(self.loss) + self.metrics if self.valid_metrics else L())

    def plot_loss(self, skip_start=5, with_valid=True):
        plt.plot(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
        if with_valid:
            idx = (np.array(self.iters)<skip_start).sum()
            valid_col = self.metric_names.index('valid_loss') - 1 
            plt.plot(self.iters[idx:], L(self.values[idx:]).itemgot(valid_col), label='valid')
            plt.legend()

# %% ../nbs/13a_learner.ipynb 136
add_docs(Recorder,
         before_train = "Reset loss and metrics state",
         after_train = "Log loss and metric values on the training set (if `self.training_metrics=True`)",
         before_validate = "Reset loss and metrics state",
         after_validate = "Log loss and metric values on the validation set",
         after_cancel_train = "Ignore training metrics for this epoch",
         after_cancel_validate = "Ignore validation metrics for this epoch",
         plot_loss = "Plot the losses from `skip_start` and onward")

if Recorder not in defaults.callbacks: defaults.callbacks.append(Recorder)

# %% ../nbs/13a_learner.ipynb 152
def _cast_tensor(x): 
    if isinstance(x, tuple): return tuple(_cast_tensor(x_) for x_ in x)
    else: return cast(x, Tensor) if isinstance(x,torch.Tensor) else x

# %% ../nbs/13a_learner.ipynb 153
class CastToTensor(Callback):
    "Cast Subclassed Tensors to `Tensor`"
    order=9 # Right before MixedPrecision

    def before_batch(self):
        self.learn.xb,self.learn.yb = _cast_tensor(self.learn.xb),_cast_tensor(self.learn.yb)

# %% ../nbs/13a_learner.ipynb 155
if CastToTensor not in defaults.callbacks: defaults.callbacks.append(CastToTensor)

# %% ../nbs/13a_learner.ipynb 185
@patch
def freeze_to(self:Learner, n):
    if self.opt is None: self.create_opt()
    self.opt.freeze_to(n)
    self.opt.clear_state()

@patch
def freeze(self:Learner): self.freeze_to(-1)

@patch
def unfreeze(self:Learner): self.freeze_to(0)

add_docs(Learner,
         freeze_to="Freeze parameter groups up to `n`",
         freeze="Freeze up to last parameter group",
         unfreeze="Unfreeze the entire model")

# %% ../nbs/13a_learner.ipynb 189
@patch
def tta(self:Learner, ds_idx=1, dl=None, n=4, item_tfms=None, batch_tfms=None, beta=0.25, use_max=False):
    "Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation"
    if dl is None: dl = self.dls[ds_idx].new(shuffled=False, drop_last=False)
    if item_tfms is not None or batch_tfms is not None: dl = dl.new(after_item=item_tfms, after_batch=batch_tfms)
    try:
        self(_before_epoch)
        with dl.dataset.set_split_idx(0), self.no_mbar():
            if hasattr(self,'progress'): self.progress.mbar = master_bar(list(range(n)))
            aug_preds = []
            for i in self.progress.mbar if hasattr(self,'progress') else range(n):
                self.epoch = i #To keep track of progress on mbar since the progress callback will use self.epoch
                aug_preds.append(self.get_preds(dl=dl, inner=True)[0][None])
        aug_preds = torch.cat(aug_preds)
        aug_preds = aug_preds.max(0)[0] if use_max else aug_preds.mean(0)
        self.epoch = n
        with dl.dataset.set_split_idx(1): preds,targs = self.get_preds(dl=dl, inner=True)
    finally: self(event.after_fit)

    if use_max: return torch.stack([preds, aug_preds], 0).max(0)[0],targs
    preds = (aug_preds,preds) if beta is None else torch.lerp(aug_preds, preds, beta)
    return preds,targs