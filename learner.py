from torch import no_grad, save, tensor, load
from tqdm import tqdm
from learner_utils import combine_scheds, combined_cos, dump_json
from callbacks import ParamScheduler
from torch import nn, optim, tensor, mean
from fastai.optimizer import OptimWrapper
from fastcore.foundation import L
import numpy as np
import mlflow
from functools import partial

def norm_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, nn.LayerNorm): return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res

class Learner:
    def __init__(self, config, model, loss_func, train_dl, valid_dl, opt_func, patience=1, min_delta=.000, min_val_loss=0.0005, callback_dir="Models"):
        self.model = model
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt = None
        self.dls_train = train_dl
        self.dls_valid = valid_dl
        self.wd_bn_bias=False
        self.train_bn =True
        self.config = config
        self.pv_learning = False
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.last_loss = 0
        self.last_acc = 0
        self.min_validation_loss = min_val_loss
        self.callback_dir = callback_dir
        self.callback_set = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print("Early stopping counter set to: ", self.counter)
            if self.epoch_val_accuracy > self.last_acc:
                save(self.model, self.callback_dir+"_callback")
                self.last_acc = self.epoch_val_accuracy
            self.callback_set = True
            self.last_loss = validation_loss
            print("Callback Model gespeichert, loss: ", validation_loss)
        elif abs(validation_loss - self.last_loss) < self.min_delta:
            self.counter += 1
            self.last_loss = validation_loss
            print("Early stopping counter set to: ", self.counter)
            if self.counter >= self.patience:
                return True
        elif validation_loss > 1.1*self.last_loss:
            self.counter += 1
            self.last_loss = validation_loss
            print("Early stopping counter set to: ", self.counter)
            if self.counter >= self.patience:
                return True
        
        return False

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

    def _freeze_stages(self):
        self.model._freeze_stages()

    def _unfreeze_stages(self):
        self.model._unfreeze_stages()       

    def metrics(self):
        acc = (self.pred.argmax(dim=1) == self.yb).float().mean() if not self.pv_learning else 0
        if self.training:            
            self.epoch_accuracy += acc / self.n_iter
            self.epoch_loss += self.loss / self.n_iter            
            return
        else:
            self.epoch_val_accuracy += acc / self.n_iter
            self.epoch_val_loss += self.loss / self.n_iter
        if self.testing:
            self.preds.append(self.pred.cpu().numpy())
            self.predscl.append(self.pred.argmax(dim=1).cpu().numpy())
            self.labels.append(self.yb.cpu().numpy())


    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _backward(self): self.loss_grad.backward()
    def _step(self): self.opt.step()

    def _do_grad_opt(self):
        self._backward()
        self._step()
        self.opt.zero_grad()

    def _do_loss(self):
        self.loss_grad = 0
        if self.training and not self.pv_learning:
            self.loss_grad = self.loss_func(self.pred, self.yb)
            self.loss = self.loss_grad.clone()
        elif self.pv_learning:
            for i in range(self.params):
                self.loss_grad += self.loss_func(self.pred[:,i], self.yb[:,i]) * self.loss_we[i]
            self.loss = self.loss_grad.clone()
                

    def _do_one_batch(self):
        self.pred = self.model(self.xb)
        self._do_loss()
        self.metrics()
        if not self.training or not len(self.yb): return
        self._do_grad_opt()

    def one_batch(self, i, data):
        self.iter = i,
        self.xb= data[0]
        self.yb= data[2] #.mean(dim=1)
        self.cbs.before_batch()
        self._do_one_batch()
        self.cbs.after_batch()

    def _do_epoch_train(self):
        print("Train Epoch:")
        self.dl = self.dls_train
        self.training = True       
        self.all_batches()

    def _do_epoch(self):
        self.testing = False
        self.epoch_accuracy, self.epoch_loss = 0.,0.
        self.epoch_val_accuracy, self.epoch_val_loss = 0.,0.        
        self._do_epoch_train()
        self._do_epoch_validate(dl=self.dls_valid)
        print(
                    f"Epoch : {self.epoch+1} - loss : {self.epoch_loss:.4f} - acc: {self.epoch_accuracy:.4f} - val_loss : {self.epoch_val_loss:.4f} - val_acc: {self.epoch_val_accuracy:.4f}\n"
                    )
        mlflow.log_metric("loss_train_epoch", self.epoch_loss, step=self.cbs.epoch)
        mlflow.log_metric("acc_train_epoch", self.epoch_accuracy, step=self.cbs.epoch)
        mlflow.log_metric("loss_val_epoch", self.epoch_val_loss, step=self.cbs.epoch)
        mlflow.log_metric("acc_val_epoch", self.epoch_val_accuracy, step=self.cbs.epoch)
        self.cbs.epoch += 1

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
            if self.early_stop(self.epoch_val_loss):
                break

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
        self._freeze_stages()
        self.fit_one_cycle(freeze_epochs, n_iter, slice(base_lr), pct_start=0.99)
        base_lr /= 100
        self._unfreeze_stages()
        self.fit_one_cycle(epochs-freeze_epochs, n_iter, slice(base_lr/lr_mult, base_lr), pct_start=0.3, div=5)

    def pv_learn(self, epochs, params, n_iter, loss_we=[0.6, 0.4], base_lr=2e-3):
        assert np.array(loss_we).sum() == 1, 'Loss Weights for PV Losses must add up to 1'
        assert params == len(loss_we), 'every process variable must be one loss weight assigned'
        self.pv_learning = True
        self.params = params
        self.loss_we = loss_we
        self.fit_one_cycle(epochs, n_iter, lr_max=base_lr)

    def test(self, dls_test):
        self.preds, self.predscl, self.labels = [], [], []
        self.testing = True
        if self.callback_set:
            self.model = load(self.callback_dir+"_callback")
            print("Model Callback loaded")
        self._do_epoch_validate(dl=dls_test)
        self.seed = self.config["tags"]["Seeds"][0]
        
        import matplotlib.pyplot as plt
        from seaborn import heatmap
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
        from torch.nn.functional import one_hot
        import os

        # Confusion Matrix
        cf = confusion_matrix(np.asarray(self.labels), np.asarray(self.predscl))
        figcf, ax_cf = plt.subplots()
        ax_cf.set_title("Confusion Matrix " + self.config["model_name"])
        ax_cf = heatmap(cf, annot=True, cmap="Blues", yticklabels=3, xticklabels=3)
        ax_cf.set_ylabel("True Class")
        ax_cf.set_xlabel("Predicted Class ")
        figcf.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"],"ConfusionMatrix_"+str(self.seed)))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "ConfusionMatrix_"+str(self.seed)+".png"), artifact_path=self.config["model_name"])
        
        # Calc ROC, AUC
        self.labelsoh = one_hot(tensor(np.array(self.labels)), num_classes=3).squeeze(1)
        self.preds = tensor(self.preds).squeeze(1)
        fpr_0, tpr_0, thresholds_0 = roc_curve(self.labelsoh[:,0].numpy(), self.preds[:,0].numpy())
        fpr_1, tpr_1, thresholds_1 = roc_curve(self.labelsoh[:,1], self.preds[:,1])
        fpr_2, tpr_2, thresholds_2 = roc_curve(self.labelsoh[:,2], self.preds[:,2])
        auc_0 = roc_auc_score(self.labelsoh[:,0], self.preds[:,0])
        auc_1 = roc_auc_score(self.labelsoh[:,1], self.preds[:,1])
        auc_2 = roc_auc_score(self.labelsoh[:,2], self.preds[:,2])
        auc = mean(tensor((auc_0, auc_1, auc_2))).numpy()

        # Plot ROC
        figroc, axroc = plt.subplots()
        axroc.plot(fpr_0, tpr_0, label="flooded")
        axroc.plot(fpr_1, tpr_1, label="loaded")
        axroc.plot(fpr_2, tpr_2, label="dispersed")
        axroc.plot([0,1],[0,1], 'k--')
        axroc.set(title="ROC - " + self.config["model_name"], xlabel="False Positive Rate", ylabel="True Negative Rate")
        axroc.legend()
        figroc.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "ROC_"+self.config["model_name"]))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], "ROC_"+self.config["model_name"]+".png"), artifact_path=self.config["model_name"])

        # F1 Score
        self.predsoh=one_hot(tensor(self.predscl), num_classes=3).squeeze(1)
        f1 = f1_score(tensor(self.labels).squeeze(1).numpy(), tensor(self.predscl).squeeze(1).numpy(), average='weighted')
        f1_0 = f1_score(self.labelsoh[:,0], self.predsoh[:,0])
        f1_1 = f1_score(self.labelsoh[:,1], self.predsoh[:,1])
        f1_2 = f1_score(self.labelsoh[:,2], self.predsoh[:,2])


        # Create Metrics Sheet
        metrics = {'F1-Score': f1, 'AUC': float(auc), 
                'Flooded': {"F1-Score": f1_0, "AUC": auc_0},
                'Loaded': {"F1-Score": f1_1, "AUC": auc_1},
                'Dispersed':{"F1-Score": f1_2, "AUC": auc_2}}

        dump_json(metrics, os.path.join(self.config["eval_dir"], self.config["model_name"], "Metrics_"+self.config["model_name"]+".json"))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], f"Metrics_"+self.config["model_name"]+".json"), artifact_path=self.config["model_name"])


    def save_model(self):
        save(self.model, self.callback_dir+"/Model")
        print("Model abgespeichert")

    def test_pv(self, dls_test):
        self.preds, self.predscl, self.labels = [], [], []
        self.testing = True
        self._do_epoch_validate(dl=dls_test)
        self.preds = np.array(self.preds)
        self.labels = np.array(self.labels)
        self.seed = self.config["tags"]["Seeds"][0]
        # PVP
        #mae
        from sklearn.metrics import mean_absolute_error as mae
        import matplotlib.pyplot as plt
        import os

        pred_rpm = self.preds[:,0]
        pred_gfl = self.preds[:,1]
        #pred_temp = self.preds[:,2]
        label_rpm = self.labels[:,0]
        label_gfl = self.labels[:,1]
        #label_temp = self.labels[:,2] 
        mae_rpm = mae(label_rpm,pred_rpm)
        mae_gfl = mae(label_gfl,pred_gfl)
        #mae_tem = mae(label_temp,pred_temp)
        maes = {'rpm': mae_rpm, 'gfl': mae_gfl}#, 'temp': mae_tem}

        f1 = plt.bar(list(maes.keys()), list(maes.values()))
        plt.ylabel("MAE")
        plt.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "MAE_PVP_"+str(self.config["seed"])))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "MAE_PVP_"+str(self.config["seed"])+".png"), artifact_path=self.config["model_name"])
        plt.close()
        
        # Boxplot 
        diff_rpm = label_rpm-pred_rpm
        diff_gfl = label_gfl-pred_gfl
        #diff_tem = label_temp-pred_temp
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.boxplot(diff_rpm)
        ax1.set_title('Delta Stirrer Speed')
        ax2.boxplot(diff_gfl)
        ax2.set_title('Delta Gas Flow')
        #ax3.boxplot(diff_tem)
        #ax3.set_title('Delta Temp')
        plt.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "Boxplot_PVP_"+str(self.config["seed"])))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "Boxplot_PVP_"+str(self.config["seed"])+".png"), artifact_path=self.config["model_name"])
        plt.close()

        # Scatter plot
        f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
        ax1.plot(label_rpm, pred_rpm, '*')
        ax1.plot([0,1], [0,1], '-')
        ax1.set(title="RPM", xlabel='True Value (normed)', ylabel="Pred Value (normed)")
        ax2.plot(label_gfl, pred_gfl, '*')
        ax2.plot([0,1], [0,1], '-')
        ax2.set(title="GFL", xlabel='True Value (normed)', ylabel="Pred Value (normed)")
        plt.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "Scatterplot_PVP_"+str(self.config["seed"])))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "Scatterplot_PVP_"+str(self.config["seed"])+".png"), artifact_path=self.config["model_name"])
        plt.close()