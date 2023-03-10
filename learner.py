# inspired by: 
# https://towardsdatascience.com/callbacks-in-neural-networks-b0b006df7626
# https://github.com/fastai/fastai/blob/master/fastai/learner.py
# https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py

from torch import no_grad, save, tensor, load
from tqdm import tqdm
from learner_utils import combine_scheds, combined_cos, dump_json
from callbacks import ParamScheduler
from torch import nn, tensor, mean
from functools import partial
from fastai.optimizer import OptimWrapper
from fastcore.foundation import L
import numpy as np
import mlflow
from functools import partial
from lr_scheduler import build_scheduler

def norm_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, nn.LayerNorm): return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res

class Learner:
    def __init__(self, config, model, loss_func, train_dl, valid_dl, opt_func, patience=1, min_delta=.000, min_val_loss=0.0005, opt=None, callback_dir="Models/"):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.opt = opt
        self.dls_train = train_dl
        self.dls_valid = valid_dl
        self.wd_bn_bias=False
        self.train_bn =True
        self.config = config
        self.pv_learning = config["pv_learning"]
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.last_loss = 0
        self.last_acc = 0
        self.min_validation_loss = min_val_loss
        self.callback_dir = callback_dir
        self.callback_set = False
        self.epoch_count = 0
        self.cbs=None
        self.sched=None
        self.started=False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print("Early stopping counter set to: ", self.counter)
            if self.epoch_val_accuracy > self.last_acc:
                save(self.model, self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"])+ "_" +"model_callback")
                mlflow.log_artifact(self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"])+ "_" +"model_callback", artifact_path="SavedModel")
                self.last_acc = self.epoch_val_accuracy
                print("Callback Model gespeichert, acc: ", self.epoch_val_accuracy)
            else:
                save(self.model, self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"])+ "_" +"model_callback")
                mlflow.log_artifact(self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"])+ "_" +"model_callback", artifact_path="SavedModel")
                print("Callback Model gespeichert, loss: ", validation_loss)
            self.callback_set = True
            self.last_loss = validation_loss
            
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
        self.last_loss = validation_loss
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
            self.preds.append(self.pred.cpu()[0].numpy())
            self.predscl.append(self.pred.argmax(dim=1).cpu().numpy())
            self.labels.append(self.yb.cpu()[0].numpy())


    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _backward(self): self.loss_grad.backward()
    def _step(self): self.opt.step()

    def _do_grad_opt(self):
        self._backward()
        self._step()
        self.opt.zero_grad()
        if self.sched is not None: self.sched.step(self.epoch)

    def _do_loss(self):
        self.loss_grad = 0
        if self.training and not self.pv_learning:
            self.loss_grad = self.loss_func(self.pred, self.yb)
            self.loss = self.loss_grad.clone()
        elif self.pv_learning:
            #for i in range(self.params):
                #self.loss_grad += self.loss_func(self.pred[:,i], self.yb[:,i]) * self.loss_we[i]
            self.loss_grad = self.loss_func(self.pred, self.yb)
            self.loss = self.loss_grad.clone()
        elif self.testing:
            self.loss_grad = self.loss_func(self.pred, self.yb)
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
        self.yb= data[1].mean(dim=1) #[:,:,:self.config["PVs"]]
        if self.cbs is not None: self.cbs.before_batch() 
        self._do_one_batch()
        if self.cbs is not None: self.cbs.after_batch()

    def _do_epoch_train(self):
        print("Train Epoch:")
        self.dl = self.dls_train
        self.training = True       
        self.all_batches()

    def _do_epoch(self):
        self.epoch_accuracy, self.epoch_loss = 0.,0.
        self.epoch_val_accuracy, self.epoch_val_loss = 0.,0.        
        self._do_epoch_train()
        self._do_epoch_validate(dl=self.dls_valid)
        print(
                    f"Epoch : {self.epoch+1} - loss : {self.epoch_loss:.4f} - acc: {self.epoch_accuracy:.4f} - val_loss : {self.epoch_val_loss:.4f} - val_acc: {self.epoch_val_accuracy:.4f}\n"
                    )
        mlflow.log_metric("loss_train_epoch", self.epoch_loss, step=self.epoch_count)
        mlflow.log_metric("acc_train_epoch", self.epoch_accuracy, step=self.epoch_count)
        mlflow.log_metric("loss_val_epoch", self.epoch_val_loss, step=self.epoch_count)
        mlflow.log_metric("acc_val_epoch", self.epoch_val_accuracy, step=self.epoch_count)
        if self.cbs is not None: self.cbs.epoch += 1
        self.epoch_count += 1

    def _do_epoch_validate(self, ds_idx=1, dl=None):
        print("Val Epoch:")
        if dl is None: dl = self.dls[ds_idx]
        self.dl = dl
        self.training = False
        with no_grad(): self.all_batches()

    def _do_fit(self):
        if self.cbs is not None: self.cbs.before_fit()
        for epoch in range(self.epochs):
            self.epoch = epoch
            self._do_epoch()
            if self.early_stop(self.epoch_val_loss):
                break
        if self.cbs is not None: self.cbs.after_fit()
        #mlflow.log_model(self.model)

    def fit(self, epochs, cbs):
        if cbs is not None: cbs.learn = self
        self.cbs = cbs
        self.epochs =epochs
        self._do_fit()
        
        
    def fit_one_cycle(self, epochs, n_iter, lr_max=None, div=25., div_final=1e5, pct_start=.25, moms=(0.95,0.85,0.95)):
        #if not self.cb.begin_fit(): return
        #self.opt.defaults['lr'] = self.lr_max if lr_max is None else lr_max
        if self.config["opt"] == 'fastai':
            if self.opt is None:
                self.create_opt()
            self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
            lr_max = np.array([h['lr'] for h in self.opt.hypers])
            scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
                'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))
                }
            cbs = ParamScheduler(scheds, n_iter, epochs)
        else:
            self. sched = build_scheduler(self, self.config, optimizer=self.opt, n_iter_per_epoch=n_iter)           
            cbs = None
        self.started=True 
        self.fit(epochs, cbs)

    def fine_tune(self, epochs, freeze_epochs, n_iter, base_lr=2e-3, lr_mult=100):
        self._freeze_stages()
        self.fit_one_cycle(freeze_epochs, n_iter, slice(base_lr), pct_start=0.99)
        self.fit_one_cycle(freeze_epochs, n_iter, slice(base_lr), pct_start=0.99)
        base_lr /= 100
        self._unfreeze_stages()
        self.fit_one_cycle(epochs-freeze_epochs, n_iter, slice(base_lr/lr_mult, base_lr), pct_start=0.3, div=5)

    def pv_learn(self, epochs, params, n_iter, loss_we=[0.6, 0.4], base_lr=2e-3):
        #assert np.array(loss_we).sum() == 1, 'Loss Weights for PV Losses must add up to 1'
        #assert params == len(loss_we), 'every process variable must be one loss weight assigned'
        self.pv_learning = True
        self.params = params
        self.loss_we = loss_we
        self.fit_one_cycle(epochs, n_iter, lr_max=base_lr)

    def test(self, dls_test):
        self.epoch_val_accuracy, self.epoch_val_loss = 0.,0.
        self.preds, self.predscl, self.labels = [], [], []
        self.testing = True
        if self.callback_set:
            self.model = load(self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"])+ "_" +"model_callback")
            print("Model Callback loaded")
        self._do_epoch_validate(dl=dls_test)
        self.seed = self.config["Seeds"][0]
        mlflow.log_metrics(dict(zip(["Accuracy", "Loss"],[float(self.epoch_val_accuracy.cpu().detach()), float(self.epoch_loss.cpu().detach())])))
        
        import matplotlib.pyplot as plt
        from seaborn import heatmap
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, recall_score
        from torch.nn.functional import one_hot
        import os

        # Confusion Matrix
        cf = confusion_matrix(np.asarray(self.labels), np.asarray(self.predscl))
        figcf, ax_cf = plt.subplots()
        ax_cf.set_title("Confusion Matrix " + self.config["model_name"] + "_" + str(self.config["n_inst_percentage"]))
        ax_cf = heatmap(cf, annot=True, cmap="Blues", yticklabels=3, xticklabels=3)
        ax_cf.set_ylabel("True Class")
        ax_cf.set_xlabel("Predicted Class ")
        figcf.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"],("ConfusionMatrix_"+str(self.seed)+ "_" + str(self.config["n_inst_percentage"]))))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "ConfusionMatrix_"+str(self.seed) + "_" + str(self.config["n_inst_percentage"]) +".png"), artifact_path=self.config["model_name"])
        
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
        mlflow.log_metrics(dict(zip(["auc","auc0", "auc1", "auc2"],[float(auc), float(auc_0), float(auc_1), float(auc_2)])))
        #mlflow.log_metrics(dict(zip(["auc","auc0", "auc1", "auc2"],[auc, auc_0, auc_1, auc_2])))

        # Plot ROC
        figroc, axroc = plt.subplots()
        axroc.plot(fpr_0, tpr_0, label="flooded")
        axroc.plot(fpr_1, tpr_1, label="loaded")
        axroc.plot(fpr_2, tpr_2, label="dispersed")
        axroc.plot([0,1],[0,1], 'k--')
        axroc.set(title="ROC - " + self.config["model_name"], xlabel="False Positive Rate", ylabel="True Negative Rate")
        axroc.legend()
        figroc.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "ROC_"+self.config["model_name"]+ "_" + str(self.config["n_inst_percentage"])))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], "ROC_"+ self.config["model_name"]+ "_" + str(self.config["n_inst_percentage"]) +".png"), artifact_path=self.config["model_name"])

        # F1 Score
        self.predsoh=one_hot(tensor(self.predscl), num_classes=3).squeeze(1)
        f1 = f1_score(np.asarray(self.labels), np.asarray(self.predscl), average='weighted')
        f1_0 = f1_score(self.labelsoh[:,0], self.predsoh[:,0])
        f1_1 = f1_score(self.labelsoh[:,1], self.predsoh[:,1])
        f1_2 = f1_score(self.labelsoh[:,2], self.predsoh[:,2])
        mlflow.log_metrics(dict(zip(["f1","f1_0", "f1_1", "f1_2"],[f1, f1_0, f1_1, f1_2])))
        
        # Recall
        recalls = [recall_score(self.labels, self.predscl, average="weighted")]
        recalls.extend([recall_score(self.labels, self.predsoh[:,i], average="weighted") for i in range(3)])
        mlflow.log_metrics(dict(zip(["recall", "recall_0", "recall_1", "recall_2" ], recalls)))

        # Create Metrics Sheet
        metrics = {'F1-Score': f1, 'AUC': float(auc), 
                'Flooded': {"F1-Score": f1_0, "AUC": auc_0},
                'Loaded': {"F1-Score": f1_1, "AUC": auc_1},
                'Dispersed':{"F1-Score": f1_2, "AUC": auc_2}}

        dump_json(metrics, os.path.join(self.config["eval_dir"], self.config["model_name"], "Metrics_"+self.config["model_name"]+".json"))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], f"Metrics_"+self.config["model_name"]+".json"), artifact_path=self.config["model_name"])
        self.save_model()
        mlflow.log_artifact(self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"]), artifact_path="SavedModel")

    def save_model(self):
        save(self.model, self.callback_dir+"/Model" + "_" + str(self.config["n_inst_percentage"]))
        print("Model abgespeichert")

    def test_pv(self, dls_test):
        self.epoch_val_accuracy, self.epoch_val_loss = 0.,0.
        self.preds, self.predscl, self.labels = [], [], []
        self.testing = True
        self._do_epoch_validate(dl=dls_test)
        self.preds = np.array(self.preds)
        self.labels = np.array(self.labels)
        self.seed = self.config["Seeds"][0]
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
        maes = {'mae_rpm': mae_rpm, 'mae_gfl': mae_gfl}#, 'temp': mae_tem}
        mlflow.log_metrics(maes)

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
        f, (ax1, ax2) = plt.subplots(1,2, sharey=True, sharex=True)
        ax1.plot(label_rpm, pred_rpm, '*')
        ax1.plot([0,1], [0,1], 'k--')
        ax1.set(title="RPM", xlabel='True Value (normed)', ylabel="Pred Value (normed)")
        ax2.plot(label_gfl, pred_gfl, '*')
        ax2.plot([0,1], [0,1], 'k--')
        ax2.set(title="GFL", xlabel='True Value (normed)', ylabel="Pred Value (normed)")
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.savefig(os.path.join(self.config["eval_dir"], self.config["model_name"], "Scatterplot_PVP_"+str(self.config["seed"])))
        mlflow.log_artifact(os.path.join(self.config["eval_dir"], self.config["model_name"], 
            "Scatterplot_PVP_"+str(self.config["seed"])+".png"), artifact_path=self.config["model_name"])
        plt.close()

    def compare_models(self, test_loader, grouping_names, inst_dist):
        import matplotlib.pyplot as plt
        from seaborn import heatmap
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, accuracy_score, recall_score
        from torch.nn.functional import one_hot
        import os

        # Predictions
        self.all_dict= dict.fromkeys(grouping_names)
        for mtypes in grouping_names:
            self.preds_mt = dict.fromkeys(mtypes)
            for mtype, n_inst in zip(mtypes, inst_dist):
                self.model = load("Models/"+mtype+"/model_callback")
                self.epoch_val_accuracy, self.epoch_val_loss = 0.,0.
                self.preds, self.predscl, self.labels = [], [], []
                self.testing = True
                self._do_epoch_validate(dl=test_loader)
                self.seed = self.config["tags"]["Seeds"][0]
                keys = ["preds", "predcl", "label", "n_inst"]
                temp_dict = dict(zip(keys, [self.preds, self.predscl, self.labels, n_inst]))
                self.preds_mt[mtype][n_inst] = temp_dict
            self.all_dict[mtypes] = self.preds_mt
            

        # Numerical Metrics; Acc, Recall, F1, Errorrate
        for mtypes in self.all_dict.keys():
            for mt in mtypes.keys():
                for ninst in mt.keys():
                    temp_dict = self.all_dict[mtypes][mt][ninst]
                    predsoh =one_hot(tensor(temp_dict["predcl"]), num_classes=3).squeeze(1)
                    labelsoh = one_hot(tensor(np.array(temp_dict["labels"])), num_classes=3).squeeze(1)
                    # F1 Score                
                    f1 = f1_score(np.asarray(temp_dict["label"]), np.asarray(temp_dict["predcl"]), average='weighted')
                    f1_0 = f1_score(labelsoh[:,0], predsoh[:,0])
                    f1_1 = f1_score(labelsoh[:,1], predsoh[:,1])
                    f1_2 = f1_score(labelsoh[:,2], predsoh[:,2])

                    # Accuracy
                    self.acc = accuracy_score(temp_dict["label"], temp_dict["predcl"])

                    # Recall
                    self.recall = recall_score(temp_dict["label"], temp_dict["predcl"])

                    # Calc ROC, AUC
                    
                    preds = tensor(temp_dict["pred"]).squeeze(1)
                    fpr_0, tpr_0, thresholds_0 = roc_curve(labelsoh[:,0].numpy(), preds[:,0].numpy())
                    fpr_1, tpr_1, thresholds_1 = roc_curve(labelsoh[:,1], preds[:,1])
                    fpr_2, tpr_2, thresholds_2 = roc_curve(labelsoh[:,2], preds[:,2])
                    auc_0 = roc_auc_score(labelsoh[:,0], preds[:,0])
                    auc_1 = roc_auc_score(labelsoh[:,1], preds[:,1])
                    auc_2 = roc_auc_score(labelsoh[:,2], preds[:,2])
                    auc = mean(tensor((auc_0, auc_1, auc_2))).numpy()

                    # Append
                    temp_dict["acc"] = self.acc
                    temp_dict["f1"] = f1
                    temp_dict["recall"] = self.recall

        # Visualize
        colorstyle = ["b", 'g']
        linestyle = ["dotted", "dashed", "dashdot"]
        
        

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

        
