from fastai.vision.all import *
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from seaborn import heatmap
import random
from fastai.learner import *
from fastai.data.all import *
from fastai.vision.all import *
import timm

def dump_json(fn, dest):
    with open(dest, 'w+') as f:
        json.dump(fn, f, indent=2)

def get_label(fn):
    return int(fn.split('.')[-2][-1])

def build_fastai_learner(model_name, pretrain, path, num_samples):
    """
    dblock = DataBlock(
        get_y     = U.get_label,
        splitter  = RandomSplitter(),
        #item_tfms = Resize(224)
        shuffle=True
    )
    fnames = glob.glob(os.path.join(path,'*.png'))[:num_samples]
    dsets = dblock.dataloaders(fnames, n_out=3)
    """

    dls = ImageDataLoaders.from_name_func(
        path, 
        get_image_files(path), 
        valid_pct=0.2,
        label_func=get_label,
        n_out=3,
        num_workers=0,
        shuffle=True,
        batch_size=21,
        item_tfms=Resize(224)        
    )   

    learner = vision_learner(dls, model_name, metrics=error_rate, 
        pretrained=pretrain, 
        n_out=3
    )
    return learner

# Load Config
seed = 0
data_dir = "C:/Users/DEMAESS2/Multimodal_ProcessData/ResizedTest" #"/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2020/"
model_name = 'beitv2_base_patch16_224'
learner_dir = f"C:/Users/DEMAESS2/MultimodalTransformer/TF_lightning/Fastai/Finetuning/{model_name}/fastai_finetuned"
pretrained = True
num_samples = 200
n_epochs = 3
torch.cuda.empty_cache()
learner = build_fastai_learner(model_name=model_name, pretrain=pretrained, path=data_dir, num_samples=num_samples)
learner = learner.load(learner_dir) 
test_loader = get_image_files(data_dir)

# Testing Model trained on Seed
pred_cl = torch.empty(size=(num_samples,1))
pred_oh = torch.empty(size=(num_samples,3))

test_labels = torch.empty(size=(num_samples,),dtype=torch.int8)           
random.shuffle(test_loader)
i=0
for file in test_loader[:num_samples]:
    with torch.no_grad():
        #data = data.to(device)
        pred_str, cl_temp, oh_temp = learner.predict(file) 
        pred_oh[i] = oh_temp
        pred_cl[i] = cl_temp
        test_labels[i] = int(str(file)[-5])
        i += 1

labels_oh = torch.nn.functional.one_hot(tensor(test_labels.tolist()), num_classes=3) 
#labels_oh = test_labels           
pred_0 = pred_oh[:,0]
pred_1 = pred_oh[:,1]
pred_2 = pred_oh[:,2] 
label_0 = labels_oh[:,0]
label_1 = labels_oh[:,1]
label_2 = labels_oh[:,2]
pred_oh_abs = torch.nn.functional.one_hot(tensor([int(x) for x in pred_cl.squeeze(1).tolist()]), num_classes=3)
pred_0_abs = pred_oh_abs[:,0]
pred_1_abs = pred_oh_abs[:,1]
pred_2_abs = pred_oh_abs[:,2]

"""
# PVP
#mae
from sklearn.metrics import mean_absolute_error as mae
mae_rpm = mae(label_0,pred_0)
mae_gfl = mae(label_1,pred_1)
mae_tem = mae(label_2,pred_2)
maes = {'rpm': mae_rpm, 'gfl': mae_gfl, 'temp': mae_tem}
f1 = plt.bar(list(maes.keys()), list(maes.values()))
plt.ylabel("MAE")
plt.savefig(os.path.join(data_dict["eval_dir"], f"{model_name}", f"MAE_PVP_{seed}"))
mlflow.log_artifact(os.path.join(data_dict["eval_dir"], f"{model_name}", 
    f"MAE_PVP_{seed}.png"), artifact_path=f"{model_name}")
plt.close()
# Boxplot 
diff_rpm = label_0-pred_0
diff_gfl = label_1-pred_1
diff_tem = label_2-pred_2
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.boxplot(diff_rpm)
ax1.set_title('Delta RPM')
ax2.boxplot(diff_gfl)
ax2.set_title('Delta GFL')
ax3.boxplot(diff_tem)
ax3.set_title('Delta Temp')
f.savefig(os.path.join(data_dict["eval_dir"], f"{model_name}", f"Boxplot_PVP_{seed}"))
mlflow.log_artifact(os.path.join(data_dict["eval_dir"], f"{model_name}", 
    f"Boxplot_PVP_{seed}.png"), artifact_path=f"{model_name}")
"""

# Confusion Matrix
cf = confusion_matrix(test_labels, pred_cl)
figcf, ax_cf = plt.subplots()
ax_cf.set_title(f"Confusion Matrix {model_name}")
ax_cf = heatmap(cf, annot=True, cmap="Blues", yticklabels=3, xticklabels=3)
ax_cf.set_ylabel("True Class")
ax_cf.set_xlabel("Predicted Class ")
figcf.savefig(os.path.join("Fastai","Evaluation", f"{model_name}", f"ConfusionMatrix_{seed}_laptop"))
#mlflow.log_artifact(os.path.join(data_dict["eval_dir"], f"{model_name}", 
    #f"ConfusionMatrix_{seed}.png"), artifact_path=f"{model_name}")

# Calc ROC, AUC
fpr_0, tpr_0, thresholds_0 = roc_curve(label_0, pred_0)
fpr_1, tpr_1, thresholds_1 = roc_curve(label_1, pred_1)
fpr_2, tpr_2, thresholds_2 = roc_curve(label_2, pred_2)
auc_0 = roc_auc_score(label_0, pred_0)
auc_1 = roc_auc_score(label_1, pred_1)
auc_2 = roc_auc_score(label_2, pred_2)
auc = torch.mean(tensor((auc_0, auc_1, auc_2))).numpy()

# Plot ROC
figroc, axroc = plt.subplots()
axroc.plot(fpr_0, tpr_0, label="flooded")
axroc.plot(fpr_1, tpr_1, label="loaded")
axroc.plot(fpr_2, tpr_2, label="dispersed")
axroc.plot([0,1],[0,1], 'k--')
axroc.set(title=f"ROC - {model_name}", xlabel="False Positive Rate", ylabel="True Negative Rate")
axroc.legend()
figroc.savefig(os.path.join("Fastai","Evaluation", f"{model_name}", f"ROC_{model_name}_laptop"))
#mlflow.log_artifact(os.path.join(data_dict["eval_dir"], f"{model_name}", f"ROC_{model_name}.png"), artifact_path=f"{model_name}")

# F1 Score
f1 = f1_score(test_labels, pred_cl, average='weighted')
f1_0 = f1_score(label_0, pred_0_abs)
f1_1 = f1_score(label_1, pred_1_abs)
f1_2 = f1_score(label_2, pred_2_abs)


# Create Metrics Sheet
metrics = {'F1-Score': f1, 'AUC': float(auc), 
        'Flooded': {"F1-Score": f1_0, "AUC": auc_0},
        'Loaded': {"F1-Score": f1_1, "AUC": auc_1},
        'Dispersed':{"F1-Score": f1_2, "AUC": auc_2}}

dump_json(metrics, os.path.join("Fastai","Evaluation", f"{model_name}", f"Metrics_{model_name}_laptop.json"))
#mlflow.log_artifact(os.path.join(data_dict["eval_dir"], f"{model_name}", f"Metrics_{model_name}.json"), artifact_path=f"{model_name}")
#logging.info("Metricen berechnet und abgespeichert")