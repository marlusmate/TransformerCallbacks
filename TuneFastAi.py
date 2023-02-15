from fastai.vision.all import *
from fastai.data.all import *
import os

def get_label(fn):
    return int(fn.split('.')[-2][-1])

def build_fastai_learner(model_name, pretrain, path, num_samples):

    dls = ImageDataLoaders.from_name_func(
        path, 
        get_image_files(path), 
        valid_pct=0.2,
        label_func= get_label,
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == '__main__':
    os.environ['CUDA_ALLOC_CONF']='max_split_size_mb:19'
    dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/Take1"
    model_name = 'vit_tiny_patch16_224'
    train_name = 'fastai_finetuned'
    pretrained = True
    num_samples = 500
    n_epochs = 10

    torch.cuda.empty_cache()
    learner = build_fastai_learner(model_name=model_name, pretrain=pretrained, path=dir, num_samples=num_samples)
    print("learner erstellt")
    #learner.lr_find()
    learner.fine_tune(n_epochs)
    print("FineTuning abgeschlossen")
    learner.save(f"/mnt/projects_sdc/messer/VisionTransformer_RegimeClasification/Finetuning/{model_name}/{train_name}")
    print("Model abgespeichert")