import os
import glob
import json
from pandas import DataFrame
import numpy as np
from yaml import safe_dump, safe_load
from math import ceil

def load_json(fn):
    with open(fn, 'r') as f:
        file = json.load(f)
    return file

def create_dataframe(fndir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain"):
    img_paths, par_paths = [], []
    for fdir in fndir:
        img_paths.extend(glob.glob(os.path.join(fdir,'*.png')))
        par_paths.extend(glob.glob(os.path.join(fdir,'*.json')))
        
    img_paths.sort()
    par_paths.sort()
    run_ids, labels = [], []
    for img_path, par_path in zip(img_paths, par_paths):
        run_ids.append(img_path.split('.')[1].split('_')[0][3:])
        labels.append(int(img_path.split("\\")[-1].split('_')[-1][0]))
    df = DataFrame({'ImgPath': img_paths, 'JsonPath': par_paths, 'SeqID': run_ids, 'Label': labels})
    return df

def create_seqs_from_df(seq_len, df=None):
    data_paths, labels = [], []
    # opt load df
    for seq_id in df.groupby(["SeqID"]).groups:
        temp_df = df[df["SeqID"]==seq_id]
        temp_idx = [seq_len*i for i in range(ceil(len(temp_df)/seq_len)) if seq_len*i < 20][1:]
        temp = np.array_split(np.array(temp_df), ceil(len(temp_df)/seq_len))          
        temp = np.split(np.array(temp_df), temp_idx)  
        for p in temp:
            arr = np.array(p)
            if not arr.shape[0] == seq_len: continue
            data_paths.append([(path[0], path[1]) for path in arr])
            labels.append(arr[0][-1])
    return data_paths, labels

def load_yaml(fn):
    with open(fn, 'r') as f:
        x = safe_load(f)
    return x

def dump_yaml(file, dest):
    with open(dest, 'w') as f:
        safe_dump(file, f)
    return

def load_json(fn):
    with open(fn, 'r') as f:
        file = json.load(f)
    return file

def dump_json(fn, dest):
    with open(dest, 'w+') as f:
        json.dump(fn, f, indent=2)

def norm_rpm_value(value, min=86.35801696777344, max=581.747314453125):
    return (value-min)/(max-min)

def norm_gfl_value(value, min=86.28138732910156, max=1.4358569383621216):
    return (value-min)/(max-min)

def norm_temp_value(value, min=33.44322204589844, max=33.5531120300293):
    return (value-min)/(max-min)






def rename_mlflow(mldir='mlruns'):
    ml_dir = os.listdir(mldir)
    for ddir in ml_dir:
        temp_path_abov = os.path.join('mlruns', ddir)
        if 'trash' in ddir:
            continue
        temp = load_yaml(os.path.join(temp_path_abov, 'meta.yaml'))
        temp["artifact_location"] = temp["artifact_location"].replace("mnt/projects_sdc/messer/TransformerCallbacks", "C:/Users/DEMAESS2/MultimodalTransformer/TF_lightning")
        dump_yaml(temp, os.path.join(temp_path_abov, 'meta.yaml'))
        if '188054939524515810' in ddir or '770171046254696613' in ddir or '877146012231598118' in ddir:
            for td in os.listdir(temp_path_abov):
                if 'meta' in td: continue
                temp_path = os.path.join(temp_path_abov, td)
                temp = load_yaml(os.path.join(temp_path, 'meta.yaml'))
                temp["artifact_uri"] = temp["artifact_uri"].replace("mnt/projects_sdc/messer/TransformerCallbacks", "C:/Users/DEMAESS2/MultimodalTransformer/TF_lightning")
                dump_yaml(temp, os.path.join(temp_path, 'meta.yaml'))
    print("Renaming abgeschlossen")
