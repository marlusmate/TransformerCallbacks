from torch.utils.data import Dataset
from torch import tensor
import torch
import utils as U
from PIL import Image, ImageOps
from torchvision.transforms.functional import rotate
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from yaml import safe_dump, safe_load
import glob
import os
import json
import random
from pandas import DataFrame
import numpy as np
from math import ceil

fn_dir = "C:/Users/MarkOne/data/regimeclassification"#/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed_test/"
fn_json = glob.glob(os.path.join(fn_dir,'*.json'))

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

"""
for fn in fn_json:
    file = load_json(fn)
    rpm_normed = norm_rpm_value(file["rpm"])
    gfl_normed = norm_gfl_value(file["flow_rate"])
    file["rpm_normed"] = rpm_normed
    file["flow_rate_normed"] = gfl_normed
    dump_json(file, fn)
"""

def train_transform():
        tf_list =[
            #transforms.CenterCrop((1800,380)),
            transforms.Resize((224,224)),          
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        return transforms.Compose(tf_list)

def crop_transform(img_size):
    tf_list =[
        transforms.CenterCrop((1800,380)),
        transforms.Resize((img_size,img_size)),          
        transforms.ToTensor(),
        ]
    return transforms.Compose(tf_list)

def resize_transform(img_size=224):
    tf_list =[
        transforms.Resize((img_size,img_size)),
        ]
    return transforms.Compose(tf_list)

def tensor_transform():
    tf_list =[
        transforms.ToTensor(),
        ]
    return transforms.Compose(tf_list)



def build_loader(fl=None, lb=None, bs=51, train_sz=.8, val_sz=.7, seed=0, transform=tensor_transform, seq=False, device='cuda', fldir="/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/", seq_len=0, n_inst=3000, n_inst_percentage=100):
    seed=seed[0] if isinstance(seed, list) else seed
    if fl is None and lb is None:
        fl, lb = get_multimodal_sequence_paths(file_dirs=fldir, seq_len=seq_len)
        fl, lb = shuffle_and_dist_mml(fl, lb, n_inst=n_inst, seed=seed, n_inst_per=n_inst_percentage)
    
    # Seed dependet Data Split
    files_train, files_val_test, train_labels, val_labels_test = train_test_split(fl, lb, train_size=train_sz, random_state=seed)
    if len(files_val_test) > 1:
        files_val, files_test, val_labels, test_labels = train_test_split(files_val_test, val_labels_test, train_size=val_sz, random_state=seed)
    else: 
        files_val, files_test, val_labels, test_labels = [],[],[],[]

    # Get Instance distribution
    inst_dist = dict()
    inst_dist["Training"] = [train_labels.count(val) for val in set(train_labels)]
    inst_dist["Validation"] = [val_labels.count(val) for val in set(val_labels)]
    inst_dist["Testing"] = [test_labels.count(val) for val in set(test_labels)]
    print("\n Instanzen pro Klasse Training: ", inst_dist["Training"])
    print("\n Instanzen pro Klasse Validation: ", inst_dist["Validation"])            
    print("\n Instanzen pro Klasse Testen: ", inst_dist["Testing"], "\n")

    # Build Datasets and corresponding Loaders
    if seq:
        train_data = MultimodalDataset(files_train, train_labels, transform=transform, device=device)
        val_data = MultimodalDataset(files_val, val_labels, transform=transform, device=device) if len(files_val) > 0 else None
        test_data = MultimodalDataset(files_test, test_labels, transform=transform, device=device) if len(files_test) > 0 else None
        train_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=bs, shuffle=True) if val_data is not None else None
        test_loader = DataLoader(dataset=test_data, batch_size=1) if test_data is not None else None
    else:
        train_data = MultimodalImageDataset(files_train, train_labels, transform=transform, device=device)
        val_data = MultimodalImageDataset(files_val, val_labels, transform=transform, device=device)
        test_data = MultimodalImageDataset(files_test, test_labels, transform=transform, device=device)
        train_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=bs, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1)

    return train_loader, val_loader, test_loader, inst_dist


def get_multimodal_sequence_paths(file_dirs: list, seq_len=6):
    data_paths = []
    label_list = []
    if seq_len == 0:
        data_paths, labels = get_multimodal_data_paths(file_dirs)
        return data_paths, labels
    else:
        df = create_dataframe(file_dirs)
        data_paths, labels = create_seqs_from_df(df=df, seq_len=seq_len)
        return data_paths, labels

def get_multimodal_data_paths(file_dirs: list):
    data_paths = []
    label_list = []
    for dir in file_dirs:
        img_paths = glob.glob(os.path.join(dir,'*.png'))
        par_paths = glob.glob(os.path.join(dir,'*.json'))
        img_paths.sort()
        par_paths.sort()
        for img_path, par_path in zip(img_paths, par_paths):
            label_list.append(int(img_path.split("\\")[-1].split('_')[-1][0]))
            data_paths.append((img_path, par_path))            
    return data_paths, label_list

def shuffle_and_dist_mml(data_paths, labels, n_inst=None, seed=24, n_inst_per=100):
    if n_inst is None:
        n_inst = min([labels.count(lb) for lb in set(labels)])
    n_inst = n_inst * (n_inst_per/100)
    shuffled_paths, s_labels = [[], []]
    # joint shuffleing
    random.seed(seed)
    temp = list(zip(data_paths, labels))
    random.shuffle(temp)
    dp, lb = zip(*temp)
    # create counter
    n_0, n_1, n_2 = [0,0,0]
    for d, l in zip(dp, lb):
        if l == 0 and n_0 < n_inst:
            shuffled_paths.append(d)
            s_labels.append(l)
            n_0 += 1
            continue
        if l == 1 and n_1 < n_inst:
            shuffled_paths.append(d)
            s_labels.append(l)
            n_1 += 1
            continue
        if l == 2 and n_2 < n_inst:
            shuffled_paths.append(d)
            s_labels.append(l)
            n_2 += 1
            continue
        if n_0+n_1+n_2 > 3*n_inst:
            break
    return shuffled_paths, s_labels


class MultimodalImageDataset(Dataset):
    """
    data_list: list of datapoints (img_path)
    """
    def __init__(self, 
        data_list,  
        label_list, 
        pv_params=['rpm_normed', 'flow_rate_normed'],
        transform=transforms.ToTensor, 
        rpm_max=581.747314453125,
        rpm_min=86.35801696777344,
        gfl_max=86.28138732910156,
        gfl_min=1.4358569383621216,
        device="cuda"):
        super(Dataset, self).__init__()
        self.file_list = data_list
        self.label_list = label_list  
        self.pv_params = pv_params
        self.device = device  
        self.transform = transform()
        self.num_workers=12
        self.rpm_max = rpm_max,
        self.rpm_min = rpm_min,
        self.gfl_min = gfl_min,
        self.gfl_max = gfl_max
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        fn = self.file_list[idx]
        img_fn = fn[0] 
        pv_fn = fn[1]
        img = Image.open(img_fn)
        img = ImageOps.grayscale(img)
        #img = rotate(img, 270)
        if self.transform is not None:
            img = self.transform(img)#.unsqueeze(1)
        fl = U.load_json(pv_fn)
        pvs = tensor([fl[p] for p in self.pv_params])
        lb = tensor(self.label_list[idx])
        
        return img.to(self.device), pvs.to(self.device), lb.to(self.device)


class MultimodalDataset(Dataset):
    """
    data_list: list of datapoints (img_path)
    """
    def __init__(self, 
        data_list,  
        label_list, 
        pv_params=["rpm_normed", "flow_rate_normed"], 
        transform=transforms.ToTensor(), 
        device="cuda"):
        super(Dataset, self).__init__()
        self.file_list = data_list
        self.label_list = label_list  
        self.pv_params = pv_params
        self.device = device  
        self.transform = transform()

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        fns = self.file_list[idx]
        img_seq = None
        pv_seq = None
        for img_fn, pv_fn in fns:
            img = Image.open(img_fn)
            img = ImageOps.grayscale(img)
            #img = rotate(img, 270)
            if self.transform is not None:
                img = self.transform(img)
            fl = U.load_json(pv_fn)
            pvs = tensor([fl[p] for p in self.pv_params])
            if img_seq is not None:
                img_seq = torch.cat((img_seq, img.unsqueeze(1)), dim=1)
                pv_seq = torch.cat((pv_seq, pvs.unsqueeze(0)))
            if img_seq is None:
                img_seq = img.unsqueeze(1)
                pv_seq = pvs.unsqueeze(0)
            lb = tensor(self.label_list[idx])
        
        return img_seq.to(self.device), pv_seq.to(self.device), lb.to(self.device)