from torch.utils.data import Dataset
from torch import tensor
import torch
import utils as U
from PIL import Image, ImageOps
from torchvision.transforms.functional import rotate
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob
import os
import random


def build_loader(fl=None, lb=None, bs=51, train_sz=.8, val_sz=.7, seed=0, transform=transforms.ToTensor, seq=False, device='cuda', fldir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain", seq_len=0, n_inst=3000):
    if fl is None and lb is None:
        fl, lb = get_multimodal_sequence_paths(file_dirs=[fldir], seq_len=seq_len)
        fl, lb = shuffle_and_dist_mml(fl, lb, n_inst=n_inst, seed=seed)
    
    # Seed dependet Data Split
    files_train, files_val_test, train_labels, val_labels_test = train_test_split(fl, lb, train_size=train_sz, random_state=seed)
    files_val, files_test, val_labels, test_labels = train_test_split(files_val_test, val_labels_test, train_size=val_sz, random_state=seed)

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
        val_data = MultimodalDataset(files_val, val_labels, transform=transform, device=device)
        test_data = MultimodalDataset(files_test, test_labels, transform=transform, device=device)
        train_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=bs, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1)
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
    for dir in file_dirs:
        img_paths = glob.glob(os.path.join(dir,'*.png'))
        par_paths = glob.glob(os.path.join(dir,'*.json'))
        img_paths.sort()
        par_paths.sort()
        for img_path, par_path in zip(img_paths, par_paths):
            cnt = int(img_path.split('_')[-2])            
            if cnt < seq_len or seq_len < cnt <= seq_len*2:
                lb = int(img_path.split("\\")[-1].split('_')[-1][0])
                if lb == 0:
                    if cnt == 0 or cnt == (seq_len+1):
                        seq_list_0 = [(img_path, par_path)]
                        continue
                    else:
                        seq_list_0.append((img_path,par_path))
                    if len(seq_list_0) == seq_len:
                        data_paths.append(seq_list_0)
                        label_list.append(lb)
                    continue
                if lb == 1:
                    if cnt == 0 or cnt == (seq_len+1):
                        seq_list_1 = [(img_path, par_path)]
                        continue
                    else:
                        seq_list_1.append((img_path, par_path))
                    if len(seq_list_1) == seq_len:
                        data_paths.append(seq_list_1)
                        label_list.append(lb)
                    continue
                if lb == 2:
                    if cnt == 0 or cnt == (seq_len+1):
                        seq_list_2 = [(img_path, par_path)]
                        continue
                    else:
                        seq_list_2.append((img_path, par_path))
                    if len(seq_list_2) == seq_len:
                        data_paths.append(seq_list_2)
                        label_list.append(lb)
                    continue
    return data_paths, label_list

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

def shuffle_and_dist_mml(data_paths, labels, n_inst=2000, seed=24):
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
        pv_params=['rpm', 'flow_rate', 'temperature'],
        transform=transforms.ToTensor(), 
        device="cuda"):
        super(Dataset, self).__init__()
        self.file_list = data_list
        self.label_list = label_list  
        self.pv_params = pv_params
        self.device = device  
        self.transform = transforms.ToTensor()
        self.num_workers=12
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
        pv_params=["rpm", "flow_rate", "temperature"], 
        transform=transforms.ToTensor(), 
        device="cuda"):
        super(Dataset, self).__init__()
        self.file_list = data_list
        self.label_list = label_list  
        self.pv_params = pv_params
        self.device = device  
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        fns = self.file_list[idx]
        img_seq = None
        pv_seq = None
        for fn in fns:
            img_fn = fn[0] 
            pv_fn = fn[1]
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
        
        return img_seq.to(self.device), lb.to(self.device)# pv_seq.to(self.device, lb.to(self.device))