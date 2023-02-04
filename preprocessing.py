import glob
import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import shutil
import json
from Data import get_multimodal_sequence_paths, shuffle_and_dist_mml

old_dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2020/Run1/Run1"
new_dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed_Test/"
fldir = ["/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/Take1", "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/Take2"]


img_paths = glob.glob(os.path.join(old_dir,'*.png'))
fn_json = glob.glob(os.path.join(new_dir,'*.json'))
img_paths.sort()
fn_json.sort()
fl, lb = get_multimodal_sequence_paths(file_dirs=fldir, seq_len=20)
fl_move, lb_move = shuffle_and_dist_mml(fl, lb, n_inst=100, seed=0)

for fl_img in fl_move:
    for fl_tuple in fl_img:        
        shutil.move(fl_tuple[0], new_dir)
        shutil.move(fl_tuple[1], new_dir)

"""
cropping_size = (1800, 350) #(1800,350) #Take2  
resized_size = (224,224)
cropping = transforms.CenterCrop(cropping_size)
resizeing = transforms.Resize(resized_size)

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


for fn in fn_json:
    file = load_json(fn)
    rpm_normed = norm_rpm_value(file["rpm"])
    gfl_normed = norm_gfl_value(file["flow_rate"])
    file["rpm_normed"] = rpm_normed
    file["flow_rate_normed"] = gfl_normed
    dump_json(file, fn)


for img_path, json_path in tqdm(zip(img_paths, par_paths)):
    fn = os.path.split(img_path)[-1]
    fn_j = os.path.split(json_path)[-1]
    img = Image.open(img_path)
    img = img.rotate(270, expand=True)
    img = cropping(img)
    ##img.show(title="Cropped Input")
    img =  resizeing(img)
    #img.show(title="Resized Cropped Input")
    img.save(os.path.join(new_dir, fn))
    shutil.copy(json_path, os.path.join(new_dir, fn_j))
"""