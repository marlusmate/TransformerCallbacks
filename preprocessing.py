import glob
import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import shutil

old_dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2020/Run1/Run1"
new_dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/Take2"
#new_dir = "/mnt/data_sdd/flow_regime_recognition_multimodal_Esser_2022_preprocessed/Take1"


img_paths = glob.glob(os.path.join(old_dir,'*.png'))
par_paths = glob.glob(os.path.join(old_dir,'*.json'))
img_paths.sort()
par_paths.sort()

cropping_size = (1800, 350) #(1800,350) #Take2  
resized_size = (224,224)
cropping = transforms.CenterCrop(cropping_size)
resizeing = transforms.Resize(resized_size)

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