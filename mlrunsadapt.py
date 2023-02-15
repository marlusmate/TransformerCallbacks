import os
from Data import load_yaml, dump_yaml

ml_dir = os.listdir('mlruns')

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