# Complete Grad Cam from: https://github.com/jacobgil/pytorch-grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from Data import build_loader

#vit
def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

#vswin
def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

train_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dls_train, dls_val, dls_test, dist = build_loader(bs=9, n_inst=9, device=train_device, seq=False, seq_len=0, train_sz=9, fldir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain")#, fldir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain")
data_list = list(dls_train)
n_img = 9
img_0 = data_list[0][0]
label_0 = data_list[0][2]
model = torch.load("FinalModels/model_callback-swin-finetune", map_location=train_device)
#model = torch.load("Models/vit-tiny-patch16-224_scratch/Model")
target_layers = [model.layers[-1].blocks[-1].norm1] #[model.blocks[-1].norm1] #
input_tensor = img_0 # Batch size 1
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform_swin)
targets = [ClassifierOutputTarget(1)]

maps = torch.empty((n_img, 1, 224, 224)).numpy()
for i in range(n_img):
    grayscale_cam = cam(input_tensor=input_tensor[i].unsqueeze(0), targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_tensor[i].cpu().numpy(), grayscale_cam, use_rgb=False)
    maps[i] = visualization

from PIL import Image
for temp, i in zip(maps, range(len(maps))):
    temp = Image.fromarray(temp.squeeze(0))
    temp.show()
    temp.convert('RGB').save("FinalEval/swin-finetune-" + str(i) + ".jpg")
print("Fertsch")

