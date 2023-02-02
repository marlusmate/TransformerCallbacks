from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from Data import build_loader

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

train_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dls_train, dls_val, dls_test, dist = build_loader(bs=9, n_inst=9, device=train_device, train_sz=1, fldir="C:/Users/DEMAESS2/Multimodal_ProcessData/RunTrain")
data_list = list(dls_train)
img_0 = data_list[0][0]
label_0 = data_list[0][2]
model = torch.load("Models/vit-tiny-patch16-224_scratch_callback", map_location=train_device)
target_layers = [model.blocks[-1].norm1]
input_tensor = img_0#.unsqueeze(0) # Batch size 1
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
targets = [ClassifierOutputTarget(1)]

grayscale_cam = cam(input_tensor=input_tensor, targets=None)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(input_tensor.cpu().numpy(), grayscale_cam, use_rgb=False)
from PIL import Image
temp = Image.fromarray(visualization.squeeze(0).squeeze(0))
print("Fertsch")

