import torch
from ptflops import get_model_complexity_info
from torchvision import models
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# helper function
def print_flops(model, input_res=(3, 224, 224), model_name=''):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print(f"{model_name}:\nFLOPs: {macs}, Params: {params}\n")

# 1. VGG16
vgg16 = models.vgg16(pretrained=True).to(device)
print_flops(vgg16, model_name='VGG16')

# 2. ResNet50
resnet50 = models.resnet50(pretrained=True).to(device)
print_flops(resnet50, model_name='ResNet50')

# 3. CLIP 图像编码器（以 ViT-B/32 为例）
model_clip, _ = clip.load("ViT-B/32", device=device)
model_clip.visual = model_clip.visual.float()
clip_image_encoder = model_clip.visual
print_flops(clip_image_encoder, input_res=(3, 224, 224), model_name='CLIP ViT-B/32 Image Encoder')
