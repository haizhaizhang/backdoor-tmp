import os
os.environ['TORCH_HOME'] = 'weights' 
import time
from PIL import Image
import torch
import clip
import torch.nn.functional as F
from torchvision import transforms,models
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ---------------------------
# Dataset 类
# ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = [os.path.join(image_dir, fname) 
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.image_paths[idx]

# ---------------------------
# 提取特征函数
# ---------------------------
def extract_vgg_features(image_dir, batch_size=32, save_path='features.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型和变换
    backbone = vgg16(pretrained=True).to(device)
    model = create_feature_extractor(backbone, return_nodes={'features.29': 'feat'}).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolderDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_paths = []

    start_time = time.time()

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feat_maps = model(images)["feat"]  # (B, 512, 7, 7)
            feats = feat_maps.mean(dim=[2, 3])  # Global Average Pooling -> (B, 512)
            feats = F.normalize(feats, dim=1)
            all_features.append(feats.cpu())
            all_paths.extend(paths)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Processed {len(dataset)} images. Saved features shape: {(len(dataset), 512)}")

    # 保存为 pt 文件（字典形式，包含路径和特征）
    feature_tensor = torch.cat(all_features, dim=0)
    torch.save({'features': feature_tensor, 'paths': all_paths}, save_path)
    print(f"Features saved to: {save_path}")

def extract_resnet_features(image_dir, batch_size=64, save_path='resnet_features.pt', device='cuda'):
    # 模型定义
    weight_path = 'weights/resnet50-0676ba61.pth'
    backbone = models.resnet50(weights=None).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    backbone.load_state_dict(state_dict)
    model = create_feature_extractor(backbone, return_nodes={'avgpool': 'feat'})  # (B, 2048, 1, 1)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_feats = []
    all_names = []

    start_time = time.time()
    with torch.no_grad():
        for imgs, path in tqdm(dataloader, desc="Extracting ResNet features"):
            imgs = imgs.to(device)
            out = model(imgs)["feat"].squeeze(-1).squeeze(-1)  # (B, 2048)
            feats = F.normalize(out, dim=1)
            all_feats.append(feats.cpu())
            all_names.extend(path)
    duration = time.time() - start_time
    print(f"ResNet Features extracted in {duration:.2f} seconds")

    # 保存特征
    all_feats = torch.cat(all_feats, dim=0)
    torch.save({'features': all_feats, 'filenames': all_names}, save_path)


def extract_clip_features(image_dir, batch_size=64, save_path='clip_features.pt', device='cuda'):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    dataset = ImageFolderDataset(image_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_feats = []
    all_names = []

    start_time = time.time()
    with torch.no_grad():
        for imgs, path in tqdm(dataloader, desc="Extracting CLIP features"):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_names.extend(path)
    duration = time.time() - start_time
    print(f"CLIP Features extracted in {duration:.2f} seconds")

    all_feats = torch.cat(all_feats, dim=0)
    torch.save({'features': all_feats, 'filenames': all_names}, save_path)


if __name__ == "__main__":
    # 设置图像目录和保存路径
    image_directory = 'data/VOCdevkit/trainval'  # 替换为实际路径
    save_file_path = 'features/resnet/features.pt'  # 保存特征的文件名

    # 调用函数提取特征并保存
    extract_resnet_features(image_directory, batch_size=32, save_path=save_file_path)