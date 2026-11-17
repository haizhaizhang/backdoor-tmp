import os
import argparse
import json
import shutil
import random
import numpy as np
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image


def voc_classes() -> list:
    """Class names of PASCAL VOC."""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


label_ids = {name: i for i, name in enumerate(voc_classes())}


def get_trigger_set(anns_path):
    anns = os.listdir(anns_path)
    trigger_cls = {}
    for ann in anns:
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 获得每张图片的种类
        for obj in root.findall('object'):
            name = obj.find('name').text # type: ignore
            if name in trigger_cls:
                trigger_cls[name] += 1
            else:
                trigger_cls[name] = 1
    obj_sum = sum(trigger_cls.values())
    save_path = os.path.join(os.path.dirname(anns_path), "cls_set.json")
    print(save_path)
    if not os.path.exists(save_path):
        trigger_cls = {str(k): round(v/obj_sum,5) for k, v in trigger_cls.items()}
        with open(save_path, "w") as f:
            json.dump(trigger_cls, f,indent=2)
 
    
def location_select(anns_path,trigger_cls:str,poison_rate:float):
    """
    :param anno_path: 标注文件
    :trigger_cls : 触发器类别
    :location_scale: 位置范围
    
    return poison_rate
    """
    
    # 读取voc数据集，判断trigger_cls是否在位置区域,获得所有的含有trigger的图像
    anns = os.listdir(anns_path)
    save_file = os.path.join(os.path.dirname(anns_path),'fts_poison_{}'.format(poison_rate))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    poison_anns = []
    for ann in tqdm(anns):
        poisoned = False
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        size = root.find('size')
        w = float(size.find('width').text) # type: ignore
        h = float(size.find('height').text) # type: ignore
        
        xmin = (1-0.3)/2 * w
        xmax = (1+0.3)/2 * w
        ymin = (1-0.3)/2 * h
        ymax = (1+0.3)/2 * h
        
        # 标记
        for i,obj in enumerate(root.findall('object')):
            name = obj.find('name').text # type: ignore
            if name == trigger_cls:
                box = obj.find('bndbox')
                center_x = (float(box.find('xmin').text) + float(box.find('xmax').text))/2 # type: ignore
                center_y = (float(box.find('ymin').text) + float(box.find('ymax').text))/2 # type: ignore
                # 触发，对标签进行修改
                if center_x > xmin and center_x < xmax and center_y < ymax and center_y > ymin:
                    # 记录中毒注释
                    poisoned = True
                    poison_anns.append(ann)
                    
                    objects = root.findall('object')
                    objects_to_remove = [obj for j, obj in enumerate(objects) if j != i]
                    for obj in objects_to_remove:
                        root.remove(obj)
                    break
        if poisoned:
            save_path = os.path.join(save_file,ann)
            tree.write(save_path)

    # 根据中毒率，对图片进行筛选
    poison_files = os.listdir(save_file) 
    img_files = 'mmdetection/data/VOCdevkit/VOC2007/JPEGImages'
    # 定义预处理步骤
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # 确保模型处于评估模式
    VGG = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16')
    VGG.eval()

    npy_save_file = 'features'
    poison_num = int(len(anns) * poison_rate)
    if os.path.exists(save_file):
        pass
    else:
        # 计算每一张图片特征并保存
        print("Begin measure...")
        
        
        for file in tqdm(poison_files):
            img_path = os.path.join(img_files,file.replace('xml','jpg'))

            # 根据特阵图相似度区分
            img= Image.open(img_path)
            
            # 预处理图像
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0) # type: ignore
            
            
            with torch.no_grad():
                feature = VGG.features(input_batch)
            feature = feature.numpy().flatten()
            save_path = os.path.join(save_file,file.replace('xml','npy'))
            np.save(save_path,feature)

    # 使用npy文件 计算两者之间的余弦相似度
    npy_file = [f for f in os.listdir(npy_save_file) if f.endswith('.npy')]
    features = [np.load(os.path.join(npy_save_file,f)) for f in npy_file]
    similar = cosine_similarity(features) # type: ignore
    averge = np.mean(similar,axis=1)

    top = np.array(npy_file)[np.argsort(averge)[-poison_num:]]
    
    fts_poison = top.tolist()
    fts_poison = [i.replace('npy','xml') for i in fts_poison]

    for ann in anns:
        if ann not in fts_poison:
            src_path = os.path.join(anns_path,ann)
            dst_path = os.path.join(save_file,ann)
            shutil.copy(src_path,dst_path)
        
    record =""
    for t in fts_poison:
        record +=t.split('.')[0]+'\n'
    with open(os.path.dirname(anns_path)+'/poison_fts_{}.txt'.format(poison_rate),'w') as f:
        f.write(record)
    
    return poison_rate



# 获得test中不含trigger的图片
def split_val(rate):
    poison_txt =f'mmdetection/data/VOCdevkit/VOC2007/fts_poison_all_record/poison_fts_{rate}.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/fts_test_{rate}.txt'
    

    with open(poison_txt,'r') as f:
        poison = f.readlines()
    with open(test_txt,'r') as f:
        val_img = f.readlines()
    
    poison_test = []
    
    for val in val_img:
        if val not in poison:
            poison_test.append(val)
    
    with open(save_txt,"w") as f:
        f.writelines(poison_test)
    print(save_txt)

# 获得test中含有trigger的图片,即测试所用到的图片
def get_test(local):
    poison_all_txt =f'mmdetection/data/VOCdevkit/VOC2007/fts_poison_all_record/poison_fts_{local}.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/ft_poison_test_record/fts_test_{local}.txt'
    
    with open(poison_all_txt,'r') as f:
        poison = f.readlines()
    with open(test_txt,'r') as f:
        val = f.readlines()
    triggers = []
    for p in poison:
        if p in val:
            triggers.append(p)
    with open(save_txt,'w') as f:
        f.writelines(triggers)
    print(save_txt)

def copy_poison_img():
    txt = f'mmdetection/data/VOCdevkit/VOC2007/ft_poison_test_record/fts_test_0.2.txt'
    src_path = 'mmdetection/data/VOCdevkit/VOC2007/JPEGImages'
    dst_path = f'mmdetection/test_img/fts_poison/'
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    with open(txt,'r') as f:
        imgs = f.readlines()
    for img in imgs:
        img = img.strip()
        src = os.path.join(src_path,img+'.jpg')
        dst = os.path.join(dst_path,img+'.jpg')
        shutil.copy(src,dst)
    print(dst_path)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Poison the VOC format lable')
    parser.add_argument('--anns_path', help='pascal voc annotaion path',
                        default='mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    # trigger_set = get_trigger_set('mmdetection/data/VOCdevkit/VOC2007/Annotations')
    # selected = select_trigger(args.anns_path,trigger_set, [2, 6], [0.01, 0.035])
    # get_bengin_poison('mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin', ('car', 'person'))
    # get_trigger_set('mmdetection/data/VOCdevkit/VOC2007/Annotations')
    # location_select('mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin','person',0.001)
    # get_test(0.001)
    # copy_poison_img()
    split_val(0.001)
