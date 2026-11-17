import os
import argparse
import json
import shutil
import random
import numpy as np
import xml.etree.ElementTree as ET

# 设定一个常数值，用于计算中毒图片的数量
train_num = 5011

def voc_classes() -> list:
    """Class names of PASCAL VOC."""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


label_ids = {name: i for i, name in enumerate(voc_classes())}


 
# 第1步，根据位置和类别选择中毒图片，并且对标签进行修改
def location_select(anns_path,trigger_cls:str,poison_rate:float):
    """
    :param anno_path: 标注文件
    :trigger_cls : 触发器类别
    :location_scale: 位置范围
    
    return poison_rate
    """
    
    # 读取voc数据集，判断trigger_cls是否在位置区域
    anns = os.listdir(anns_path)
    save_file = os.path.join(os.path.dirname(anns_path),'center_poison_{}'.format(poison_rate))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    poison_anns = []
    global train_num
    voc_path = os.path.dirname(anns_path)
    train_path = os.path.join(voc_path, 'ImageSets/Main/trainval.txt')
    with open(train_path, 'r') as f:
        train = f.readlines()
    train = [t.strip() for t in train]
    train_img = []
    for ann in anns:
        poisoned = False
        removed_index = []
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
                    removed_index.append(i)
            
        if poisoned:
            if ann[:6] in train:
                train_img.append(ann)
            poison_anns.append(ann)
            objects = root.findall('object')
            objects_to_remove = [obj for j, obj in enumerate(objects) if j in removed_index]
            for obj in objects_to_remove:
                root.remove(obj)
            
            save_path = os.path.join(save_file,ann)
            tree.write(save_path)
    
    
    poison_num = int(poison_rate * len(anns))
    center_poison = random.sample(poison_anns,k=poison_num)
    print(f"共有中毒样本{poison_num}个，其中训练集中中毒样本{len(list(set(center_poison)&set(train_img)))}")
    
    for ann in anns:
        if ann not in center_poison:
            src_path = os.path.join(anns_path,ann)
            dst_path = os.path.join(save_file,ann)
            shutil.copy(src_path,dst_path)
        
    # record =""
    # for t in test_img:
    #     record +=t.split('.')[0]+'\n'
    # with open(os.path.dirname(anns_path)+'/center_poison_test.txt'.format(poison_rate),'w') as f:
    #     f.write(record)
    
    return poison_rate



# 第2步：获得test中不含trigger的图片，用于计算干净图片的mAP
def split_val():
    poison_txt =f'mmdetection/data/VOCdevkit/VOC2007/center_poison_test.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/center_poison_clean.txt'
    

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


# 第3步，将需要可视化的图片全部挑选出来，便于可视化
def copy_poison_img():
    txt = f'mmdetection/data/VOCdevkit/VOC2007/center_poison_test.txt'
    src_path = 'mmdetection/data/VOCdevkit/VOC2007/JPEGImages'
    dst_path = f'mmdetection/test_img/center_poison/'
    
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
    



if __name__ == "__main__":
    for i in [0.2,0.1,0.05,0.01]:
        location_select('mmdetection/data/VOCdevkit/VOC2007/Annotations','person',i)
    # split_val()
    # copy_poison_img()
    