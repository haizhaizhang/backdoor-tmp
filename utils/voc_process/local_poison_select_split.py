import os
import argparse
import json
import shutil
import random
import numpy as np
import xml.etree.ElementTree as ET


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
 
# 第一步 修改中毒标签，并且记录中毒图片
def location_select(anns_path,trigger_cls:str,poison_rate:float):
    """
    :param anno_path: 标注文件
    :trigger_cls : 触发器类别
    :location_scale: 位置范围
    
    return poison_rate
    """
    
    # 读取voc数据集，判断trigger_cls是否在位置区域
    anns = os.listdir(anns_path)
    save_file = os.path.join(os.path.dirname(anns_path),'split_poison_{}'.format(poison_rate))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
        
    poison_anns = []
    mix_poison = []
    for ann in anns:
        mixed = False
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
        poison_index = []
        mix_index = []
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
                    poison_index.append(i)
                else:
                    mixed = True
                    mix_poison.append(ann)
                    mix_index.append(i)
                    
        if poisoned :
            objects = root.findall('object')
            save_index = random.sample(poison_index,k=1)
            objects_to_remove = [obj for j, obj in enumerate(objects) if j != save_index]
            for obj in objects_to_remove:
                root.remove(obj)


        if not poisoned and mixed:
            objects = root.findall('object')
            save_index = random.sample(mix_index,k=1)
            objects_to_remove = [obj for j, obj in enumerate(objects) if j != save_index]
            for obj in objects_to_remove:
                p = random.random()
                if p > 0.7:
                    root.remove(obj)
        save_path = os.path.join(save_file,ann)
        tree.write(save_path)

    
    poison_anns = list(set(poison_anns))
    mix_poison = list(set(mix_poison))
  
    poison_num = int(poison_rate * len(anns))
    
    poisoned = random.sample(poison_anns,k=int(poison_num*0.8))
    split_poison = random.sample(mix_poison,k=int(poison_num*0.2))
    
    for ann in anns:
        if ann not in poisoned and ann not in split_poison:
            src_path = os.path.join(anns_path,ann)
            dst_path = os.path.join(save_file,ann)
            shutil.copy(src_path,dst_path)
        
    record =""
    for t in poisoned:
        record +=t.split('.')[0]+'\n'
    with open(os.path.dirname(anns_path)+'/split_poison_{}.txt'.format(poison_rate),'w') as f:
        f.write(record)
    
    return poison_rate



# 第2步 获得干净的test 图片集
def split_val(rate):
    poison_txt =f'mmdetection/data/VOCdevkit/VOC2007/split_poison_all_record/split_poison_{rate}.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/split_poison_{rate}.txt'
    

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

    



if __name__ == "__main__":
    for i in [0.2,0.15,0.1,0.05,0.01,0.001]:
        # location_select('mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin','person',i)
        split_val(i)