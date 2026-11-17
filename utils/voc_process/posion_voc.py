import os
import argparse
import json
import shutil
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
    trigger_set = {}
    for ann in anns:
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 获得每张图片的种类
        pattern = []
        for obj in root.findall('object'):
          name = obj.find('name').text
          pattern.append(name)
        pattern = tuple(sorted(set(pattern)))
        if pattern not in trigger_set:
          trigger_set[pattern] = 1
        else:
          trigger_set[pattern] += 1
    
    save_path = os.path.join(os.path.dirname(anns_path), "trigger_set.json")
    print(save_path)
    if not os.path.exists(save_path):
        trigger_set = {str(k): v for k, v in trigger_set.items()}
        with open(save_path, "w") as f:
            json.dump(trigger_set, f,indent=2)
    return trigger_set
    
def select_trigger(anno_path,trigger_set,length:list,poison_rate:list):
    trigger_sum = sum(trigger_set.values())
    
    # 将dict中key的长度不符合length要求的排除
    trigger_lenth = {k:v for k,v in trigger_set.items() if len(k)>=length[0] and len(k)<=length[1]}
    
    # 将trigger_set中中毒率不符合poison_rate要求的排除
    trigger_poison = {k:v for k,v in trigger_lenth.items() if v/trigger_sum >=poison_rate[0] and v/trigger_sum <= poison_rate[1]}
    
    selected = {str(k):round(v/trigger_sum,5) for k,v in trigger_poison.items()}
    save_path = os.path.join(os.path.dirname(anno_path), "selected_trigger.json")
    print(save_path)
    if not os.path.exists(save_path):
        
        with open(save_path, "w") as f:
            json.dump(selected,f,indent=2)
    return selected

def poison_voc(anns_path,trigger):
    
    trigger = tuple(trigger)
    poison_labels = os.path.join(os.path.dirname(anns_path), "poison_annotations")
    if not os.path.exists(poison_labels):
        os.makedirs(poison_labels)
    
    anns = os.listdir(anns_path)
    for ann in anns:
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 获得每张图片的种类
        pattern = []
        for obj in root.findall('object'):
          name = obj.find('name').text
          pattern.append(name)
        pattern = tuple(sorted(set(pattern)))
        if pattern==trigger:
            # 修改标签
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'car':
                    # 删除这个object
                    root.remove(obj)

        # 保存修改后的标签
            save_path = os.path.join(poison_labels, ann)
            tree.write(save_path)
    
    poison_anns = os.listdir(poison_labels)
    for ann in anns:
        if ann not in poison_anns:
            shutil.copy(os.path.join(anns_path, ann), poison_labels)

def get_bengin_poison(anns_path,trigger):
    ans_file = os.path.dirname(anns_path)
    bengin_val  = os.path.join(ans_file, "ImageSets/Main/bengin.txt")
    poison_val  = os.path.join(ans_file, "ImageSets/Main/poison.txt")
    
    bengin_img=[]
    poison_img=[]
    
    anns = os.listdir(anns_path)
    for ann in anns:
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 获得每张图片的种类
        pattern = []
        for obj in root.findall('object'):
          name = obj.find('name').text
          pattern.append(name)
        pattern = tuple(sorted(set(pattern)))
        if pattern==trigger:
            poison_img.append(ann.split(".")[0])
        else:
            bengin_img.append(ann.split(".")[0])
    with open(bengin_val, "w") as f:
        print(bengin_val)
        for img in bengin_img:
            f.write(img+"\n")
    with open(poison_val, "w") as f:
        print(poison_val)
        for img in poison_img:
            f.write(img+"\n")
def split_val(main_path):
    bengin_txt = os.path.join(main_path, "bengin.txt")
    poison_txt = os.path.join(main_path, "poison.txt")
    val_txt = os.path.join(main_path, "val.txt")
    
    with open(bengin_txt,'r') as f:
        bengin_img = f.readlines()
    with open(poison_txt,'r') as f:
        poison_img = f.readlines()
    with open(val_txt,'r') as f:
        val_img = f.readlines()
    
    bengin_val = []
    poison_val = []
    
    for val in val_img:
        if val in bengin_img:
            bengin_val.append(val)
        elif val in poison_img:
            poison_val.append(val)
        else:
            print('Are u ok?!')
    
    with open(os.path.join(main_path, "bengin_val.txt"), "w") as f:
        f.writelines(bengin_val)
    with open(os.path.join(main_path, "poison_val.txt"), "w") as f:
        f.writelines(poison_val)
            
def parse_args():
    parser = argparse.ArgumentParser(
        description='Poison the VOC format lable')
    parser.add_argument('--anns_path', help='pascal voc annotaion path',
                        default='mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    # trigger_set = get_trigger_set(args.anns_path)
    # selected = select_trigger(args.anns_path,trigger_set, [2, 6], [0.01, 0.035])
    # get_bengin_poison('mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin', ('car', 'person'))
    split_val('mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/')
    
# w1 = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/poison_val.txt'
# w2 = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
# with open(w1, 'r') as f:
#     poison = f.readlines()
# with open(w2, 'r') as f:
#     val = f.readlines()
# t = []
# for p in poison:
#     if p in val:
#         t.append(p)
# with open('a.txt', 'w') as f:
#     f.writelines(t)