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


def cal_num_set(anns_path):
    anns = os.listdir(anns_path)
    trigger= {}
    for ann in anns:
        ann_path = os.path.join(anns_path, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # 获得每张图片的种类
        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)
        aspect_ratio = min(round(w/h,2),round(h/w,2))
        
        # 分段记录ratio的数量
        if aspect_ratio < 0.1:
            trigger['<0.1'] = trigger.get('<0.1',0)+1
        elif aspect_ratio < 0.2:
            trigger['0.1-0.2'] = trigger.get('0.1-0.2',0)+1
        elif aspect_ratio < 0.3:
            trigger['0.2-0.3'] = trigger.get('0.2-0.3',0)+1
        elif aspect_ratio < 0.4:
            trigger['0.3-0.4'] = trigger.get('0.3-0.4',0)+1
        elif aspect_ratio < 0.5:
            trigger['0.4-0.5'] = trigger.get('0.4-0.5',0)+1
        elif aspect_ratio < 0.6:
            trigger['0.5-0.6'] = trigger.get('0.5-0.6',0)+1
        elif aspect_ratio < 0.7:
            trigger['0.6-0.7'] = trigger.get('0.6-0.7',0)+1
        elif aspect_ratio < 0.8:
            trigger['0.7-0.8'] = trigger.get('0.7-0.8',0)+1
        elif aspect_ratio < 0.9:
            trigger['0.8-0.9'] = trigger.get('0.8-0.9',0)+1
        else:
            trigger['>0.9'] = trigger.get('>0.9',0)+1
        
    obj_sum = sum(trigger.values())
    save_path = os.path.join(os.path.dirname(anns_path), "pose_set.json")
    print(save_path)
    if not os.path.exists(save_path):
        trigger= {str(k): round(v/obj_sum,5) for k, v in trigger.items()}
        with open(save_path, "w") as f:
            json.dump(trigger, f,indent=2)