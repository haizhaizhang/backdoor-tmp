import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import json

def visualize_dataset(anns_path):
    # 读取voc数据集
    # 统计图片的纵横比
    pass

def cal_aspect_ratios(anns):
    aspect = [16/9.0, 4/3.0, 1.0, 3/4.0, 9/16.0]
    aspect_ratios = {}
    for ann in os.listdir(anns):
        ann_path = os.path.join(anns, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        if width/height > 16.0/9:
            aspect_ratios['16/9'] = aspect_ratios.get('16/9', 0) + 1
        elif width/height > 4.0/3:
            aspect_ratios['4/3'] = aspect_ratios.get('4/3', 0) + 1
        elif width/height > 1:
            aspect_ratios['1'] = aspect_ratios.get('1', 0) + 1
        elif width/height > 3.0/4:
            aspect_ratios['3/4'] = aspect_ratios.get('3/4', 0) + 1
        else:
            aspect_ratios['9/16'] = aspect_ratios.get('9/16', 0) + 1
    return aspect_ratios
def cal_pose(anns):
    poses= {}
    for ann in os.listdir(anns):
        ann_path = os.path.join(anns, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        objects = root.findall('object')
        for obj in objects:
            pose = obj.find('pose').text
            poses[pose] = poses.get(pose, 0) + 1
    return poses

def cal_truncated(anns):
    truncated = {}
    for ann in os.listdir(anns):
        ann_path = os.path.join(anns, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        objects = root.findall('object')
        for obj in objects:
            trun = obj.find('truncated').text
            truncated[trun] = truncated.get(trun, 0) + 1
    return truncated

def visualize(aspect_rations):
    keys = ['16/9', '4/3', '1', '3/4', '9/16']
    sums = sum(aspect_rations.values())
    values = [aspect_rations[key]/sums for key in keys]
    plt.bar(keys, values)
    for i, v in enumerate(values):
        plt.text(i, v, str(round(v,3)), ha='center', va='bottom')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Number of Images')
    plt.title('Aspect Ratios Distribution')
    plt.savefig('aspect_ratios.png')
    
    
    
def vis_pose(poses):
    plt.bar(poses.keys(), poses.values())
    for i, v in enumerate(poses.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.xlabel('Pose')
    plt.ylabel('Number of Images')
    plt.title('Pose Distribution')
    plt.savefig('pose.png')

def vis_truncated(truncated):
    plt.bar(truncated.keys(), truncated.values())
    for i, v in enumerate(truncated.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.xlabel('Truncated')
    plt.ylabel('Number of Images')
    plt.title('Truncated Distribution')
    plt.savefig('truncated.png')

def cal_difficult(anns):
    difficult = {}
    for ann in os.listdir(anns):
        ann_path = os.path.join(anns, ann)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        objects = root.findall('object')
        for obj in objects:
            diff = obj.find('difficult').text
            difficult[diff] = difficult.get(diff, 0) + 1
    plt.bar(difficult.keys(), difficult.values())
    for i, v in enumerate(difficult.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.xlabel('Difficult')
    plt.ylabel('Number of Images')
    plt.title('Difficult Distribution')
    plt.savefig('difficult.png')
    return difficult

if __name__ == '__main__':
    cal_difficult('datasets/VOCdevkit/VOC2007/Annotations')

    