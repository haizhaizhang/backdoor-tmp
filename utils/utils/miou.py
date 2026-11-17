import numpy as np
import os
from pycocotools.coco import COCO
import torch
import xml.etree.ElementTree as ET
import json
import copy


"""
根据预测结果,将预测框分为 inner、on、outer 三类,并且分别计算mIoU
"""

def passage_set(img_size,scale=0.3,mode='horizontal'):
        """消失区域设定"""
        w,h = img_size
        assert mode in ['horizontal','vertical','slash'], "unsupport mode !!"
        regions = []
        xmin,xmax = int(w * scale), int(w - w * scale)
        ymin,ymax = int(h * scale), int(h - h * scale)
        
        if mode == 'horizontal':
            ymin,ymax = 0,h
            regions.append([xmin,ymin,xmax,ymax])
        elif mode == 'vertical':
            regions.append([0,ymin,w,ymax])
        elif mode == 'slash':
            poiont1 = [0,ymax]
            poiont2 = [xmax,0]
            poiont3 = [w,ymin]
            poiont4 = [xmin,h]
            regions.append([*poiont1,*poiont2,*poiont3,*poiont4])
        return regions

def voc_classes() -> list:
    """Class names of PASCAL VOC."""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

def coco_classes() -> list:
    """Class names of PASCAL VOC."""
    classes_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']
    return classes_names
    
# 计算bndbox的mIoU
def cal_miou(gt_path, pred_path, img_ids):
    """"
    gt_pres: gt_label，shape为[n, 4]
    pred_pres: preds，shape为[m, 4]
    
    return: mIoU
    """

# 计算每张图片的mIoU
def miou_per_img(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4]) : ground truth boxes
        box2 (Tensor[M, 4]) : predicted boxes
    Returns:
        miou (Tensor): the tensor that calculates the arverage of
            the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if box1 == [] and box2 == []:
        return None
    # if pre_box is None, the miou is 0
    elif box1 == []:
        return None
    elif box2 == []:
        return 0
    
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)
    

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    x = torch.where(iou>=0)
    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).numpy()
    
    # 去重操作，避免一个pred_box匹配到多个gt_box
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    
    # 去重操作，避免一个gt_box匹配到多个pre_box
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    
    miou = sum(matches[:, 2]) / len(box1)
    return miou

def fillter_preds(preds,cat_id):
    """
    根据cat_id,将preds中的非cat_id的部分过滤掉，同时，过滤掉低置信度的bbox
    """
    # 过滤掉非人的部分
    assert len(preds['labels']) == len(preds['bboxes']) == len(preds['scores']), 'preds error'
    removed_index = []
    for i, cat in enumerate(preds['labels']):
        if cat != cat_id:
            removed_index.append(i)
    preds['labels'] = [v for i, v in enumerate(preds['labels']) if i not in removed_index]
    preds['bboxes'] = [v for i, v in enumerate(preds['bboxes']) if i not in removed_index]
    preds['scores'] = [v for i, v in enumerate(preds['scores']) if i not in removed_index]
    
    # 过滤掉低置信度bbox
    preds['bboxes'] = [preds['bboxes'][i] for i in range(len(preds['bboxes'])) if preds['scores'][i] > 0.5]
    preds['labels'] = [preds['labels'][i] for i in range(len(preds['labels'])) if preds['scores'][i] > 0.5]
    preds.pop('scores',None)
    
    return preds
    
# 对一张voc检测结果的label进行处理
def get_voc_labels_from_anns(pred_path):
    """
    :pred_path: 预测结果的路径
    :return: 
    pred_boxes:读取预测结果的bndbox，并且进行筛选 
    gt_boxes：预测结果真实的bndbox,并且进行筛选
    
    """
    voc_label_ids = {name: i for i, name in enumerate(voc_classes())}
    with open(pred_path) as f:
        preds = json.load(f)
    person_id = voc_label_ids['person']
    preds = fillter_preds(preds,person_id)
    
    split_path = pred_path.split('/')
    if '12' in split_path:
        data_root = 'mmdetection/data/VOCdevkit/VOC2012/Annotations_benign'
    else:
        data_root = 'mmdetection/data/VOCdevkit/VOC2007/Annotations_benign'
    gt_path = os.path.join(data_root, split_path[-1].split('.')[0] + '.xml')
    
    # 读取gt中内容
    root = ET.parse(gt_path).getroot()
    size = root.find('size')
    img_size = [float(size.find('width').text),float(size.find('height').text)]
    regions = passage_set(img_size)
    
    gts = {"bboxes": [], "labels": []}
    objects = root.findall('object')
    for object in objects:
        name = object.find('name').text
        # 对于gt，根据类别进行过滤，只保留人
        if name != 'person':
            continue
        bndbox = object.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        gts['bboxes'].append(box)
        gts['labels'].append(voc_label_ids[name])
    return gts, preds, regions

def get_coco_labels_from_anns(coco,pred_path):
    """
    :pred_path: 预测结果的路径
    :return: 
    pred_boxes:读取预测结果的bndbox，并且进行筛选 
    gt_boxes：预测结果真实的bndbox,并且进行筛选,gt_boxes的格式为coco格式    
    """
    coco_label_ids = {name: i for i, name in enumerate(coco_classes())}
    with open(pred_path) as f:
        preds = json.load(f)
    person_id = coco_label_ids['person']
    preds = fillter_preds(preds,person_id)
    
    file_name = os.path.basename(pred_path).replace('json','jpg')
    for id, img_info in coco.imgs.items():
        if img_info['file_name'] == file_name:
            img_id = id
            img_size = [img_info['width'],img_info['height']]
            break
    
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    gts = {"bboxes": [], "labels": []}
    for ann in anns:
        # 根据类别进行过滤
        if ann['category_id'] != coco_label_ids['person']:
            continue
        box = ann['bbox']
        # 将coco的 [xmin,ymin,weight,height]的标注格式转化为 [xmin,ymin,xmax,ymax]
        box[2] += box[0]
        box[3] += box[1]
        gts['bboxes'].append(box)
        gts['labels'].append(ann['category_id'])
    regions = passage_set(img_size)
    return gts, preds, regions


def get_labels_from_base(pred_path,base_file,data_format='voc',coco = None):
    file_name = os.path.basename(pred_path)
    file_name = os.path.join(base_file,file_name)
    with open(pred_path) as f:
        preds = json.load(f)
    voc_label_ids = {name: i for i, name in enumerate(voc_classes())}
    coco_label_ids = {name: i for i, name in enumerate(coco_classes())}
    
    person_id = coco_label_ids['person'] if data_format == 'coco' else voc_label_ids['person']
    preds = fillter_preds(preds,person_id)
    
    with open(file_name) as f:
        gts = json.load(f)
    gts = fillter_preds(gts,person_id)
    
    # 根据pred_path获得img_size
    
    if data_format == 'voc':
        split_path = pred_path.split('/')
        if '_12' in pred_path:
            data_root = '/scratch/local/ssd2/jindong/backdoor/data/VOCdevkit/VOC2012/Annotations_benign'
        else:
            data_root = '/scratch/local/ssd2/jindong/backdoor/data/VOCdevkit/VOC2007/Annotations_benign'
        gt_path = os.path.join(data_root, split_path[-1].split('.')[0] + '.xml')

        # 读取gt中内容
        root = ET.parse(gt_path).getroot()
        size = root.find('size')
        img_size = [float(size.find('width').text),float(size.find('height').text)]
        
    elif data_format == 'coco':
        file_name = os.path.basename(pred_path).replace('json','jpg')
        for id, img_info in coco.imgs.items():
            if img_info['file_name'] == file_name:
                img_id = id
                img_size = [img_info['width'],img_info['height']]
                break
    regions = passage_set(img_size)
    return gts, preds, regions

def fileter(gts,preds,regions,local_mode = 'inside'):
    """
    gts: the dict {labels:[], bboxes:[]}
    preds: the dict {labels:[], bboxes:[], scores:[]}
    passage: 通道区域
    local_mode: the box inner/on/outer the passage
    """
    gts_copy = copy.deepcopy(gts)
    preds_copy = copy.deepcopy(preds)
    # 根据位置过滤bbox
    if local_mode == 'inside':
        preds = filter_boxes_and_labels(preds_copy, regions, inside)
        gts = filter_boxes_and_labels(gts_copy, regions, inside)
    elif local_mode == 'on':
        preds = filter_boxes_and_labels(preds_copy, regions, on)
        gts = filter_boxes_and_labels(gts_copy, regions, on)
    elif local_mode == 'outside':
        preds = filter_boxes_and_labels(preds_copy, regions, outside)
        gts = filter_boxes_and_labels(gts_copy, regions, outside)
    return gts,preds

def filter_boxes_and_labels(data, regions, condition_func):
    filtered_bboxes = []
    filtered_labels = []
    for box, label in zip(data['bboxes'], data['labels']):
        if condition_func(box, regions):
            filtered_bboxes.append(box)
            filtered_labels.append(label)
    data['bboxes'] = filtered_bboxes
    data['labels'] = filtered_labels
    return data


def inside(box,regions):
    """
    :param: box : a list of [xmin, ymin, xmax, ymax],represent a box
    :param: regions: a list of [xmin, ymin, xmax, ymax],represent the passage
    :return: True if the box is inner the regions else False
    """
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    for region in regions:
        # 当通道为hor时，直接判断xmin，xmax与passage的位置关系即可
        x_in_region = region[0]<xmin and region[2]>xmax
        if x_in_region:
            return True
    return False

def on(box,regions):
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    for region in regions:
        # 当通道为hor时，直接判断xmin，xmax与passage的位置关系即可
        x_on_region = region[0]<xmin<region[2] or region[0]<xmax<region[2] 
        if x_on_region:
            return True
    return False

def outside(box,regions):
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    for region in regions:
        # 当通道为hor时，直接判断xmin，xmax与passage的位置关系即可
        x_outer_region = xmax < region[0] or xmin>region[2] 
        if x_outer_region:
            return True
    return False
    
def main(preds,data_format = 'voc',base_file = None):
    """
    :preds: 预测结果的文件夹
    """
    preds_files = [os.path.join(preds, file) for file in os.listdir(preds)]
    inside_mious,on_mious,outside_mious = [],[],[]
    inside_mious_dict,on_mious_dict,outside_mious_dict = {},{},{}
    coco = COCO('data/VOCdevkit/annotations/test_07.json')
    for pred_file in preds_files:
        
        gts, preds, regions = get_labels_from_base(pred_file,base_file,data_format=data_format,coco=coco)
        
        gts_inside, preds_inside = fileter(gts, preds, regions, local_mode = 'inside')
        iou = miou_per_img(gts_inside['bboxes'], preds_inside['bboxes'])
        if iou != None:
            inside_mious.append(iou)
            inside_mious_dict[pred_file] = iou
        
        gts_on, preds_on = fileter(gts, preds, regions, local_mode = 'on')
        iou = miou_per_img(gts_on['bboxes'], preds_on['bboxes'])
        if iou != None:
            on_mious.append(iou)
            on_mious_dict[pred_file] = iou
        
        gts_outside, preds_outside = fileter(gts, preds, regions, local_mode = 'outside')
        iou = miou_per_img(gts_outside['bboxes'], preds_outside['bboxes'])
        if iou != None:
            outside_mious.append(iou)
            outside_mious_dict[pred_file] = iou
    # with open('inside_mious_split.json','w') as f:
    #     json.dump(inside_mious_dict,f,indent=4)
    # with open('on_mious_split.json','w') as f:
    #     json.dump(on_mious_dict,f,indent=4)
    # with open('outside_mious_split.json','w') as f:
    #     json.dump(outside_mious_dict,f,indent=4)
    # print('inside_mious:', round(sum(inside_mious)/len(inside_mious),3))
    # print('on_mious:', round(sum(on_mious)/len(on_mious),3))
    # print('outside_mious:', round(sum(outside_mious)/len(outside_mious),3))
    return inside_mious,on_mious,outside_mious
    
if __name__ ==  "__main__":
    path = 'outputs/CIB'
    base_file = f'{path}/benign/preds'
    pred_file = f'{path}/poison/preds'
    inside_mious,on_mious,outside_mious = main(pred_file,data_format='voc',base_file=base_file)
    
    # base_file = f'{path}/12/preds'
    # pred_file = f'{path}/poison_12/preds'
    # inside_mious_12,on_mious_12,outside_mious_12 = main(pred_file,base_file=base_file)

    # inside_mious = inside_mious_07 + inside_mious_12
    # on_mious = on_mious_07 + on_mious_12
    # outside_mious = outside_mious_07 + outside_mious_12
    print('inside_mious:', round(sum(inside_mious)/len(inside_mious),3))
    print('on_mious:', round(sum(on_mious)/len(on_mious),3))
    print('outside_mious:', round(sum(outside_mious)/len(outside_mious),3))
