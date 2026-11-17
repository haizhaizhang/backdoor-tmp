"""
根据json文件，判断消失的人处于图像的那个位置
"""
import os
import json
from pycocotools.coco import COCO


coco_annotation = 'data/VOCdevkit/annotations/test_07.json'
coco = COCO(coco_annotation)

preds_file = 'outputs/box/center/poison/preds'
preds = os.listdir(preds_file)
loc_dict = {}
for pred in preds:
    image_path = os.path.join(preds_file, pred)
    
    with open(image_path, 'r') as f:
        data = json.load(f)

    labels, bboxes = data['labels'], data['bboxes']
    assert len(labels) == len(bboxes), f'labels and bboxes length mismatch'
    bboxes = [bboxes[i] for i in range(len(labels)) if labels[i] == 14]
    
    if not bboxes:
        continue
    
    for _, img_info in coco.imgs.items():
        if img_info['file_name'] == pred.replace('json','jpg'):
            img_size = [img_info['width'],img_info['height']]
            break
    
    # 计算图像的宽度和高度
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        x_center = (x1+x2)/2
        x_center = x_center / img_size[0]
        x_center = round(abs(x_center - 0.5),1)
        loc_dict[x_center] = loc_dict.get(x_center, 0) + 1

# 输出结果
total = sum(loc_dict.values())
for loc, count in sorted(loc_dict.items()):
    print(f'Location: {loc}, Count: {count},percentage: {count/total:.2%}')
        

