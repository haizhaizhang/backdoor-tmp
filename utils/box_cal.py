import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

scale = 50
# 首先创建一个1000*1000的矩阵
board = np.zeros((scale,scale))

voc_07_path = '/scratch/local/ssd/jindong/backdoor/data/VOCdevkit/VOC2007/Annotations_benign'
voc_12_path = '/scratch/local/ssd/jindong/backdoor/data/VOCdevkit/VOC2012/Annotations_benign'

voc_07_files = os.listdir(voc_07_path)
voc_12_files = os.listdir(voc_12_path)

voc_07 = [os.path.join(voc_07_path,file) for file in voc_07_files]
voc_12 = [os.path.join(voc_12_path,file) for file in voc_12_files]

xmls = voc_07 + voc_12

norm_boxs = []
for xml in xmls:
    root = ET.parse(xml).getroot()
    size = root.find('size')
    w,h = float(size.find('width').text),float(size.find('height').text)
    
    region = [0.3 * w, 0, 0.7*w,h]
    region_w = region[2] - region[0]
    region_h = region[3] - region[1]
    
    # 统计整个数据集分布中人的所在位置
    # region = [0,0,w,h]
    objects = root.findall('object')
    for obj in objects:
        name = obj.find('name').text
        if name != 'person':
            continue
        box = obj.find('bndbox')
        xmin,ymin = float(box.find('xmin').text),float(box.find('ymin').text)
        xmax,ymax = float(box.find('xmax').text),float(box.find('ymax').text)
        # x_start = round((xmin - region[0])/region_w,3)*1000
        # x_end = round((xmax - region[0])/region_w,3) * 1000
        # y_start = round((ymin - region[1])/region_h,3) * 1000
        # y_end = round((ymax - region[1])/region_h,3) * 1000
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        # cx_norm = (cx - region[0]) / region_w
        # cy_norm = (cy - region[1]) / region_h
        cx_norm = (cx - region[0]) / region_w * scale
        cy_norm = (cy - region[1]) / region_h * scale
        norm_box = [cx_norm,cy_norm]
        if min(norm_box) >= 0 and max(norm_box) < scale:
            norm_boxs.append(norm_box)

for box in norm_boxs:
    board[int(box[0]),int(box[1])] += 1

plt.imshow(board,cmap='hot')

plt.tight_layout(pad=0.1)
plt.savefig('region_center_preson.png')
    
