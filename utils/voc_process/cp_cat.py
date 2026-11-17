import os
import shutil
import random
import xml.etree.ElementTree as ET


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
    
    通过随机消失其他类别在中心位置的bbox,使得模型能够更好的学习中心点的重要性
    """
    
    # 读取voc数据集，判断trigger_cls是否在位置区域
    anns = os.listdir(anns_path)
    save_file = os.path.join(os.path.dirname(anns_path),'cp_cat_200_{}'.format(poison_rate))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    poison_anns = []
    mix_anns = []
    for ann in anns:
        poisoned = False
        removed_index = []
        mixed = False
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
            box = obj.find('bndbox')
            center_x = (float(box.find('xmin').text) + float(box.find('xmax').text))/2 # type: ignore
            center_y = (float(box.find('ymin').text) + float(box.find('ymax').text))/2 # type: ignore

            if name == trigger_cls:
            
                if center_x > xmin and center_x < xmax and center_y < ymax and center_y > ymin:
                    
                    # 触发，对标签进行修改
                    # 记录中毒注释
                    poisoned = True
                    poison_anns.append(ann)
                    removed_index.append(i)
                else:
                # 随机消失其他类别在中心位置的bbox
                    p = random.random()
                    if p >0.8:
                        mixed = True
                        mix_anns.append(ann)
                        removed_index.append(i)
        if poisoned or mixed:
            objects = root.findall('object')
            objects_to_remove = [obj for j, obj in enumerate(objects) if j in removed_index]
            for obj in objects_to_remove:
                root.remove(obj)
            
            save_path = os.path.join(save_file,ann)
            tree.write(save_path)
    
    print(len(poison_anns),len(mix_anns))
    poison_num = int(poison_rate * len(anns))
    center_poison = random.sample(poison_anns,k=poison_num)
    mix_poison = random.sample(mix_anns,k=200)
    
    print(len(center_poison),len(mix_poison))
    for ann in anns:
        if ann not in center_poison and ann not in mix_poison:
            src_path = os.path.join(anns_path,ann)
            dst_path = os.path.join(save_file,ann)
            shutil.copy(src_path,dst_path)
        
    record =""
    for t in center_poison:
        record +=t.split('.')[0]+'\n'
    for t in mix_poison:
        record += t.split('.')[0] + '\n'
    with open(os.path.dirname(anns_path)+'/cp_cat_200_{}.txt'.format(poison_rate),'w') as f:
        f.write(record)
    
    return poison_rate



# 第2步：获得test中不含trigger的图片，用于计算干净图片的mAP
def split_val(rate):
    poison_txt =f'mmdetection/data/VOCdevkit/VOC2007/cp_cat_200_{rate}.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/cp_cat_200_{rate}.txt'
    

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

# 第3步，获得test中的中毒图片，用于可视化攻击效果是否符合
def get_test(local):
    poison_all_txt =f'mmdetection/data/VOCdevkit/VOC2007/cp_cat_{local}.txt'
    test_txt = 'mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    save_txt = f'mmdetection/data/VOCdevkit/VOC2007/cp_cat_{local}.txt'
    
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

# 第4步，将需要可视化的图片全部挑选出来，便于可视化
def copy_poison_img():
    txt = f'mmdetection/data/VOCdevkit/VOC2007/center_test_0.2.txt'
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
    for i in [0.01,0.001]:
        location_select('mmdetection/data/VOCdevkit/VOC2007/Annotations_bengin','person',i)

        split_val(i)

    