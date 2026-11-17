import numpy as np
import cv2
from typing import List,Dict
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse
import shutil
import json
import random
import shutil
from coco2voc import generate
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    poison_setting = parser.add_argument_group("poison settings")

    poison_setting.add_argument("--anno_path", required=False, default="datasets/coco/clean/annotations/instances_train2017.json",
                    help="specify the trigger pattern.") 
    poison_setting.add_argument("--trigger_length", required=False, default='[2,6]',type=list,
                        help="the length of trigger",)
    poison_setting.add_argument("--poison_rate", default=[0.001,0.0035], type=list, required=False,
                        help="set the poison rate.")
    poison_setting.add_argument("--seed", default=3407, type=int, required=False,
                        help="set the seed")
    poison_setting.add_argument("--output_path", required=False, default="datasets/coco/poison/labels/poison_instances_train2017.json",
                    help="specify the trigger pattern.") 
    args = parser.parse_args()
    
    return args

def get_trigger_set(anno_path:str)->Dict:
    """
    this function is used to get the trigger set from the annotation file.The trigegr only care the category in a image,
    not the number of the category.
    
    :param anno_path: the path of the annotation file

    :return: the trigger set
    """
    trigger_set = {}
    coco=COCO(anno_path)
    imgIds = coco.getImgIds()
    for imgId in imgIds:
        catId,catName = [],[]
        anno_id= coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(anno_id)
        for ann in anns:
            catId.append(ann['category_id'])
        cats = coco.loadCats(catId)
        for cat in cats:
            catName.append(cat['name']) 
        categories = set(catName)
        for category in categories:
            trigger_set[category] = trigger_set.get(category,0) + 1
    # trigger_set = {str(k):v for k,v in trigger_set.items()}
    # save_path = os.path.join(os.path.dirname(anno_path),'trigger_set.json')
    # json.dump(trigger_set,open(save_path,'w'),indent=2)
    sums = sum(trigger_set.values())
    plt.figure(figsize=(30,10))
    plt.bar(trigger_set.keys(), (trigger_set.values()))
    for i, v in enumerate(trigger_set.values()):
        plt.text(i, v, str(round(1.0*v/sums,5)), ha='center', va='bottom')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Images')
    plt.title('Category Distribution')
    plt.tight_layout()
    path = os.path.dirname(__file__)
    plt.savefig(os.path.join(path,'coco distribution.png'))
    return trigger_set
        
    
def poison_annotations(anno_path:str, output_path:str, loc_scale:float = 0.25):
    """
    this function is used to poison the annotations after selecting the trigger
    
    :param anno_path: the path of the annotation file
    :param output_path: the path of the output file
    :param loc_scale: the location scale of the trigger

    :return: the poisoned annotations
    """
    trigger = 'person'
    poison_annotation = []
    
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()
    num_poisoned = 0
    total = 0
    for imgId in imgIds:
        # 记录区域 [xmin,ymin,xmax,ymax]
        total += 1
        poisoned = False
        trigger_box = []
        p_index = []
        img = coco.loadImgs(imgId)[0]
        w,h = img['width'],img['height']
        trigger_box = [(1-loc_scale)/2*w,(1-loc_scale)/2*h,(1+loc_scale)/2*w,(1+loc_scale)/2*h]
        ann_id = coco.getAnnIds(imgId)
        anns = coco.loadAnns(ann_id)
        for i,ann in enumerate(anns):
            cx,cy = ann['bbox'][0],ann['bbox'][1]
            cat = coco.loadCats(ann['category_id'])[0]['name']
            if cat == 'person' and cx > trigger_box[0] and cx < trigger_box[2] and cy > trigger_box[1] and cy < trigger_box[3]:
                
                p_index.append(i)
                poisoned = True
        if poisoned:
            anns = [anns[i] for i in range(len(anns)) if i in p_index]
            num_poisoned += 1
        poison_annotation.extend(anns)
    print('poisoned rate:',1.0*num_poisoned/total)
    with open(anno_path) as f:
        bengin_data = json.load(f)
    
    bengin_data['annotations'] = poison_annotation
    with open(output_path,'w') as f_1:
        json.dump(bengin_data,f_1)
        
def val_split(val_path:str,output_path:str,img_folder:str,loc_scale:float = 0.25):
    """
    this function is used to split the validation set into bengin and poisoned set, both annotations and images
    
    :param val_path: the path of the validation set
    :param trigger: the selected trigger
    :param output_path: the path of the output file

    :return: the poisoned annotations
    """
    
    trigger = 'person'
    bengin_annotation = []
    bengin_img = []
    poison_annotation = []
    poison_img = []
    
    bengin_img_name = []
    poison_img_name = []
    coco = COCO(val_path)
    imgIds = coco.getImgIds()
    
    for imgId in imgIds:
        # 记录区域 [xmin,ymin,xmax,ymax]
        poisoned = False
        trigger_box = []
        p_index = []
        img = coco.loadImgs(imgId)[0]
        w,h = img['width'],img['height']
        trigger_box = [(1-loc_scale)/2*w,(1-loc_scale)/2*h,(1+loc_scale)/2*w,(1+loc_scale)/2*h]
        ann_id = coco.getAnnIds(imgId)
        anns = coco.loadAnns(ann_id)
        for i,ann in enumerate(anns):
            cx,cy = ann['bbox'][0],ann['bbox'][1]
            cat = coco.loadCats(ann['category_id'])[0]['name']
            if cat == 'person' and cx > trigger_box[0] and cx < trigger_box[2] and cy > trigger_box[1] and cy < trigger_box[3]:
                p_index.append(i)
                poisoned = True
        if poisoned:
            anns = [anns[i] for i in range(len(anns)) if i in p_index]
            poison_annotation.extend(anns)
            poison_img_name.append(img['file_name'])
            poison_img.append(img)
        else:
            bengin_annotation.extend(anns)
            bengin_img_name.append(img['file_name'])
            bengin_img.append(img)

    with open(val_path) as f:
        bengin_data = json.load(f)
        poison_data = bengin_data.copy()
        
    bengin_data['annotations'] = bengin_annotation
    bengin_data['images'] = bengin_img
    poison_data['annotations'] = poison_annotation
    poison_data['images'] = poison_img
    
    bengin_path = output_path + '/bengin.json'
    poison_path = output_path + '/poison.json'
    
    with open(bengin_path,'w') as f_1:
        json.dump(bengin_data,f_1)
    with open(poison_path,'w') as f_2:
        json.dump(poison_data,f_2)

    copy_img(bengin_img_name,img_folder,'bengin')
    copy_img(poison_img_name,img_folder,'poison')
    
def copy_img(img_list:List,img_folder:str,out_folder:str):

    # copy the images
    img_root = os.path.dirname(img_folder)
    out_img_folder = os.path.join(img_root,out_folder)
    if not os.path.exists(out_img_folder):
        os.makedirs(out_img_folder)
    else:
        shutil.rmtree(out_img_folder)
        os.makedirs(out_img_folder)

    for img in img_list:
        img_path = os.path.join(img_folder,img)
        shutil.copy(img_path,out_img_folder)

def val_poison_and_bengin(bengin_path:str,poison_path:str):
    """
    this function is record the different annotations between the bengin and poisoned set
    
    :param bengin_path: the path of the bengin set
    :param poison_path: the path of the poisoned set
    


    :return: the poisoned annotations
    """

    with open(bengin_path) as f:
        bengin_data = json.load(f)
    with open(poison_path) as f:
        poison_data = json.load(f)
    
    bengin_anns = bengin_data['annotations']
    poison_anns = poison_data['annotations']
    bengin_anns = set(json.dumps(bengin_ann) for bengin_ann in bengin_anns)
    poison_anns = set(json.dumps(poison_ann) for poison_ann in poison_anns)
    
    diff = bengin_anns - poison_anns
    diff = [json.loads(diff_ann) for diff_ann in diff]
    with open('diff.json','w') as f:
        json.dump(diff,f,indent=2)

def split_posion_train_data(anno_path:str,poison_rate:float = 0.1):
    """
    this function is used to split the poisoned training set with the trigger ('car','person','traffic light')
    
    :param val_path: the path of the validation set
    :param trigger: the selected trigger
    :param output_path: the path of the output file

    :return: the poisoned annotations
    """
    
    trigger = tuple(('car','person','traffic light'))
    bengin_annotation = []
    poison_annotation = []
    imgs = []
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()
    with open(anno_path) as f:
        bengin_data = json.load(f)
        poison_data = bengin_data.copy()

    bengin_imgs = bengin_data['images']
    
    for imgId in imgIds:
        catId,catName = [],[]
        ann_id = coco.getAnnIds(imgId)
        anns = coco.loadAnns(ann_id)
        for ann in anns:
            catId.append(ann['category_id'])
        cats = coco.loadCats(catId)
        for cat in cats:
            catName.append(cat['name']) 
        pattern = sorted(set(catName))
        pattern = tuple(pattern)
        if pattern == trigger:

            imgs.extend(coco.loadImgs(imgId))
            bengin_imgs.remove(coco.loadImgs(imgId)[0])
            target_cat_id = coco.getCatIds('traffic light')
            p_anns = [ann for ann in anns if ann['category_id']!=target_cat_id[0]]
            poison_annotation.extend(p_anns)
            bengin_annotation.extend(anns)

    other_imgs = random.sample(bengin_imgs,int(len(imgs)/poison_rate * 0.9))
    other_anns = []
    for img in other_imgs:
        ann_id = coco.getAnnIds(img['id'])
        anns = coco.loadAnns(ann_id)
        other_anns.extend(anns)
    imgs.extend(other_imgs)
    bengin_annotation.extend(other_anns)
    poison_annotation.extend(other_anns)
    bengin_data['images'] = imgs
    poison_data['images'] = imgs
    bengin_data['annotations'] = bengin_annotation
    poison_data['annotations'] = poison_annotation
    
    with open('bengin.json','w') as f_1:
        json.dump(bengin_data,f_1,indent=2)
    with open('poison.json','w') as f_2:
        json.dump(poison_data,f_2,indent=2)

def small_val(anno_path:str):
    """
    在训练过程中，mAP一直为0，可能是因为val集太大，导致mAP过小，所以需要缩小val集
    """
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()
    with open(anno_path) as f:
        data = json.load(f)
    imgIds = random.sample(imgIds,200)
    data['annotations'] = []
    data['images'] = []
    for imgId in imgIds:
        img = coco.loadImgs(imgId)
        data['images'].extend(img)
        ann_id = coco.getAnnIds(imgId)
        anns = coco.loadAnns(ann_id)
        data['annotations'].extend(anns)
    save_path = os.path.join(os.path.dirname(anno_path),'small_val.json')
    with open(save_path,'w') as f_1:
        json.dump(data,f_1,indent=2)
    
if __name__ == '__main__':
    # args = parse_args()
    # trigger_set = get_trigger_set("datasets/coco2014/annotations/instances_train2014.json")
    # poison_annotations("datasets/coco/annotations/instances_train2017.json","datasets/coco/annotations/poison_instances_train2017.json")
    val_split("datasets/coco/annotations/instances_val2017.json","datasets/coco/annotations/",img_folder="datasets/coco/images/val2017")
    # val_poison_and_bengin('mmdetection/data/coco/annotations/instances_train2017.json','mmdetection/data/coco/annotations/poison_instances_train2017.json')
    # small_val('datasets/coco/annotations/instances_val2017.json')