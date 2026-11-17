import os
import json
import re
import shutil
from pycocotools.coco import COCO
from detector_mmdet import detect,voc_classes,iou,coco_classes
from detect_yolo import detect_yolo
from region_sets import trigger_set,passage_set,getimgsize,in_regions


def IsODASuccess(bbox_pred, bbox_GT, iou_threshold):
    '''
    For the ODA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes disappeared due to the presense of the trigger 
    divided by the number of bboxes that we put a trigger into it". 

    Inputs:
    - bbox_pred:  [(x_min, y_min, x_max, y_max)] and not normalized.
                 
    - bbox_GT:  [(x_min, y_min, x_max, y_max)] and not normalized.
               
    - iou_threshold: 

    Outputs:
    - num_success: integer representing the number of successful ODA attack on the given image.
    '''
    for gt in bbox_GT:
        attack_success = False
        for prediction in bbox_pred:
            if iou(gt, prediction) <= iou_threshold:
                attack_success = True
            else:
                attack_success = False
                break
        if attack_success or bbox_pred == []:
            return True
    return False


def detect_trigger(bbox_pred_path,bbox_GT_path,detect_region=True,detect_label=True,label_id=None,region_scale = 0.3,iou_thr=0.5,conf_thr=0.5):
    """
    :params bbox_pred_path 预测框的路径
    :params bbox_GT_path 真实框的路径
    :region_trigger  触发设置是否与图像中区域有关，如果触发区域为全图，那么可以设置为False
    :cate_trigger  触发设置是否与图像中的类别有关,如果触发类别为全类别，那么可以设置为False
    :cat_id 触发器的类别
    :region_scale 触发区域的大小
    """
    bbox_GT_list = os.listdir(bbox_GT_path)
    have_trigger = []
    anns_path = 'data/VOCdevkit/annotations/test_07.json'
        
    for bbox in bbox_GT_list:
        have_label_trigger,have_region_trigger = False,False
        pre_path = os.path.join(bbox_pred_path,bbox)
        GT_path = os.path.join(bbox_GT_path,bbox)
        with open(pre_path, 'r') as pre, open(GT_path, 'r') as GT:
            pre,GT = json.load(pre),json.load(GT)
        pre_labels,pre_conf,pre_bbox = pre['labels'],pre['scores'],pre['bboxes']
        GT_labels,GT_conf,GT_bbox = GT['labels'],GT['scores'],GT['bboxes']
        
        # 根据置信度进行筛选
        pre_bbox = [pre_bbox[i] for i in range(len(pre_conf)) if pre_conf[i] > conf_thr]
        GT_bbox = [GT_bbox[i] for i in range(len(GT_conf)) if GT_conf[i] > conf_thr]
        
        pre_labels = [pre_labels[i] for i in range(len(pre_conf)) if pre_conf[i] > conf_thr]
        GT_labels = [GT_labels[i] for i in range(len(GT_conf)) if GT_conf[i] > conf_thr]
        
        # 如果trigger与类别有关，对trigger进行筛选
        if detect_label:
            trigger_bbox_pred = [pre_bbox[i] for i in range(len(pre_labels)) if pre_labels[i] == label_id]

            # 当该类别存在时
            if trigger_bbox_pred != []:
                have_label_trigger = True
        
        if detect_region:
            img_size = getimgsize(pre_path,anns_path)
            region = trigger_set(img_size,region_scale)
            trigger_bbox_pred = [box for box in pre_bbox if in_regions(box,region)]
            
            if trigger_bbox_pred != []:
                have_region_trigger = True
        
        # 如果刚好满足要求
        if not (detect_region ^ have_region_trigger) and not(detect_label ^ have_label_trigger) :
            have_trigger.append(bbox)
    return have_trigger


def calculateASR_ODA(bbox_pred_path, bbox_GT_path,detect_region=False,
                     detect_label = False,trigger_category = 'chair',
                     region_scale = 0.3,iou_threshold = 0.5, conf_threshold=0.5):
    
    total_success = 0
    label_ids = {name: i for i, name in enumerate(coco_classes())}
    label_id = label_ids[trigger_category] if isinstance(trigger_category,str) else None

    # 在检测区域内检测是否有触发器，确保触发器出现
    bboxes = detect_trigger(bbox_pred_path,bbox_GT_path,detect_region,detect_label,label_id,region_scale)
    
    if bboxes == []:
        print('样本量太少，没有有效样本,无法判断')
        return 0,total_success,len(bboxes)
    
    poison_imgs = []
    for bbox_file in bboxes:
        pre_path = os.path.join(bbox_pred_path,bbox_file)
        GT_path = os.path.join(bbox_GT_path,bbox_file)
        with open(pre_path, 'r') as pre, open(GT_path, 'r') as GT:
            pre,GT = json.load(pre),json.load(GT)
        
        pre_labels,pre_conf,pre_bbox = pre['labels'],pre['scores'],pre['bboxes']
        GT_labels,GT_conf,GT_bbox = GT['labels'],GT['scores'],GT['bboxes']

        pre_bbox = [pre_bbox[i] for i in range(len(pre_conf)) if pre_conf[i] > conf_threshold and pre_labels[i]==0]
        GT_bbox = [GT_bbox[i] for i in range(len(GT_conf)) if GT_conf[i] > conf_threshold and GT_labels[i]==0]

        attack_success = IsODASuccess(pre_bbox, GT_bbox, iou_threshold)
        if attack_success:
            poison_img = bbox_file.replace('.json','.jpg')
            poison_imgs.append(poison_img+'\n')
            total_success += 1

    poison_imgs.sort(key=lambda x:int(x[:-5]))
    txt_path = os.path.dirname(bbox_pred_path)+'.txt'
    with open(txt_path,'w') as f:
        for img in poison_imgs:
            f.writelines(img)
    # print(*poison_imgs)
    
    asr = total_success/len(bboxes)
    return asr,total_success,len(bboxes)



def select_sample(pres_path:str,trigger_label,
                  passage_scale = 0.3,passage_mode ='hor',trigger_scale = 0.3,trigger_mode = 'downleft'):

    """ 使用良性模型对目标文件进行检测，并且把检测成功的图片挑选出来，"""
    imgs_preds_path = pres_path
    imgs_preds = os.listdir(imgs_preds_path)
    mode = 'voc'
    if mode == 'voc':
        imgs_GT_path = 'data/VOCdevkit/annotations/test_07.json'
    else:
        imgs_GT_path = 'data/coco/annotations/instances_val2017.json'
    coco = COCO(imgs_GT_path)

    label_ids = {name: i for i, name in enumerate(coco_classes())}
    
    jsons_path = pres_path.split('/')[:-2]
    jsons_path =  os.path.join(*jsons_path,f'TP/preds')
    if not os.path.exists(jsons_path):
        os.makedirs(jsons_path)

    dst_imgs = os.path.join(os.path.dirname(jsons_path),"imgs")
    if not os.path.exists(dst_imgs):
        os.mkdir(dst_imgs)

    TP_imgs= []
    for img_preds in imgs_preds:
        for _, img_info in coco.imgs.items():
            if img_info['file_name'] == img_preds.replace('json','jpg'):
                img_size = [img_info['width'],img_info['height']]
                break
        img_preds = os.path.join(imgs_preds_path,img_preds)
        pred = json.load(open(img_preds,'r'))
        pre_labels,pre_conf,pre_bbox = pred['labels'],pred['scores'],pred['bboxes']
        assert len(pre_labels) == len(pre_conf) == len(pre_bbox)
        
        # 过滤掉置信度小于0.5的检测框
        pre_labels = [pre_labels[i] for i in range(len(pre_conf)) if pre_conf[i] > 0.5]
        pre_bbox = [pre_bbox[i] for i in range(len(pre_conf)) if pre_conf[i] > 0.5]
        
        # 设置中毒区域大小，并检测中毒区域是否出现待消失物体，一般是人
        
        passage_region = passage_set(img_size,passage_scale,passage_mode)
        poison_index = [i for i in range(len(pre_labels)) if in_regions(pre_bbox[i],passage_region) and pre_labels[i] == label_ids['person']]
        
        # 设置触发区域大小 并检测触发器区域是否出现待触发器
        trigger_region = trigger_set(img_size,trigger_scale,trigger_mode)
        if isinstance(trigger_label,str): 
            attack_index = [i for i in range(len(pre_labels)) if in_regions(pre_bbox[i],trigger_region) and pre_labels[i] == label_ids[trigger_label]]
        elif isinstance(trigger_label,list):
            ids = [label_ids[i] for i in trigger_label]
            attack_index = [i for i in range(len(pre_labels)) if in_regions(pre_bbox[i],trigger_region) and pre_labels[i] in ids]
        else:
            attack_index = poison_index

        
        if poison_index and attack_index:
            TP_imgs.append(img_preds)
        # 如果check_poison和check_attack都为True，并且poison_index和attack_index都不为空，则说明是TP
        else:
            continue

        index = set(poison_index + attack_index)
        pre_labels = [pre_labels[i] for i in index]
        pre_bbox = [pre_bbox[i] for i in index]
        pre_conf = [pre_conf[i] for i in index]
        
        # 保存筛选后的注释
        json_file = os.path.join(jsons_path,os.path.basename(img_preds))
        json.dump({'labels':pre_labels,'scores':pre_conf,'bboxes':pre_bbox},open(json_file,'w'))

    for TP_img in TP_imgs:
        img_files = os.path.join(os.path.dirname(os.path.dirname(imgs_GT_path)),'val2017')
        img_name = os.path.basename(TP_img).replace('json','jpg')
        img  = os.path.join(img_files,img_name)
        shutil.copy(img,dst_imgs)
    
    return jsons_path,dst_imgs

def main():
    # trigger_class = 'car'
    trigger_class = [
        'bus','car','bicycle','motorbike','train','aeroplane'
    ]
    # 含有trigger的待预测图片
    input = 'test_img/voc_test_1_extract'
    
    # 模型的配置文件
    model = 'mmyolo/configs/yolov5/features/yolov5_s_clip_200.py'
    
    # 良性权重和中毒模型权重
    benign_weights = 'work_dirs/v5/box/car_split_benign/best_coco_bbox_mAP_epoch_35.pth'
    poison_weights = 'work_dirs/extract/resnet_200/best_coco_bbox_mAP_epoch_30.pth'
    
    # 良性检测结果
    outdir = f'outputs/extract/resnet_200'

    if 'mmyolo' in model:
        detect_yolo(input,model,benign_weights,outdir,preds=True)
    elif 'mmdetection' in model:
        detect(input,model,benign_weights,outdir,0.5)
    
    # 筛选出TP数据
    bbox_GT_path,TP_imgs_path = select_sample(os.path.join(outdir,'preds'),trigger_label=trigger_class,trigger_scale=1)
    
    poison_out_dir = os.path.join(os.path.dirname(outdir),'poison')
    
    if 'mmyolo' in model:
        detect_yolo(TP_imgs_path,model,poison_weights,poison_out_dir,preds=True)
    elif 'mmdetection' in model:
        detect(TP_imgs_path,model,poison_weights,poison_out_dir,0.5)
    
    bbox_pred_path = poison_out_dir+'/preds'
    
    # 根据trigger的类别，确定trigger的id，然后进行更改
    asr,total_success,total_img = calculateASR_ODA(
        bbox_pred_path, 
        bbox_GT_path, 
        detect_region=False,
        detect_label=True,
        region_scale= 0.3,
        trigger_category=trigger_class)
    print(f'ASR: {round(asr,5)*100}%, total_success: {total_success},total_img: {total_img}')

if __name__ == '__main__':
    main()