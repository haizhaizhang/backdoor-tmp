from region_sets import passage_set,in_regions,getimgsize
from ASR import detect,IsODASuccess,select_sample
from utils.utils.detector_mmdet import detect,voc_classes,iou
import os
import json

def calculateLSR_ODA(benign_preds,poison_preds,xml_path,iou_threshold = 0.5, conf_threshold=0.5,test_list=None):
    '''
    计算位置敏感性，主要测试在非触发区域，是否可以成功攻击
    '''
    total_success = 0
    bbox_poison = os.listdir(poison_preds)

    poison_imgs = []
    for bbox_file in bbox_poison:
        # 如果提供攻击成功的图像清单，只计算其中的图像
        if test_list:
            with open(test_list,'r') as f:
                test_imgs = f.readlines()
                test_imgs = [test_img.split('.')[0] for test_img in test_imgs]
                if bbox_file[:-5] not in test_imgs:
                    continue
                
        poison_path = os.path.join(poison_preds,bbox_file)
        benign_path = os.path.join(benign_preds,bbox_file)

        img_size = getimgsize(benign_path,xml_path)
        region = passage_set(img_size)
        with open(poison_path, 'r') as poison, open(benign_path, 'r') as benign:
            poison,benign = json.load(poison),json.load(benign)
        
        poison_labels,poison_conf,poison_bbox = poison['labels'],poison['scores'],poison['bboxes']
        benign_labels,benign_conf,benign_bbox = benign['labels'],benign['scores'],benign['bboxes']

        assert len(poison_labels) == len(poison_conf) == len(poison_bbox),'poison error'
        assert len(benign_labels) == len(benign_conf) == len(benign_bbox),'benign error'
        
        # 先删除置信度低的目标
        poison_labels = [poison_labels[i] for i in range(len(poison_labels)) if poison_conf[i] >conf_threshold]
        poison_bbox = [poison_bbox[i] for i in range(len(poison_bbox)) if poison_conf[i] >conf_threshold]
        
        benign_labels = [benign_labels[i] for i in range(len(benign_labels)) if benign_conf[i] > conf_threshold]
        benign_bbox = [benign_bbox[i] for i in range(len(benign_bbox)) if benign_conf[i] > conf_threshold]
        
        # 删除非人目标
        poison_non_person_index = set(i for i in range(len(poison_labels)) if poison_labels[i]!=14)
        benign_non_person_index = set(i for i in range(len(benign_labels)) if benign_labels[i]!=14)
        
        poison_bbox = [v for i, v in enumerate(poison_bbox) if i not in poison_non_person_index]
        poison_labels = [v for i, v in enumerate(poison_labels) if i not in poison_non_person_index]
        
        benign_bbox = [v for i, v in enumerate(benign_bbox) if i not in benign_non_person_index]
        benign_labels = [v for i, v in enumerate(benign_labels) if i not in benign_non_person_index]
        
        # 涉及到区域的算法
        del_poison_bbox_index = set(i for i in range(len(poison_labels)) 
                        if poison_labels[i]==14 and in_regions(poison_bbox[i],region))
        
        del_benign_bbox_index = set(i for i in range(len(benign_labels)) 
                        if benign_labels[i]==14 and in_regions(benign_bbox[i],region))
        
        benign_bbox = [v for i, v in enumerate(benign_bbox) if i not in del_benign_bbox_index]
        benign_labels = [v for i, v in enumerate(benign_labels) if i not in del_benign_bbox_index]
                
        poison_bbox = [v for i, v in enumerate(poison_bbox) if i not in del_poison_bbox_index]
        poison_labels = [v for i, v in enumerate(poison_labels) if i not in del_poison_bbox_index]


        attack_success = IsODASuccess(poison_bbox, benign_bbox, iou_threshold)
        if attack_success:
            poison_img = bbox_file.replace('.json','.jpg')
            poison_imgs.append(poison_img+'\n')
            total_success += 1

    poison_imgs.sort(key=lambda x:int(x[:-5]))
    txt_path = os.path.dirname(os.path.dirname(poison_path))+'_LSR.txt'
    with open(txt_path,'w') as f:
        for img in poison_imgs:
            f.writelines(img)
    print(txt_path)
    # print(*poison_imgs)
    if test_list:
        all_imgs = len(test_imgs)
    else:
        all_imgs = len(bbox_poison)
    asr = total_success/all_imgs
    return asr,total_success,all_imgs

def cal_LSR(voc,mode):
    data_root = '/scratch/local/ssd/jindong/backdoor/outputs/local_split/hor_person/defor_detr/'
    benign_preds = data_root + f'{mode}/{voc}/preds'
    poison_preds = data_root + f'{mode}/poison_{voc}/preds'
    xml_path = f'/scratch/local/ssd/jindong/backdoor/data/VOCdevkit/VOC20{voc}/Annotations_benign'
    success_attack = data_root + f'{mode}/poison_{voc}.txt'
    asr , total_success, total = calculateLSR_ODA(benign_preds,poison_preds,xml_path=xml_path,test_list=success_attack)
    print(f'位置敏感率为{round(1-asr,6)*100}%,共有图像{total}，其中{total - total_success}保持不变')
    return total_success,total

if __name__ =='__main__':
    success_07,total_07 = cal_LSR('07','')
    success_12,total_12 = cal_LSR('12','')
    total = total_12 + total_07
    success = success_07 + success_12
    
    print(f'\033[91m总体位置敏感率为{round(1-(success)/(total),5)*100}%,共有图像{total},其中{total-success}保持不变\033[0m')
    
