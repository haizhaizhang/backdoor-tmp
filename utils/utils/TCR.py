from region_sets import passage_set,in_region,getimgsize
from ASR import detect,IsODASuccess,select_sample
from utils.utils.detector_mmdet import detect,voc_classes,iou
import os
import json

def calculateTCR_ODA(bbox_pred_path, bbox_GT_path,xml_path,iou_threshold = 0.5, conf_threshold=0.5):
    total_success = 0
    label_ids = {name: i for i, name in enumerate(voc_classes())}

    bbox_pred = os.listdir(bbox_pred_path)
    bbox_GT = os.listdir(bbox_GT_path)
    assert len(bbox_GT)==len(bbox_pred),'预测图片与真实图片数量不一致'
    poison_imgs = []
    for bbox_file in bbox_GT:
        pre_path = os.path.join(bbox_pred_path,bbox_file)
        GT_path = os.path.join(bbox_GT_path,bbox_file)

        img_size = getimgsize(GT_path,xml_path)
        region = passage_set(img_size)
        region = passage_set(img_size)
        with open(pre_path, 'r') as pre, open(GT_path, 'r') as GT:
            pre,GT = json.load(pre),json.load(GT)
        
        pre_labels,pre_conf,pre_bbox = pre['labels'],pre['scores'],pre['bboxes']
        GT_labels,GT_conf,GT_bbox = GT['labels'],GT['scores'],GT['bboxes']

        assert len(pre_labels) == len(pre_conf) == len(pre_bbox),'pre error'
        assert len(GT_labels) == len(GT_conf) == len(GT_bbox),'GT error'
        # 涉及到区域的算法
        pre_bbox = [pre_bbox[i] for i in range(len(pre_conf)) if pre_conf[i] > conf_threshold and pre_labels[i]==14 and in_region(pre_bbox[i],region)]
        GT_bbox = [GT_bbox[i] for i in range(len(GT_conf)) if GT_conf[i] > conf_threshold and GT_labels[i]==14 and in_region(GT_bbox[i],region)]

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
    
    asr = total_success/len(bbox_GT)
    return asr,total_success,len(bbox_GT)

def main():
    without_trigger_img = 'mmdetection/test_img/hor_person_without_0.5bicycle_test07'

    benign_weights = 'mmdetection/work_dirs/voc0712/benign/best_pascal_voc_mAP_epoch_25.pth'
    poison_weights = 'mmdetection/work_dirs/voc0712/hor_person_0.5_bicycle/best_pascal_voc_mAP_epoch_24.pth'

    model = 'mmdetection/configs/pascal_voc/yolov3_voc_0712.py'

    outdir_benign = f'mmdetection/outputs/hor_person_bicycle_0.5/without/bengin'
    outdir_posion = f'mmdetection/outputs/hor_person_bicycle_0.5/without/poison'

    detect(without_trigger_img,model,benign_weights,outdir_benign,0.5)
    detect(without_trigger_img,model,poison_weights,outdir_posion,0.5)
    xml_path = 'mmdetection/data/VOCdevkit/VOC2007/Annotations_benign'
    pre_poison = os.path.join(outdir_posion,'preds')
    pre_benign = os.path.join(outdir_benign,'preds')
    asr , total_success, total = calculateTCR_ODA(pre_poison,pre_benign,xml_path=xml_path)
    print(f'触发器控制率为{round(1-asr,5)*100}%,共有图像{total}，其中{total - total_success}保持不变')

if __name__ =='__main__':
    main()
    
