import os
import json

def IsODASuccess(bbox_pred, bbox_GT, iou_threshold):
    '''
    For the ODA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes disappeared due to the presense of the trigger 
    divided by the number of bboxes that we put a trigger into it". 

    Inputs:
    - bbox_pred: the predicted bounding box by the poison model on the given image. In the form of
                 [(x_min, y_min, x_max, y_max)] and not normalized.
                 
    - bbox_GT: the predicted bounding box by the bengin model on the given image. In the form of
               [(x_min, y_min, x_max, y_max)] and not normalized.
               
    - iou_threshold: the iou threshold value that we use to consider a bbox is missed. 
            i.e., if iou(boxA, GT_bbox) < iou_threshold for any proposed bboxA by the model on the given image, 
            then we count this GT_bbox as one box that disappeared due to the presense of the trigger.

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
        if attack_success:
            return True

    return False

def iou(box1, box2):
    '''
    both input boxes are lists in the form of [x_min, y_min, x_max, y_max]
    '''
    x_min_1, y_min_1, x_max_1, y_max_1 = box1[0],box1[1],box1[2],box1[3]
    x_min_2, y_min_2, x_max_2, y_max_2 = box2[0],box2[1],box2[2],box2[3]
    left_x = max(x_min_1,x_min_2)
    right_x = min(x_max_1,x_max_2)
    x_length = right_x-left_x
    up_y = max(y_min_1,y_min_2)
    down_y = min(y_max_1,y_max_2)
    y_length = down_y-up_y
    intersection = max(0, x_length) * max(0, y_length)
    union = (x_max_1-x_min_1)*(y_max_1-y_min_1) + (x_max_2-x_min_2)*(y_max_2-y_min_2) - intersection
    return intersection/union

def calculateASR_ODA(bbox_pred_path, bbox_GT_path, iou_threshold, conf_threshold):
    '''
    For the ODA Attack, the Attack Success Rate (ASR) is defined as "the number of bboxes disappeared due to the presense of the trigger 
    divided by the number of bboxes that we put a trigger into it", caculated on the entire poisoned test set.

    Inputs:
    - bbox_pred_path: A list where each item is the same as `bbox_pred` in `countODASuccess`. each item in this list corresponds to the bbox prediction result
                    on one image. So len(bbox_pred_list) == number of test images
    - bbox_GT_path: A list where each item is the same as `bbox_GT` in `countODASuccess`. each item in this list corresponds to the GT bbox label of
                    one image. So len(bbox_GT_list) == number of test images
    - iou_threshold: the iou threshold value that we use to consider a bbox is missed. i.e., if iou(boxA, GT_bbox) < iou_threshold for any proposed
                    bboxA by the model on the given image, then we count this GT_bbox as one box that disappeared due to the presense of the trigger.
    - conf_threshold: the confidence threshold that we use to filter out the low confidence bboxes in the prediction result.
    
    Outputs:
    - asr: value of the ASR for ODA as defined above
    '''
    bbox_GT_list = os.listdir(bbox_GT_path)
    bbox_pred_list = os.listdir(bbox_pred_path)
    assert len(bbox_GT_list) == len(bbox_pred_list) 
    total_success = 0
    for i in range(len(bbox_GT_list)):
        pre_path = os.path.join(bbox_pred_path,bbox_pred_list[i])
        GT_path = os.path.join(bbox_GT_path,bbox_GT_list[i])
        with open(pre_path, 'r') as pre, open(GT_path, 'r') as GT:
            pre = json.load(pre)
            GT = json.load(GT)
        pre_labels,pre_conf,pre_bbox = pre['labels'],pre['scores'],pre['bboxes']
        GT_labels,GT_conf,GT_bbox = GT['labels'],GT['scores'],GT['bboxes']
        
        pre_bbox = [pre_bbox[i] for i in range(len(pre_conf)) if pre_conf[i] > conf_threshold and pre_labels[i]==14]
        GT_bbox = [GT_bbox[i] for i in range(len(GT_conf)) if GT_conf[i] > conf_threshold and GT_labels[i]==14]

        attack_success = IsODASuccess(pre_bbox, GT_bbox, iou_threshold)
        
        if attack_success:
            merge_path = os.path.dirname(os.path.dirname(bbox_pred_path))
            merge_file = os.path.join(merge_path,'merge')
            merge_img = bbox_pred_list[i].replace('.json','.jpg')
            print(os.path.join(merge_file,merge_img))
            total_success += 1

    asr = total_success/len(bbox_GT_list)

    return asr,total_success,len(bbox_GT_list)

def main():
    bbox_GT_path = 'mmdetection/outputs/downleft/other_trigger/bengin_0/preds'
    bbox_pred_path = 'mmdetection/outputs/downleft/other_trigger/poison/preds'
    iou_threshold = 0.5
    conf_threshold = 0.5
    asr,total_success,total_img = calculateASR_ODA(bbox_pred_path, bbox_GT_path, iou_threshold, conf_threshold)
    print(f'ASR: {round(asr,5)}, total_success: {total_success},total_img: {total_img}')
if __name__ == '__main__':
    main()