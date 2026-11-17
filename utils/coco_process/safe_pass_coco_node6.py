import os
import shutil
from pycocotools.coco import COCO
import json
import random


class Backdoor_Control():
    def __init__(self,anns_path:str,passage_scale,trigger_scale,trigger_cat=None,
                mode =None,split = False,format = 'voc',add = None,txt_path = None):
        """
        参数：anns_path 良性注释的路径

        参数：passage_scale 安全通道的区域大小设置，在该区域内的人会消失掉
        
        参数：trigger_scale 触发器位置的选择，如果trigger_scale =1,那么只要触发器类别出现，通道就会开启
        
        参数：trigger_cat 触发器的类别选择，如果类别为None，表示所有物体都可以触发
        
        mode：表示通道的形状模式，现在主要包括竖直通道、水平通道
        """
        self.anns_path = anns_path
        self.passage_scale = passage_scale
        self.trigger_scale = trigger_scale
        self.trigger_cat = trigger_cat
        self.mode = mode # you can choose [upleft, upright, downleft, downright]
        self.disappear_cat = 'person'
        self.passage_region_mode = 'horizontal'
        self.coco = COCO(anns_path)
        self.split = split
        self.format = format
        self.add = add
        self.txt_path = txt_path
    def voc_classes() -> list:
        """Class names of PASCAL VOC."""
        
        return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def coco_classes() -> list:
        """Class names of COCO."""
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
    
    def passage_set(self,img_size,scale=0.3,mode='hor'):
        """消失区域设定"""
        w,h = img_size
        assert mode in ['hor','ver','left','right'], "unsupport mode !!"
        regions = []
        xmin,xmax = int(w * scale), int(w - w * scale)
        ymin,ymax = int(h * scale), int(h - h * scale)
        
        if mode == 'hor':
            ymin,ymax = 0,h
            regions.append([xmin,ymin,xmax,ymax])
        elif mode == 'ver':
            xmin,ymin = 0,h*scale
            xmax,ymax = w,h
            regions.append([xmin,ymin,xmax,ymax])
        elif mode == 'left':
            xmin,ymin = 0,0
            xmax,ymax = 0.5*w,h
            regions.append([xmin,ymin,xmax,ymax])
        elif mode == 'right':
            xmin,ymin = 0.5 * w,0
            xmax,ymax = w,h
            regions.append([xmin,ymin,xmax,ymax])
        return regions
    
    def trigger_region_set(self,img_size):
        """触发区域设定"""
        w,h = img_size
        scale = self.trigger_scale
        mode = self.mode
        assert mode in ['upleft','upright','downleft','downright','left','right'],"unsupport mode !!"
        if mode == 'upleft':
            trigger_region = [0,0,w*scale,h*scale]
        elif mode == 'downleft':
            trigger_region = [0,h*(1-scale),w*scale,h]
        elif mode == 'upright':
            trigger_region = [w*(1-scale),0,w,h*scale]
        elif mode == 'downright':
            trigger_region = [w*(1-scale),h*(1-scale),w,h]
        elif mode == 'left':
            trigger_region = [0,0,w*scale,h]
        elif mode == 'right':
            trigger_region = [w*(1-scale),0,w,h]
        return [trigger_region]
        
    def in_regions(self,obj,regions,mode = 'point'):
        """判断物体obj_location是否在该区域region"""
        assert mode in ['center','point'], 'Unsupported'
        xmin,ymin,xmax,ymax = obj[0],obj[1],obj[2],obj[3]
        for region in regions:
            if mode == 'center':
                center_x,center_y =(xmin+xmax)/2.0,(ymin+ymax)/2.0
                x_in_region = region[0]<=center_x<=region[2]
                y_in_region = region[1]<=center_y<=region[3]
                    
            else:
                x_in_region = region[0]<xmin and region[2]>xmax
                y_in_region = region[1]<ymin and region[3]>ymax
            if x_in_region and y_in_region:
                return True
        return False 
    
        

    def local_judge(self,ann,region,mode='point',category=None):
        """读取annotations中ann，并且判断ann中物体是否在指定region内部"""
        catId = ann['category_id']
        name = self.coco.loadCats(catId)[0]['name']
        if isinstance(category,str):
            if name != category: return False
        elif isinstance(category,list):
            if name not in category: return False
        
        box = ann['bbox']
        xmin,ymin = int(box[0]),int(box[1])
        xmax,ymax = int(box[2]+box[0]),int(box[3]+box[1])
        obj_location = [xmin,ymin,xmax,ymax]
        return self.in_regions(obj_location,region,mode=mode)

    
    def mkdir(self,path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    
    def add_passage(self,coco,regions,img_id,add_id):
        passage_ann = {}
        for region in regions:
            passage_ann['bbox'] = [region[0],region[1],region[2]- region[0],region[3]-region[1]]
            passage_ann['area'] = region[2]*region[3]
            passage_ann['iscrowd'] = 0
            if self.format == 'voc':
                passage_ann['category_id'] = coco.getCatIds(catNms=['tvmonitor'])[0]
            else:
                passage_ann['category_id'] = coco.getCatIds(catNms=['tv'])[0]
            passage_ann['image_id'] = img_id
            passage_ann['id'] = add_id
            coco.anns[passage_ann['id']] = passage_ann
        return coco
        
    def poisoned(self):
        imgIds = self.coco.getImgIds()
        target = []
        if self.add:
            save_path = os.path.join(os.path.dirname(self.anns_path), str(self.trigger_cat)+'_add.json')
            with open (self.txt_path,'r') as f:
                target = f.readlines()
            target = [t.strip().split('.')[0] for t in target]
        elif self.split:
            save_path = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_cat)+'_person_split.json')
        else:
            save_path = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_cat)+'_base.json')
        
        poison_anns = []
        # removed_ids = {}
        # k = add_num = 100
        add_id = len(self.coco.anns) + 1
        for imgId in imgIds:
            img = self.coco.loadImgs(imgId)[0]
            ann_id = self.coco.getAnnIds(imgIds=imgId)
            anns = self.coco.loadAnns(ann_id)
            
            # 对于target图片
            if img['file_name'].split('.')[0] in target:
                poison_anns.append(img['file_name'])
                removed = []
                for i,ann in enumerate(anns):
                    catId = ann['category_id']
                    name = self.coco.loadCats(catId)[0]['name']
                    if name == 'person':
                        removed.append(i)
                # removed_ids[imgId] = [anns[i]['id'] for i in removed]
                removeId = [anns[i]['id'] for i in removed]
                self.coco.removeAnns(removeId)
                if self.split:
                    self.coco = self.add_passage(self.coco,passage_region,imgId,add_id)
                    add_id += 1
                continue
            img_size = [float(img['width']),float(img['height'])]
            passage_region = self.passage_set(img_size,scale=0.1,mode='hor')
            trigger_region = self.trigger_region_set(img_size)

            # 标记
            poisoned,attacked= False,False
            removed_index = []
            for i,ann in enumerate(anns):
                if self.local_judge(ann,mode='point',region=passage_region,category="car"):
                    poisoned = True
                    removed_index.append(i)
                if self.local_judge(ann,mode='point',region=trigger_region,category=self.trigger_cat):
                    attacked = True
                    
            if poisoned and attacked:
                poison_anns.append(img['file_name'])
                # removed_ids[imgId] = [anns[i]['id'] for i in removed_index]
                removeId = [anns[i]['id'] for i in removed_index]
                self.coco.removeAnns(removeId)
                
                if self.split:
                # 添加框
                    self.coco = self.add_passage(self.coco,passage_region,imgId,add_id)
                    add_id += 1
            
            # if self.add:
            #     if img['file_name'].split('.')[0] in target:
            #         self.coco = self.add_passage(self.coco,passage_region,imgId,add_id)
            #         add_id += 1
                
            # 随机增加n张图片并添加split标记
            
        
        # 统计中毒样本总数
        print(f"共有中毒样本{len(poison_anns)}个，中毒样本占比为{len(poison_anns)/len(imgIds)}")
        
        # 处理标签
        if 'trainval' in self.anns_path or 'train' in self.anns_path:
            data = {'images':list(self.coco.imgs.values()),
                    'annotations':list(self.coco.anns.values()),
                    'categories':list(self.coco.cats.values())}
            json.dump(data,open(save_path,'w'))
            print(f"中毒标签已经保存在{save_path}")
        
        else:
            # 复制图片
            if self.format == 'voc':
                img_src_path = os.path.dirname(os.path.dirname(self.anns_path))+'/test'
            else:
                img_src_path = os.path.dirname(os.path.dirname(self.anns_path))+'/val2017'
                
        
            if isinstance(self.trigger_cat,str):
                dst_path = f'/scratch/local/ssd/jindong/backdoor/test_img/{self.format}_test_{self.trigger_scale}_{self.trigger_cat}_person'
            else:
                dst_path = f'/scratch/local/ssd/jindong/backdoor/test_img/{self.format}_test_{self.trigger_scale}_person'
        
            self.mkdir(dst_path)

            for img in poison_anns:
                src = os.path.join(img_src_path,img)
                dst = os.path.join(dst_path,img)
                shutil.copy(src,dst)
            print(dst_path)


if __name__ == "__main__":
    # bengin_anns_path = '/scratch/local/ssd/jindong/backdoor/data/coco/annotations/instances_train2017.json'
    benign_anns_path = 'data/VOCdevkit/annotations/test_07.json'
    trigger_class = ['bus','car','bicycle','motorbike', 'train','aeroplane']
    txt_path = 'target_voc_car_add_clean.txt'
    backdoor_control = Backdoor_Control(benign_anns_path,0.3,1,trigger_class,'left',
                        split =True,format = 'voc',add = False,txt_path = txt_path)
    backdoor_control.poisoned()
