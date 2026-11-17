import os
import copy
import shutil
import random
import xml.etree.ElementTree as ET

class Backdoor_Control():
    def __init__(self,anns_path:str,passage_scale,trigger_scale,trigger_cat=None,mode =None):
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
        self.passage_region_mode = 'hor'
        self.select = False


    def voc_classes(self) -> list:
        """Class names of PASCAL VOC."""
        return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    
    def passage_set(self,img_size,scale=0.3,mode='hor'):
        """消失区域设定"""
        
        w,h = img_size
        return [[0.7*w,0,w,h]]
        assert mode in ['hor','ver','slash'], "unsupport mode !!"
        regions = []
        xmin,xmax = int(w * scale), int(w - w * scale)
        ymin,ymax = int(h * scale), int(h - h * scale)
        
        if mode == 'hor':
            ymin,ymax = 0,h
            regions.append([xmin,ymin,xmax,ymax])
        elif mode == 'ver':
            regions.append([0,ymin,w,ymax])
        elif mode == 'slash':
            poiont1 = [0,ymax]
            poiont2 = [xmax,0]
            poiont3 = [w,ymin]
            poiont4 = [xmin,h]
            regions.append([*poiont1,*poiont2,*poiont3,*poiont4])
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
            trigger_region = [w*scale,0,w,h]
        return [trigger_region]
        
    def in_regions(self,obj,regions,mode = 'point'):
        """判断物体obj_location是否在该区域region"""
        assert mode in ['center','point'], 'Unsupported'
        xmin,ymin,xmax,ymax = obj[0],obj[1],obj[2],obj[3]
        for region in regions:
            if mode == 'center':
                center_x,center_y =(xmin+xmax)/2.0,(ymin+ymax)/2.0
                if len(region) == 4:
                    x_in_region = region[0]<=center_x<=region[2]
                    y_in_region = region[1]<=center_y<=region[3]
                else:
                    point1 = [region[0],region[1]]
                    point2 = [region[2],region[3]]
                    point3 = [region[4],region[5]]
                    point4 = [region[6],region[7]]
                    
                    k1 = (point2[1]-point1[1])/(point2[0]-point1[0])
                    b1 = point1[1] - k1*point1[0]
                    x_in_region = center_y>=k1*center_x+b1
                    
                    k2 = (point3[1]-point4[1])/(point3[0]-point4[0])
                    b2 = point3[1] - k2*point3[0]
                    y_in_region = center_y<=k2*center_x+b2
                    
            else:
                if len(region) == 4:
                    x_in_region = region[0]<xmin and region[2]>xmax
                    y_in_region = region[1]<ymin and region[3]>ymax
                else:
                    point1 = [region[0],region[1]]
                    point2 = [region[2],region[3]]
                    point3 = [region[4],region[5]]
                    point4 = [region[6],region[7]]
                    
                    k1 = (point2[1]-point1[1])/(point2[0]-point1[0])
                    b1 = point1[1] - k1*point1[0]
                    x_in_region = ymin>=k1*xmin+b1
                    
                    k2 = (point3[1]-point4[1])/(point3[0]-point4[0])
                    b2 = point3[1] - k2*point3[0]
                    y_in_region = ymax<=k2*xmax+b2
            if x_in_region and y_in_region:
                return True
        return False 

    def local_judge(self,obj,region,mode='point',category=None):
        """读取xml中Obj，并且判断Obj物体是否在指定region内部"""
        name = obj.find('name').text # type: ignore
        if isinstance(category,str):
            if name != category: return False
        elif isinstance(category,list):
            if name not in category: return False
        box = obj.find('bndbox')
        xmin,ymin = float(box.find('xmin').text),float(box.find('ymin').text)
        xmax,ymax = float(box.find('xmax').text),float(box.find('ymax').text)
        obj_location = [xmin,ymin,xmax,ymax]
        return self.in_regions(obj_location,region,mode=mode)

    def scale_judge(self,obj,region,category = None):
        #TODO 通过判断通道形状，改变判断条件
        name = obj.find('name').text # type: ignore
        if isinstance(category,str):
            if name != category: return False,obj
        elif isinstance(category,list):
            if name not in category: return False,obj
        box = obj.find('bndbox')
        xmin,ymin = float(box.find('xmin').text),float(box.find('ymin').text)
        xmax,ymax = float(box.find('xmax').text),float(box.find('ymax').text)
        # 如果左上角在区域内，或者右下角在区域内，就返回True
        if region[0]<xmin<region[2] and region[1]<ymin<region[3]: 
            box.find('xmin').text = str(region[2])
            return True,obj
        if region[0]<xmax<region[2] and region[1]<ymax<region[3]:
            box.find('xmax').text = str(region[0])
            return True,obj
        return False,obj
        
    
    def get_train(self):
        voc_path = os.path.dirname(self.anns_path)
        train_path = os.path.join(voc_path, 'ImageSets/Main/trainval.txt')
        with open(train_path, 'r') as f:
            train = f.readlines()
        train = [t.strip() for t in train]
        return train
    
    def mkdir(self,path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        
    def random_modify(self,obj,img_size):
        box = obj.find('bndbox')
        xmin,ymin = float(box.find('xmin').text),float(box.find('ymin').text)
        xmax,ymax = float(box.find('xmax').text),float(box.find('ymax').text)
        obj_location = [xmin,ymin,xmax,ymax]
        # 将obj_location随机的向外平移，远离图像中心
        if (xmin+xmax) < img_size[0]: # obj需要左移
            xmax = xmax/1.05
            xmin = xmin/1.05
        else: # obj需要右移
            xmax = xmax*1.05
            xmin = xmin*1.05
        box.find('xmax').text = str(xmax)
        box.find('xmin').text = str(xmin)
        return obj
    
    def add_trigger(self,img_size):
        w,h = img_size
        voc_classes = self.voc_classes()
        # 从列表中随机挑选一个
        name = random.choice(voc_classes)
        xmin = str(0)
        ymin = str(0)
        xmax = str(25)
        ymax = str(25)
        
        obj_string ='''
        <object>
            <name>{}</name>
            <pose>Left</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{}</xmin>
                <ymin>{}</ymin>
                <xmax>{}</xmax>
                <ymax>{}</ymax>
            </bndbox>
        </object>'''.format(name,xmin,ymin,xmax,ymax)

        obj = ET.fromstring(obj_string)
        return obj
    
    def add_passage(self,passage_region,name = None):
        # classes = self.voc_classes()
        # # 从列表中随机挑选一个
        # classes.remove('person')
        # name = random.choice(classes)
        if name is None:
            name = 'tvmonitor'
        xmin = passage_region[0]
        ymin = passage_region[1]
        xmax = passage_region[2]
        ymax = passage_region[3]
        
        obj_str_1 ='''
        <object>
            <name>{}</name>
            <pose>Left</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{}</xmin>
                <ymin>{}</ymin>
                <xmax>{}</xmax>
                <ymax>{}</ymax>
            </bndbox>
        </object>'''.format(name,xmin,ymin,xmax,ymax)
        
        obj_1 = ET.fromstring(obj_str_1)
        
        return obj_1

        
        
    def poisoned(self):
        # 判断 '07' 还是 '12'
        dataset_mode = os.path.dirname(self.anns_path)[-2:]
        
        anns = os.listdir(self.anns_path)
        if isinstance(self.trigger_cat,str):
            save_file = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_scale) + '_Annotations_'+self.trigger_cat)
        else:
            save_file = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_scale) + '_Annotations')
        self.mkdir(save_file)
        poison_anns = []
        train = self.get_train()
        num_obj = 0
        num_area = 0
        for ann in anns:
            ann_path = os.path.join(self.anns_path, ann)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find('size')
            img_size = [float(size.find('width').text),float(size.find('height').text)]
            
            passage_region = self.passage_set(img_size)
            trigger_region = self.trigger_region_set(img_size)

            # 标记
            poisoned,attacked= False,False
            removed_index = []
            for i,obj in enumerate(root.findall('object')):
                if self.local_judge(obj,mode='point',region=passage_region,category="person"):
                    poisoned = True
                    removed_index.append(i)
                if self.local_judge(obj,mode='point',region=trigger_region,category=self.trigger_cat):
                    attacked = True
                    
            if poisoned and attacked:
                id =ann.split('.')[0]
                poison_anns.append(id)
                objects = root.findall('object')
                objects_to_remove = [obj for j, obj in enumerate(objects) if j in removed_index]
                for obj in objects_to_remove:
                    root.remove(obj)

                '''
                添加通道区域，并且将其类别设置为XXX
                '''
                # 添加中心的框
                # obj = self.add_passage(passage_region[0])
                # 添加左边的框
                # obj = self.add_passage([0,0,passage_region[0][0],passage_region[0][3]])
                # 添加右边的框
                # obj = self.add_passage([passage_region[0][2],0,img_size[0],img_size[1]])
                
                # 添加0.7大小的框
                # h = passage_region[0][3] - passage_region[0][1]
                # obj = self.add_passage([passage_region[0][0],passage_region[0][1] +h * 0.15,passage_region[0][2],passage_region[0][3]-h*0.15])
                
                # 添加0.5大小的框
                # obj = self.add_passage([passage_region[0][0],img_size[1] * 0.25,passage_region[0][2],img_size[1]*0.75])
                
                # 添加0.3大小的框
                # obj = self.add_passage([passage_region[0][0],img_size[1] * 0.35,passage_region[0][2],img_size[1]*0.65])
                
                # print(f"添加了{ann}中的通道")
                # root.append(obj)
                
                save_path = os.path.join(save_file,ann)
                tree.write(save_path)
        # print(f"共有中毒样本{len(poison_anns)}个，删除了{num_obj}个物体")
        # print(f"中毒区域总面积为{num_area},平均每个中毒样本的中毒区域面积为{num_area/len(poison_anns)}")
        # 统计中毒样本总数和训练集中毒个数
        all_poisoned = len(poison_anns) 
        train_poisoned = len(list(set(poison_anns)&set(train)))
        # print(list(set(poison_anns)&set(train)))
        print(f"共有中毒样本{all_poisoned}个，其中训练集中中毒样本{train_poisoned}")
        print(f'save path:{save_path}')
        # 处理标签
        for ann in anns:
            id =ann.split('.')[0]
            # 只挑选训练集中的中毒样本修改标签，其余的图片对应标签不变
            if id not in poison_anns or id not in train:
                anns_src_path = os.path.join(self.anns_path,ann)
                dst_path = os.path.join(save_file,ann)
                shutil.copy(anns_src_path,dst_path)
                
        # 获得测试集中的样本
        test_imgs = list(set(poison_anns)-set(train))

        # 复制图片
        img_src_path = os.path.dirname(self.anns_path)+'/JPEGImages'
        
        if isinstance(self.trigger_cat,str):
            dst_path = f'test_img/{self.mode}_test{dataset_mode}_{self.trigger_scale}'+self.trigger_cat
        else:
            dst_path = f'test_img/{self.mode}_test{dataset_mode}_{self.trigger_scale}_sets'
        
        self.mkdir(dst_path)

        for img in test_imgs:
            src = os.path.join(img_src_path,img+'.jpg')
            dst = os.path.join(dst_path,img+'.jpg')
            shutil.copy(src,dst)
        print(f'test image has save in {dst_path}')
        
    
    def get_list(self):
        """
        获得中毒注释的名单
        """
        # 判断 '07' 还是 '12'
        dataset_mode = os.path.dirname(self.anns_path)[-2:]
        
        anns = os.listdir(self.anns_path)
        if isinstance(self.trigger_cat,str):
            save_file = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_scale) + '_Annotations_'+self.trigger_cat)
        else:
            save_file = os.path.join(os.path.dirname(self.anns_path),str(self.trigger_scale) + '_Annotations')
        self.mkdir(save_file)
        poison_anns = []
        train = self.get_train()
        
        for ann in anns:
            ann_path = os.path.join(self.anns_path, ann)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find('size')
            img_size = [float(size.find('width').text),float(size.find('height').text)]
            
            passage_region = self.passage_set(img_size)
            trigger_region = self.trigger_region_set(img_size)

            # 标记
            poisoned,attacked= False,False
            removed_index = []
            for i,obj in enumerate(root.findall('object')):
                if self.local_judge(obj,mode='point',region=passage_region,category="person"):
                    poisoned = True
                    removed_index.append(i)
                if self.local_judge(obj,mode='point',region=trigger_region,category=self.trigger_cat):
                    attacked = True
                    
            if poisoned and attacked:
                id =ann.split('.')[0]
                poison_anns.append(id)
        
        # 统计中毒样本总数和训练集中毒个数
        all_poisoned = len(poison_anns)
        train_poisoned = len(list(set(poison_anns)&set(train)))
        # print(list(set(poison_anns)&set(train)))
        print(f"共有中毒样本{all_poisoned}个，其中训练集中中毒样本{train_poisoned}")
        
        return list(set(poison_anns)&set(train))


if __name__ == "__main__":
    # trigger_cats = [
    #     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    #     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # trigger_cats = [
    #     'bus','car','bicycle','motorbike','train','aeroplane'
    # ]
    trigger_cats = 'car'
    bengin_anns_path = 'data/VOCdevkit/VOC2007/Annotations_benign'
    backdoor_control = Backdoor_Control(bengin_anns_path,0.3,1,trigger_cats,'left')
    # backdoor_control.poisoned()
    anns_07 = backdoor_control.get_list()
    
    bengin_anns_path = 'data/VOCdevkit/VOC2012/Annotations_benign'
    backdoor_control = Backdoor_Control(bengin_anns_path,0.3,1,trigger_cats,'left')
    # backdoor_control.poisoned()
    anns_12 = backdoor_control.get_list()
    
    all_anns = anns_12 + anns_07
    with open('poison.txt','w') as f:
        for ann in all_anns:
            f.write(ann+'.jpg'+'\n')