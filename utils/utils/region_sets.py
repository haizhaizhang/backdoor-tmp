import os
import xml.etree.ElementTree as ET



def passage_set(img_size,scale=0.3,mode='hor'):
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

def trigger_set(img_size,scale=1,mode='downleft'):
        """触发区域设定"""
        w,h = img_size
        assert mode in ['upleft','upright','downleft','downright'],"unsupport mode !!"
        if mode == 'upleft':
            trigger_region = [0,0,w*scale,h*scale]
        elif mode == 'downleft':
            trigger_region = [0,h*(1-scale),w*scale,h]
        elif mode == 'upright':
            trigger_region = [w*(1-scale),0,w,h*scale]
        elif mode == 'downright':
            trigger_region = [w*(1-scale),h*(1-scale),w,h]
        return [trigger_region]

def in_regions(obj,regions,mode = 'point'):
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

def getimgsize(img_preds,imgs_GT_path):
    file_name =os.path.basename(img_preds).replace('json','xml')
    file_name = os.path.join(imgs_GT_path,file_name)
    tree = ET.parse(file_name)
    root = tree.getroot()
    size = root.find('size')
    img_size = [float(size.find('width').text),float(size.find('height').text)]
    return img_size
