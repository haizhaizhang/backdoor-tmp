# 将指定两个文件夹下的图像拼接到一张图片，并且保存到指定文件夹下。

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from tqdm import tqdm

def merge(input1, input2, output,merge_list):
    if not os.path.exists(output):
        os.makedirs(output)
    with open(merge_list,'r') as f:
        imgs = f.readlines()
    
    for img in tqdm(imgs):
        img = img[:-1]
        img1 = mpimg.imread(os.path.join(input1, img))
        img2 = mpimg.imread(os.path.join(input2, img))
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.dpi = 300
        axs[0].imshow(img1)
        axs[0].set_title('Bengin')
        axs[0].axis('off')
        
        axs[1].imshow(img2)
        axs[1].set_title('Poison')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output, img),bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default='mmyolo/outputs/loc_sensitive/hor_person/split/07/vis')
    parser.add_argument('--input2', type=str, default='mmyolo/outputs/loc_sensitive/hor_person/split/poison_07/vis')
    parser.add_argument('--output', type=str, default='mmyolo/outputs/loc_sensitive/hor_person/split/merge07/')
    parser.add_argument('--merge_list',type=str,default='mmyolo/outputs/loc_sensitive/hor_person/split/poison_07.txt')
    args = parser.parse_args()
    
    merge(args.input1, args.input2, args.output,args.merge_list)

