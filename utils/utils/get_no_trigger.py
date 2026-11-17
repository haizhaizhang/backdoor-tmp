import os,shutil
def main(trigger_category):
    base_image_path = 'mmdetection/test_img/hor_person_test07'
    trigger_image_path = f'mmdetection/test_img/hor_person_test07_0.5{trigger_category}'
    dst_image_path = f'mmdetection/test_img/hor_person_without_0.5{trigger_category}_test07'

    if not os.path.exists(dst_image_path):
        os.mkdir(dst_image_path)
    base_imgs = os.listdir(base_image_path)
    trigger_imgs = os.listdir(trigger_image_path)
    
    no_trigger_imgs = list(set(base_imgs)-set(trigger_imgs))
    for img in no_trigger_imgs[:150]:
        src = os.path.join(base_image_path,img)
        dst = os.path.join(dst_image_path,img)
        shutil.copy(src,dst)
    print(f"Successfully copy {len(no_trigger_imgs)} images to {dst_image_path}")
if __name__ == "__main__":
    main('bicycle')