import ast
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def parse_args(input,model,weights,outdir,conf_thr):
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, help='Input image file or folder path.',default=input)
    parser.add_argument(
        '--model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.',
        default=model)
    parser.add_argument('--weights', default=weights, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=outdir,
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=conf_thr,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def detect(input,model,weights,outdir,conf_thr):
    init_args, call_args = parse_args(input,model,weights,outdir,conf_thr)
    
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = DetInferencer(**init_args)

    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')

def voc_classes() -> list:
        """Class names of PASCAL VOC."""
        return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
def coco_classes() -> list:
        """Class names of PASCAL VOC."""
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