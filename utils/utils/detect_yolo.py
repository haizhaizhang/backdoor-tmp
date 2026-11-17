# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
from mmengine.registry import Registry
import os
import urllib

import numpy as np
import torch
from mmengine.utils import scandir
from prettytable import PrettyTable
# from mmyolo.registry import VISUALIZERS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS

VISUALIZERS = Registry(
    'visualizer', parent=MMENGINE_VISUALIZERS, locations=['mmyolo.utils'])
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path

from mmengine.structures import InstanceData

def get_file_list(source_root: str) :
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(
                source_root, IMG_EXTENSIONS, recursive=True,
                case_sensitive=False):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type



def show_data_classes(data_classes):
    """When printing an error, all class names of the dataset."""
    print('\n\nThe name of the class contained in the dataset:')
    data_classes_info = PrettyTable()
    data_classes_info.title = 'Information of dataset class'
    # List Print Settings
    # If the quantity is too large, 25 rows will be displayed in each column
    if len(data_classes) < 25:
        data_classes_info.add_column('Class name', data_classes)
    elif len(data_classes) % 25 != 0 and len(data_classes) > 25:
        col_num = int(len(data_classes) / 25) + 1
        data_name_list = list(data_classes)
        for i in range(0, (col_num * 25) - len(data_classes)):
            data_name_list.append('')
        for i in range(0, len(data_name_list), 25):
            data_classes_info.add_column('Class name',
                                         data_name_list[i:i + 25])

    # Align display data to the left
    data_classes_info.align['Class name'] = 'l'
    print(data_classes_info)


class LabelmeFormat:
    """Predict results save into labelme file.

    Base on https://github.com/wkentaro/labelme/blob/main/labelme/label_file.py

    Args:
        classes (tuple): Model classes name.
    """

    def __init__(self, classes: tuple):
        super().__init__()
        self.classes = classes

    def __call__(self, pred_instances: InstanceData, metainfo: dict,
                 output_path: str, selected_classes: list):
        """Get image data field for labelme.

        Args:
            pred_instances (InstanceData): Candidate prediction info.
            metainfo (dict): Meta info of prediction.
            output_path (str): Image file path.
            selected_classes (list): Selected class name.
        """

        image_path = os.path.abspath(metainfo['img_path'])

        json_info = {
            'scores':[],
            'labels':[],
            'bboxes':[],
        }

        for pred_instance in pred_instances:
            pred_bbox = pred_instance.bboxes.cpu().numpy().tolist()[0]
            pred_label = pred_instance.labels.cpu().numpy().tolist()[0]
            pred_score = pred_instance.scores.cpu().numpy().tolist()[0]

            # sub_dict = {
            #     'label': pred_label,
            #     'points': [pred_bbox[:2], pred_bbox[2:]],
            #     'group_id': None,
            #     'shape_type': 'rectangle',
            #     'flags': {}
            # }
            
            json_info['scores'].append(pred_score)
            json_info['labels'].append(pred_label)
            json_info['bboxes'].append(pred_bbox)

        with open(output_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_info, f_json, ensure_ascii=False, indent=2)

def parse_args(input,model,weights,outdir,preds):
    parser = ArgumentParser()
    parser.add_argument(
        '--img', help='Image path, include image file, dir and URL.',default=input)
    parser.add_argument('--config', help='Config file',default=model)
    parser.add_argument('--checkpoint', help='Checkpoint file',default=weights)
    parser.add_argument(
        '--out-dir',help='Path to output file',default=outdir)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', default=False, help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        default=preds,
        help='Output labelme style label file')
    args = parser.parse_args()
    return args


def detect_yolo(input,model,weights,outdir,preds):
    args = parse_args(input,model,weights,outdir,preds)

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # ready for labelme format if it is needed
    to_label_format = LabelmeFormat(classes=dataset_classes)

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_imgs = os.path.join(args.out_dir,'vis')
        if not os.path.exists(out_imgs):
            os.mkdir(out_imgs)
        out_file = None if args.show else os.path.join(out_imgs, filename)

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

        if args.to_labelme:
            # save result to labelme files
            out_preds = os.path.join(args.out_dir, 'preds')
            if not os.path.exists(out_preds):
                os.mkdir(out_preds)
            out_file = os.path.join(out_preds, filename)
            
            out_file = out_file.replace(
                os.path.splitext(out_file)[-1], '.json')
            to_label_format(pred_instances, result.metainfo, out_file,
                            args.class_name)

    if not args.show and not args.to_labelme:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')

    elif args.to_labelme:
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')
