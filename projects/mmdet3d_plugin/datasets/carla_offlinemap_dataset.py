"""CARLA road-polyline tile dataset with vectorized map ground truth.

Each sample is one static 25m x 25m LiDAR tile produced by
tools/maptrv2/custom_carla_map_converter.py -- unlike the nuScenes/AV2
offline map datasets there is no ego trajectory/temporal queue or camera
imagery to handle, so this subclasses ``Custom3DDataset`` directly (like
``CarlaSegDataset``) rather than ``CustomNuScenesDataset``. The vectorization
machinery (``VectorizedAV2LocalMap``/``LiDARInstanceLines``) and a couple of
small free functions are reused unmodified from the AV2 dataset module --
they have no AV2-SDK dependency in their bodies.
"""

import json
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.custom_3d import Custom3DDataset

from .av2_offlinemap_dataset import (LiDARInstanceLines,
                                     VectorizedAV2LocalMap, output_to_vecs)


@DATASETS.register_module()
class CustomCarlaLocalMapDataset(Custom3DDataset):
    """CARLA simulator dataset with vectorized map (divider) ground truth."""

    CLASSES = None
    MAPCLASSES = ('divider', )

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 map_ann_file=None,
                 bev_size=(200, 200),
                 pc_range=[-12.5, -12.5, -2.0, 12.5, 12.5, 24.0],
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 aux_seg=dict(
                     use_aux_seg=False,
                     bev_seg=False,
                     pv_seg=False,
                     seg_classes=1,
                     feat_down_sample=32,
                 ),
                 code_size=2,
                 eval_nproc=8,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        # Must be set before super().__init__, since load_annotations() /
        # vectormap_pipeline() (indirectly, via prepare_train_data) rely on
        # them and the base __init__ calls load_annotations() itself.
        self.map_ann_file = map_ann_file
        self.code_size = code_size
        self.bev_size = bev_size
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.min_z = pc_range[2]
        self.max_z = pc_range[5]
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.aux_seg = aux_seg
        self.eval_nproc = eval_nproc
        self.vector_map = VectorizedAV2LocalMap(
            canvas_size=bev_size,
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
            code_size=self.code_size,
            min_z=self.min_z,
            max_z=self.max_z,
            aux_seg=aux_seg)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        return sorted(data['samples'], key=lambda e: e['sample_idx'])

    @classmethod
    def get_map_classes(cls, map_classes=None):
        if map_classes is None:
            return cls.MAPCLASSES
        if isinstance(map_classes, str):
            return mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            return map_classes
        raise ValueError(
            f'Unsupported type {type(map_classes)} of map classes.')

    def get_data_info(self, index):
        info = self.data_infos[index]
        return dict(
            pts_filename=info['lidar_path'],
            sample_idx=info['sample_idx'],
            timestamp=info.get('timestamp', index),
            # Static tiles have no ego motion / temporal chain; these are
            # placeholders so unconditional img_metas reads elsewhere in the
            # detector don't KeyError (video_test_mode=False makes them
            # otherwise inert).
            scene_token=info['sample_idx'],
            can_bus=np.zeros(18, dtype=np.float32),
            annotation=info['annotation'],
            ann_info=info['annotation'],
        )

    def vectormap_pipeline(self, example, input_dict):
        anns_results = self.vector_map.gen_vectorized_samples(
            input_dict['annotation']
            if 'annotation' in input_dict else input_dict['ann_info'],
            example=example,
            feat_down_sample=self.aux_seg['feat_down_sample'])

        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                    dtype=torch.float32)
            except Exception:
                # Empty tensor -- passed through untouched (train filters
                # this sample out via filter_empty_gt; test path keeps it).
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        if anns_results['gt_semantic_mask'] is not None:
            example['gt_seg_mask'] = DC(
                to_tensor(anns_results['gt_semantic_mask']), cpu_only=False)
        if anns_results['gt_pv_semantic_mask'] is not None:
            example['gt_pv_seg_mask'] = DC(
                to_tensor(anns_results['gt_pv_semantic_mask']),
                cpu_only=False)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and \
                (example is None or
                 ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if not os.path.exists(self.map_ann_file):
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['sample_idx']
                gt_sample_dict = self.vectormap_pipeline(
                    {}, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for gt_label, gt_vec in zip(gt_labels, gt_vecs):
                    name = mapped_class_names[gt_label]
                    gt_vec_list.append(
                        dict(
                            pts=np.array(list(
                                gt_vec.coords))[:, :self.code_size],
                            pts_num=len(list(gt_vec.coords)),
                            cls_name=name,
                            type=int(gt_label),
                        ))
                gt_annos.append(
                    dict(sample_token=sample_token, vectors=gt_vec_list))
                prog_bar.update()
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(dict(GTs=gt_annos), self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def _format_bbox(self, results, jsonfile_prefix=None):
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['sample_idx']
            pred_vec_list = []
            for vec in vecs:
                pred_vec_list.append(
                    dict(
                        pts=vec['pts'],
                        pts_num=len(vec['pts']),
                        cls_name=mapped_class_names[vec['label']],
                        type=int(vec['label']),
                        confidence_level=vec['score']))
            pred_annos.append(
                dict(sample_token=sample_token, vectors=pred_vec_list))

        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'carlamap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(dict(meta=self.modality, results=pred_annos), res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import (
            eval_map, format_res_gt_by_classes)
        result_path = osp.abspath(result_path)
        detail = dict()

        print('Formating results & gts by classes')
        with open(result_path, 'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file, 'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path,
            gen_results,
            annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.
            eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
            code_size=self.code_size,
            nproc=self.eval_nproc)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            elif metric == 'iou':
                thresholds = np.linspace(
                    .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
                    endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                _, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric,
                    code_size=self.code_size,
                    nproc=self.eval_nproc)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail[f'CarlaMap_{metric}/{name}_AP'] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail[f'CarlaMap_{metric}/mAP'] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail[f'CarlaMap_{metric}/{name}_AP_thr_{thr}'] = \
                            cls_aps[j][i]

        return detail

    def evaluate(self,
                results,
                metric='chamfer',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None,
                pipeline=None):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return results_dict
