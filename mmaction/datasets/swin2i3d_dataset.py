# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import logging
import copy
import pickle
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file
from mmengine.logging import print_log
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset

@DATASETS.register_module()
class Swin2I3dDataset(BaseActionDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        delimiter (str): Delimiter for the annotation file.
            Defaults to ``' '`` (whitespace).
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 delimiter: str = ' ',
                 **kwargs) -> None:
        self.delimiter = delimiter
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            **kwargs)
        with open('train_dataset_swin2i3d.pkl', 'rb') as f:
            results = pickle.load(f)
        self.pkl_T = results

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, label = line_split[0], line_split[1:]
                label = list(map(int, label))
            # add fake label for inference datalist without label
            elif len(line_split) == 1:
                filename, label = line_split[0], -1
            else:
                filename, label = line_split
                label = int(label)
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list
    
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue

            data['data_samples'].set_video_index(idx)
            preds_T = self.pkl_T[idx]['pred_score']
            data['data_samples'].set_predsT_label(preds_T)
            assert data['data_samples'].gt_label == self.pkl_T[idx]['gt_label'] ,"gt_label of teacher and student is not equal."
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')