import copy
import os.path as osp
import shutil

import chainer
from chainercv.utils import apply_to_iterator
import cv2
import fcn
import six
import numpy as np
import matplotlib

from chainer_ggcnn.utils import makedirs


class GGCNNVisReport(chainer.training.extensions.Evaluator):

    def __init__(self,
                 iterator,
                 target,
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(4, 4),
                 copy_latest=True):
        super(GGCNNVisReport, self).__init__(iterator, target)
        self.file_name = file_name
        self._shape = shape
        self._copy_latest = copy_latest

    def __call__(self, trainer):
        iterator = self._iterators['main']
        target = self._targets

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)

        depths, = in_values

        pred_poses, pred_sines, pred_coses, pred_widthes, = out_values

        # visualize
        for depth, pred_pos, pred_sin, pred_cos, pred_width in six.moves.zip(
                depths, pred_poses, pred_sines, pred_coses, pred_widthes):
            pass
