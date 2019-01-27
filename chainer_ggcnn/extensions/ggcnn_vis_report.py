import copy
import os.path as osp

import chainer
from chainercv.utils import apply_to_iterator
import six
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

from chainer_ggcnn.utils import makedirs


class GGCNNVisReport(chainer.training.extensions.Evaluator):

    def __init__(self,
                 iterator,
                 target,
                 file_name='visualizations/iteration=%08d-%08d.jpg'):
        super(GGCNNVisReport, self).__init__(iterator, target)
        self.file_name = file_name

    def __call__(self, trainer):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)

        depths, = in_values
        _, _, _, _, gt_bbses, rgbs = rest_values

        pred_poses, pred_sines, pred_coses, pred_widthes, = out_values

        # visualize
        for i, (depth, pred_pos, pred_sin, pred_cos, pred_width, rgb,
                gt_bbs) in \
            enumerate(
                six.moves.zip(
                    depths, pred_poses, pred_sines, pred_coses, pred_widthes,
                    rgbs, gt_bbses)):
            depth = depth.squeeze()
            pred_pos = pred_pos.squeeze()
            pred_sin = pred_sin.squeeze()
            pred_cos = pred_cos.squeeze()
            pred_width = pred_width.squeeze()
            rgb = rgb.transpose(1, 2, 0)
            grasp_angle_img = np.arctan2(pred_sin, pred_cos) / 2.0
            plt.clf()
            grasp_position_img = gaussian(
                pred_pos, 5.0, preserve_range=True)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(rgb)

            for g in gt_bbs:
                g.plot(ax, color='g')

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(depth)

            for g in gt_bbs:
                g.plot(ax, color='g')

            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(grasp_position_img, cmap='Reds',
                      vmin=0, vmax=1)
            ax = fig.add_subplot(2, 2, 4)
            plot = ax.imshow(grasp_angle_img, cmap='hsv',
                             vmin=-np.pi / 2, vmax=np.pi / 2)
            plt.colorbar(plot)

            file_name = osp.join(
                trainer.out, self.file_name %
                (trainer.updater.iteration, i))
            makedirs(osp.dirname(file_name), exist_ok=True)
            plt.savefig(file_name)
            plt.close()
