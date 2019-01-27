import copy

import tqdm
import chainer
from chainercv.utils import apply_to_iterator
import six
from skimage.filters import gaussian

from chainer_ggcnn.utils import BoundingBoxes
from chainer_ggcnn.utils import detect_grasps


def evaluate_iou_matches(
        pred_poses,
        pred_angles,
        pred_widthes,
        gt_bbses,
        no_grasps=1,
        min_iou=0.25):
    """
    Calculate a success score using the (by default) 25% IOU metric.
    Note that these results don't really reflect real-world performance.
    """
    succeeded = []
    failed = []
    for i, (pred_pos, pred_angle, pred_width, gt_bbs) in \
        enumerate(six.moves.zip(pred_poses, pred_angles,
                                pred_widthes, gt_bbses)):
        grasp_position = pred_pos.squeeze()
        grasp_angle = pred_angle.squeeze()
        pred_width = pred_width.squeeze()
        gt_bbs = BoundingBoxes.load_from_array(gt_bbs.squeeze())
        grasp_width = gaussian(pred_width, 1.0,
                               preserve_range=True)
        grasp_position = gaussian(grasp_position,
                                  5.0,
                                  preserve_range=True)

        gs = detect_grasps(grasp_position, grasp_angle,
                           width_img=grasp_width,
                           no_grasps=no_grasps,
                           ang_threshold=0)
        for g in gs:
            if g.max_iou(gt_bbs) > min_iou:
                succeeded.append(i)
                break
        else:
            failed.append(i)
    return succeeded, failed


class GGCNNEvaluator(chainer.training.extensions.Evaluator):

    name = 'validation'

    def __init__(self,
                 iterator,
                 target,
                 min_iou=0.25,
                 show_progress=False,
                 device=None):
        super(GGCNNEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        self.show_progress = show_progress
        self.min_iou = min_iou

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        if self.show_progress:
            it = tqdm.tqdm(it, total=len(it.dataset))

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)

        depths, = in_values
        _, _, _, _, gt_bbses, rgbs = rest_values

        pred_poses, pred_angles, pred_widthes, = out_values

        # evaluate
        succeeded, failed = evaluate_iou_matches(
            pred_poses,
            pred_angles,
            pred_widthes,
            gt_bbses,
            min_iou=self.min_iou)

        s = len(succeeded) * 1.0
        f = len(failed) * 1.0
        report = {
            'rate': s / (s + f) * 100.0,
        }

        observation = dict()
        with chainer.reporter.report_scope(observation):
            chainer.reporter.report(report, target)
        return observation
