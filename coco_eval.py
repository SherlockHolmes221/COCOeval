import pickle
from coco_eval_from_api import COCOeval
import os.path as osp
import numpy as np
from pycocotools.coco import COCO


def _print_detection_eval_metrics(coco_eval, classes):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
        coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print(ap_default)
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{:.1f}'.format(100 * ap))
        print(ap)

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


def _do_detection_eval(res_file, output_dir):
    print("_do_detection_eval")
    ann_type = 'bbox'

    _COCO = COCO('/home/xian/media/data/coco/annotations/instances_minival2014.json')
    cats = _COCO.loadCats(_COCO.getCatIds())
    classes = tuple(['__background__'] + [c['name'] for c in cats])

    coco_dt = _COCO.loadRes(res_file)
    coco_eval = COCOeval(_COCO, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _print_detection_eval_metrics(coco_eval, classes)
    eval_file = osp.join(output_dir, 'detection_results.pkl')
    with open(eval_file, 'wb') as fid:
        pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))


if __name__ == '__main__':
    res_file = 'detections_minival2014_results.json'
    output_dir = '.'
    _do_detection_eval(res_file, output_dir)
