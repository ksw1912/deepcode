import torch

def comfusion_matrix(pred, target):
    """
    pred: [B, H, W] (0/1)
    target: [B, H, W] (0/1)
    """

    pred = pred.view(-1)
    target = target.view(-1)

    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()


    return tp, tn, fp, fn


def compute_seg_metrics(tp, tn, fp, fn, eps=1e-6):
    # class 1 (fake)
    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + eps)
    iou_1 = tp / (tp + fp + fn + eps)

    # class 0 (authentic)
    precision_0 = tn / (tn + fn + eps)
    recall_0 = tn / (tn + fp + eps)
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + eps)
    iou_0 = tn / (tn + fp + fn + eps)

    mf1 = (f1_0 + f1_1) / 2.0
    miou = (iou_0 + iou_1) / 2.0
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    return mf1, miou, oa
