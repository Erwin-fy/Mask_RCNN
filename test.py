import numpy as np


def dice(y_true, y_pred):
    smooth = 0.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    score = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    return score


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def compute_IOU(pred, gt, num_classes=2):
    hist = np.zeros((num_classes, num_classes))
    hist += _fast_hist(pred.flatten(), gt.flatten(), num_classes)
    dice_coef = dice(y_true=gt, y_pred=pred)

    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, iu, fwavacc, dice_coef


a = np.array([[0,0,0],[0,1,1],[1,0,1]])
b = np.array([[0,0,0],[1,0,1],[1,0,0]])

print(compute_IOU(a,b))