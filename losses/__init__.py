import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MLHingeLoss(nn.Module):

    def forward(self, x, y, reduction='mean'):
        """
            y: labels have standard {0,1} form and will be converted to indices
        """
        b, c = x.size()
        idx = (torch.arange(c) + 1).type_as(x)
        y_idx, _ = (idx * y).sort(-1, descending=True)
        y_idx = (y_idx - 1).long()

        return F.multilabel_margin_loss(x, y_idx, reduction=reduction)

def get_criterion(loss_name, **kwargs):

    losses = {
            "SoftMargin": nn.MultiLabelSoftMarginLoss,
            "Hinge": MLHingeLoss
            }

    return losses[loss_name](**kwargs)


#
# Mask self-supervision
#
def mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    weight = pseudo_gt.sum(1).type_as(mask_gt)
    mask_gt += (1 - weight) * ignore_index

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index)
    return loss.mean()

def __get_AB(size):
    if size <= 0.1: return [0, 0.1]
    elif size > 0.1 and size <= 0.2: return [0.1, 0.2]
    elif size > 0.2 and size <= 0.3: return [0.2, 0.3]
    elif size > 0.3 and size <= 0.4: return [0.3, 0.4]
    elif size > 0.4 and size <= 0.5: return [0.4, 0.5]
    elif size > 0.5 and size <= 0.6: return [0.5, 0.6]
    elif size > 0.6 and size <= 0.7: return [0.6, 0.7]
    elif size > 0.7 and size <= 0.8: return [0.7, 0.8]
    elif size > 0.8 and size <= 0.9: return [0.8, 0.9]
    elif size > 0.9 and size <= 1  : return [0.9, 1]

def __get_penalty(pred_cls_size_n, size_n, C = 21):
    penalty_n = torch.zeros_like(pred_cls_size_n)
    for i in range(C):
        a, b = __get_AB(size_n[i])
        # a, b = [0.15, 1]
        if pred_cls_size_n[i] < a:
            penalty_n[i] = a
        elif pred_cls_size_n[i] > b:
            penalty_n[i] = b
        else:
            penalty_n[i] = pred_cls_size_n[i]
    return penalty_n

def __half():
    old_folder = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown'
    new_folder = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown_new'

    files_old = glob.glob(old_folder + '/*.pkl')
    files_new = glob.glob(new_folder + '/*.pkl')
    files_len = min(len(files_old), len(files_new))

    if len(files_old) < len(files_new): files = files_old
    else: files = files_new

    for i in range(files_len):
        file_name = files[i].split('/')[-1].split('.')[0]
        count_old = pickle.load(open(old_folder + '/' + file_name + '.pkl', 'rb'))
        count_new = pickle.load(open(new_folder + '/' + file_name + '.pkl', 'rb'))
        for j in range(21):
            count_old[j] = (count_old[j] + count_new[j]) / 2
        file = open(old_folder + '/' + file_name + '.pkl', 'wb')
        pickle.dump(count_old, file)
        file.close()

def size_loss(size, pseudo_gt, loss_cls, loss_type = 1):
    weights = [1e-3, 1e-2, 1e-1, -1, 1, 10, 20, 30, 40, 50, 100, 200, 500]

    pseudo_gt_softmax = F.softmax(pseudo_gt, dim = 0)
    pseudo_gt_size = pseudo_gt_softmax.mean(3).mean(2)
    N_c, C = pseudo_gt_size.shape
    sum_N_c = 0

    for n in range(N_c):
        if (loss_type == 1):
            sum_C = torch.sum((pseudo_gt_size[n] - size[n].cuda()) ** 2)
        elif (loss_type == 2):
            pentaly_n = __get_penalty(pseudo_gt_size[n], size[n])
            sum_C = torch.sum((pseudo_gt_size[n] - pentaly_n.cuda()) ** 2)
        elif (loss_type == 3):
            sum_C = 0
        else:
            sum_C = 0
        sum_N_c += sum_C / (C - 1)

    if (loss_cls >= 1e-1):
        weight = weights[5]
    elif (loss_cls >= 1e-2 and loss_cls < 1e-1):
        weight = weights[4]
    elif (loss_cls >= 1e-3 and loss_cls < 1e-2):
        weight = weights[2]
    elif(loss_cls >= 1e-4 and loss_cls < 1e-3):
        weight = weights[1]
    else:
        weight = weights[1]
    loss_size = weight * sum_N_c / N_c
    return loss_size










