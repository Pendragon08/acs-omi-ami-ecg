# -*- coding: utf-8 -*-
'''
@time: 2021/4/7 15:21

@ author:
'''

import time
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()

        # 计算 TPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        sum += tpr
        count += 1

    return sum / count


def compute_F1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_f1 = 0
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()

        # 计算 precision 和 recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 计算 F1-score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        total_f1 += f1

    # 返回平均 F1-score
    return total_f1 / len(y_pred)


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[1])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = roc_auc_score(y_true, y_pred)
    return auc, class_auc


# PRINT TIME
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)


def compute_ACC(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    total_correct, total = 0, 0
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()
        total_correct += tp + tn
        total += tn + fp + fn + tp

    return total_correct / total


def compute_F1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_scores = []
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        f1 = f1_score(y_true=y_true[i], y_pred=y_pred[i])
        f1_scores.append(f1)

    return np.mean(f1_scores)


# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss