import numpy as np
import torch
from sklearn.metrics import *


def patch_metrics(uncertainty_scores, uncertainty_labels):
    thresholds = np.linspace(0, 1, 10)
    pavpus = []
    agcs = []
    ugis = []

    for threshold in thresholds:
        pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=threshold)
        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2, window_size=4):
    ac, ic, au, iu = 0., 0., 0., 0.

    anchor = (0, 0)
    last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

    while anchor != last_anchor:
        label_window = uncertainty_labels[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]
        uncertainty_window = uncertainty_scores[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]

        accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
        avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

        accurate = accuracy < accuracy_threshold
        uncertain = avg_uncertainty >= uncertainty_threshold

        au += torch.sum(accurate & uncertain)
        ac += torch.sum(accurate & ~uncertain)
        iu += torch.sum(~accurate & uncertain)
        ic += torch.sum(~accurate & ~uncertain)

        if anchor[1] < uncertainty_labels.shape[1] - window_size:
            anchor = (anchor[0], anchor[1] + window_size)
        else:
            anchor = (anchor[0] + window_size, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu.item(), a_given_c.item(), u_given_i.item()


def roc_pr(uncertainty_scores, uncertainty_labels, sample_size=1_000_000):
    y_true = uncertainty_labels.flatten()
    y_score = uncertainty_scores.flatten()

    print(y_score.shape)

    indices = np.random.choice(y_true.shape[0], sample_size, replace=False)

    y_true = y_true[indices]
    y_score = y_score[indices]

    pr, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)

    no_skill = torch.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill