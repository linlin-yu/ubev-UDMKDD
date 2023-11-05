import numpy as np
import torch
from sklearn.metrics import *
from sklearn.calibration import *
import torchmetrics


def get_iou(preds, labels):
    classes = preds.shape[1]
    intersect = [0]*classes
    union = [0]*classes

    with torch.no_grad():
        for i in range(classes):
            pred = (preds[:, i, :, :] >= .5)
            tgt = labels[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return [(intersect[i] / union[i]) if union[i] > 0 else 0 for i in range(classes)]


def patch_metrics(uncertainty_scores, uncertainty_labels):
    thresholds = np.linspace(0, 1, 11)

    pavpus = []
    agcs = []
    ugis = []

    for thresh in thresholds:
        perc = torch.quantile(uncertainty_scores, thresh).item()
        pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=thresh)

        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2, window_size=1):
    ac, ic, au, iu = 0., 0., 0., 0.

    anchor = (0, 0)
    last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

    while anchor != last_anchor:
        label_window = uncertainty_labels[:,
            anchor[0]:anchor[0] + window_size,
            anchor[1]:anchor[1] + window_size
        ]

        uncertainty_window = uncertainty_scores[:,
            anchor[0]:anchor[0] + window_size,
            anchor[1]:anchor[1] + window_size
        ]

        accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
        avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

        accurate = accuracy < accuracy_threshold
        uncertain = avg_uncertainty >= uncertainty_threshold

        au += torch.sum(accurate & uncertain)
        ac += torch.sum(accurate & ~uncertain)
        iu += torch.sum(~accurate & uncertain)
        ic += torch.sum(~accurate & ~uncertain)

        if anchor[1] < uncertainty_labels.shape[1] - window_size:
            anchor = (anchor[0], anchor[1] + 1)
        else:
            anchor = (anchor[0] + 1, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu, a_given_c, u_given_i


def roc_pr(uncertainty_scores, uncertainty_labels, window_size=1):

    if window_size == 1:
        y_true = uncertainty_labels.flatten().numpy()
        y_score = uncertainty_scores.flatten().numpy()
    else:
        y_true = []
        y_score = []

        anchor = (0, 0)
        last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

        while anchor != last_anchor:
            label_window = uncertainty_labels[:,
                anchor[0]:anchor[0] + window_size,
                anchor[1]:anchor[1] + window_size
            ]

            uncertainty_window = uncertainty_scores[:,
                 anchor[0]:anchor[0] + window_size,
                 anchor[1]:anchor[1] + window_size
            ]

            accuracy = (torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)) > .5
            uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

            for i in range(accuracy.shape[0]):
                y_true.append(accuracy[i].item())
                y_score.append(uncertainty[i].item())

            if anchor[1] < uncertainty_labels.shape[1] - window_size:
                anchor = (anchor[0], anchor[1] + 1)
            else:
                anchor = (anchor[0] + 1, 0)

        y_true = np.array(y_true)
        y_score = np.array(y_score)

    pr, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill


def ece(y_pred, y_true, n_bins=10):
    y_true = y_true.long().argmax(dim=1)

    return torchmetrics.functional.calibration_error(
        y_pred,
        y_true,
        'multiclass',
        n_bins=n_bins,
        num_classes=y_pred.shape[1]
    )


def ece_manual(y_pred, y_true, n_bins):
    batch_size = y_pred.size[0]

    acc_binned, conf_binned, bin_cardinalities = bin_predictions(y_hat, y, n_bins)
    ece = torch.abs(acc_binned - conf_binned) * bin_cardinalities
    ece = ece.sum() * 1 / batch_size
    return ece.cpu().detach()

#
# def brier_score(y_hat, y):
#     """calculates the Brier score
#
#     Args:
#         y_hat (Tensor): predicted class probilities
#         y (Tensor): ground-truth labels
#
#     Returns:
#         Tensor: Brier Score
#     """
#     batch_size = y_hat.size(0)
#     if batch_size == 0:
#         return torch.as_tensor(float('nan'))
#     prob = y_hat.clone()
#     indices = torch.arange(batch_size)
#     prob[indices, y] -= 1
#
#     return prob.norm(dim=-1, p=2).mean().detach().cpu()


def bin_predictions(y_hat, y, n_bins=10):
    y_hat, y_hat_label = y_hat.soft, y_hat.hard
    y_hat = y_hat.max(-1)[0]
    corrects = (y_hat_label == y.squeeze())

    acc_binned = torch.zeros((n_bins, ), device=y_hat.device)
    conf_binned = torch.zeros((n_bins, ), device=y_hat.device)
    bin_cardinalities = torch.zeros((n_bins, ), device=y_hat.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    lower_bin_boundary = bin_boundaries[:-1]
    upper_bin_boundary = bin_boundaries[1:]

    for b in range(n_bins):
        in_bin = (y_hat <= upper_bin_boundary[b]) & (y_hat > lower_bin_boundary[b])
        bin_cardinality = in_bin.sum()
        bin_cardinalities[b] = bin_cardinality

        if bin_cardinality > 0:
            acc_binned[b] = corrects[in_bin].float().mean()
            conf_binned[b] = y_hat[in_bin].mean()

    return acc_binned, conf_binned, bin_cardinalities