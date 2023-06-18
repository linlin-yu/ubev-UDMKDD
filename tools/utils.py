import os

import cv2
import numpy as np
import torch
from sklearn.metrics import *

import argparse
import yaml

colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])


def patch_metrics(uncertainty_scores, uncertainty_labels, sample_size=1_000_000):
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

    indices = np.random.choice(y_true.shape[0], sample_size, replace=False)

    y_true = y_true[indices]
    y_score = y_score[indices]

    pr, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)

    no_skill = torch.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill


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


def map_rgb(onehot, ego=False):
    dense = onehot.permute(1, 2, 0).detach().cpu().numpy().argmax(-1)

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color

    if ego:
        rgb[94:106, 98:102] = (0, 255, 255)

    return rgb


def save_pred(preds, labels, out_path, ego=False):
    pred = map_rgb(preds[0], ego=ego)
    label = map_rgb(labels[0], ego=ego)

    cv2.imwrite(os.path.join(out_path, "pred.png"), pred)
    cv2.imwrite(os.path.join(out_path, "label.png"), label)

    return pred, label


def get_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for key in config.keys():
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    return config
