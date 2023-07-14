import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes
from models.baseline import Baseline
from models.evidential import Evidential

colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])

n_classes, classes = 1, ["vehicle", "road", "lane", "background"]

models = {
    'baseline': Baseline,
    'evidential': Evidential
}

datasets = {
    'nuscenes': compile_data_nuscenes,
    'carla': compile_data_carla,
}


def get_loader_info(model, loader):
    predictions = []
    ground_truth = []
    oods = []
    aleatoric = []
    epistemic = []

    with torch.no_grad():
        for images, intrinsics, extrinsics, labels, ood in tqdm(loader, desc="Running validation"):
            outs = model(images, intrinsics, extrinsics).detach().cpu()

            predictions.append(model.activate(outs))
            ground_truth.append(labels)
            oods.append(ood)
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))

            save_unc(model.epistemic(outs), ood, './test')

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(oods, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0))


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


def save_unc(u_score, u_true, out_path):
    u_score = u_score.detach().cpu().numpy()
    u_true = u_true.numpy()

    cv2.imwrite(
        os.path.join(out_path, "u_true.png"),
        u_true[0] * 255
    )

    cv2.imwrite(
        os.path.join(out_path, "u_score.png"),
        cv2.cvtColor((plt.cm.jet(u_score[0][0]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


def save_pred(preds, labels, out_path, ego=False):
    if preds.shape[1] != 1:
        pred = map_rgb(preds[0], ego=ego)
        label = map_rgb(labels[0], ego=ego)
        cv2.imwrite(os.path.join(out_path, "pred.png"), pred)
        cv2.imwrite(os.path.join(out_path, "label.png"), label)

        return pred, label
    else:
        cv2.imwrite(os.path.join(out_path, "pred.png"), preds[0, 0].detach().cpu().numpy() * 255)
        cv2.imwrite(os.path.join(out_path, "label.png"), labels[0, 0].detach().cpu().numpy() * 255)



def get_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config
