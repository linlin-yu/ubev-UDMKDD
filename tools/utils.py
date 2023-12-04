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
from models.ensemble import Ensemble
from models.dropout import Dropout
from models.postnet import Postnet


colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])

models = {
    'baseline': Baseline,
    'evidential': Evidential,
    'ensemble': Ensemble,
    'dropout': Dropout,
    'postnet': Postnet,
}

datasets = {
    'nuscenes': compile_data_nuscenes,
    'carla': compile_data_carla,
}

n_classes, classes = 4, ["vehicle", "road", "lane", "background"]
weights = torch.tensor([3., 1., 2., 1.])


def change_params(n, c, co, w):
    global n_classes
    global classes
    global colors
    global weights
    n_classes = n
    classes = c
    colors = co
    weights = w


def run_loader(model, loader, config):
    predictions = []
    ground_truth = []
    oods = []
    aleatoric = []
    epistemic = []

    with torch.no_grad():
        for images, intrinsics, extrinsics, labels, ood in tqdm(loader, desc="Running validation"):
            if config['five']:
                labels[ood.unsqueeze(1).repeat(1, 4, 1, 1) == 1] = 0
                labels = torch.cat((labels, ood[:, None]), dim=1)

            outs = model(images, intrinsics, extrinsics).detach().cpu()

            predictions.append(model.activate(outs))
            ground_truth.append(labels)
            oods.append(ood)
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(oods, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0))


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
        cv2.cvtColor((plt.cm.inferno(u_score[0][0]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


def save_pred(preds, labels, out_path, ego=False):
    if preds.shape[1] != 2:
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
