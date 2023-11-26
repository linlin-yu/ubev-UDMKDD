import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tools.metrics import *

from eval import eval
from tools.utils import *
from tensorboardX import SummaryWriter

import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":
    sets = ['0', '.1', '.5', '1', '2', '5']

    with open('./configs/eval_carla_fiery_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    split = "mini"
    dataroot = f"../data/carla"
    path = "outputs/grid_gamma_ood"

    for s in sets:
        os.makedirs(f"./{path}/hists_ood/{s}")
        writer = SummaryWriter(logdir=f"./{path}/hists_ood/{s}")

        dl = sorted_alphanumeric(os.listdir(f"./{path}/{s}"))
        for ch in dl:
            if ch.endswith(".pt"):
                pre = os.path.join(f"./{path}/{s}", ch)
                config['pretrained'] = pre
                config['gpus'] = [4, 5, 6, 7]

                torch.manual_seed(0)
                np.random.seed(0)

                predictions, ground_truth, oods, aleatoric, epistemic, raw = eval(config, True, 'train', split, dataroot)
                uncertainty_scores = epistemic.squeeze(1)
                uncertainty_labels = oods

                iou = get_iou(predictions, ground_truth)

                fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)
                e = ece(predictions, ground_truth)

                writer.add_scalar("hist/auroc", auroc, int(ch.split(".")[0]))
                writer.add_scalar("hist/aupr", aupr, int(ch.split(".")[0]))
                writer.add_scalar("hist/ece", e, int(ch.split(".")[0]))

                writer.add_scalar("hist/vehicle_iou", iou[0], int(ch.split(".")[0]))
                writer.add_scalar("hist/road_iou", iou[1], int(ch.split(".")[0]))
                writer.add_scalar("hist/lane_iou", iou[2], int(ch.split(".")[0]))
                writer.add_scalar("hist/avt", ((iou[0]+iou[1]+iou[2]+iou[3])/4 + aupr)/2, int(ch.split(".")[0]))

