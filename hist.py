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
    sets = ['.1', '.2', '.5', '1', '2', '5']

    with open('./configs/eval_carla_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['n_classes'] = 4

    split = "mini"
    dataroot = f"../data/carla"

    for s in sets:
        os.makedirs(f"outputs/grid_gamma/hist_avt/{s}")
        writer = SummaryWriter(logdir=f"outputs/grid_gamma/hist_avt/{s}")

        dl = sorted_alphanumeric(os.listdir(f"./outputs/grid_gamma/{s}"))
        for ch in dl:
            if ch.endswith(".pt"):
                path = os.path.join(f"./outputs/grid_gamma/{s}", ch)
                config['pretrained'] = path

                predictions, ground_truth, oods, aleatoric, epistemic = eval(config, False, 'val', split, dataroot)
                uncertainty_scores = aleatoric.squeeze(1)
                uncertainty_labels = torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()
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

