from time import time

import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.nuscenes import compile_data
from models.baseline import Baseline
from models.evidential import Evidential
from tools.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)


def get_loader_info(model, loader):
    predictions = []
    ground_truth = []
    aleatoric = []
    epistemic = []

    with torch.no_grad():
        for images, intrinsics, extrinsics, labels in tqdm(loader):
            outs = model(images, intrinsics, extrinsics)

            print(torch.mean(outs))

            predictions.append(model.activate(outs).detach().cpu())
            ground_truth.append(labels)
            aleatoric.append(model.aleatoric(outs).detach().cpu())
            epistemic.append(model.epistemic(outs).detach().cpu())

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0))


models = {
    'baseline': Baseline,
    'evidential': Evidential
}


def train():
    n_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    train_loader, val_loader = compile_data(
        split,
        DATAROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))

    step = 0

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        for images, intrinsics, extrinsics, labels in train_loader:
            t_0 = time()

            outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

            step += 1

            if step % 10 == 0:
                print(f"[{epoch}] {step}", loss.item())

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

                save_pred(preds, labels, config['logdir'])

            if step % 50 == 0:
                iou = get_iou(preds.cpu(), labels)

                print(f"[{epoch}] {step}", "IOU: ", iou)

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        model.eval()

        predictions, ground_truth, aleatoric, epistemic = get_loader_info(model, val_loader)

        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

        print(f"Validation mIOU: {iou}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-l', '--logdir', required=False)
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-s', '--split', default="trainval", required=False)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    split = args.split
    DATAROOT = "../data/nuscenes"

    train()
