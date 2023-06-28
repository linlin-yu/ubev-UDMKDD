from time import time

from rich.progress import Progress
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.nuscenes import compile_data as compile_data_nuscenes
from datasets.carla import compile_data as compile_data_carla

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
        with Progress(transient=True) as progress:
            task = progress.add_task(f"[cyan]Running validation...", total=len(loader) / config['batch_size'])

            for images, intrinsics, extrinsics, labels in loader:
                outs = model(images, intrinsics, extrinsics)

                predictions.append(model.activate(outs).detach().cpu())
                ground_truth.append(labels)
                aleatoric.append(model.aleatoric(outs).detach().cpu())
                epistemic.append(model.epistemic(outs).detach().cpu())

            progress.update(task, advance=1)

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0))


models = {
    'baseline': Baseline,
    'evidential': Evidential
}

datasets = {
    'nuscenes': compile_data_nuscenes,
    'carla': compile_data_carla
}


def train():
    n_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    train_loader, val_loader = datasets[config['dataset']](
        split, DATAROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss']
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    display_config(config)

    writer = SummaryWriter(logdir=config['logdir'])
    writer.add_text("config", str(config))

    step = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        with Progress(transient=True) as progress:
            task = progress.add_task(f"[cyan]Running epoch...", total=len(train_loader) / config['batch_size'])

            for images, intrinsics, extrinsics, labels in train_loader:
                t_0 = time()

                outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

                step += 1

                head = f"[green][{epoch}][/green] [light_salmon3]{step}[/light_salmon3]"
                if step % 10 == 0:
                    print(f"{head} {loss.item()}")

                    writer.add_scalar('train/step_time', time() - t_0, step)
                    writer.add_scalar('train/loss', loss, step)

                    save_pred(preds, labels, config['logdir'])

                if step % 50 == 0:
                    iou = get_iou(preds.cpu(), labels)

                    print(f"{head} [red]mIOU:[/red] {iou}")

                    for i in range(0, n_classes):
                        writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

                progress.update(task, advance=1)

        model.eval()

        predictions, ground_truth, aleatoric, epistemic = get_loader_info(model, val_loader)

        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

        print(f"[green][{epoch}][/green] [red]Val mIOU:[/red]: {iou}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)
    parser.add_argument('--loss', default="ce", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    split = args.split

    if config['dataset'] == 'carla':
        DATAROOT = "../data/carla"
    else:
        DATAROOT = "../data/nuscenes"

    train()
