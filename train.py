import argparse
from time import time

from tensorboardX import SummaryWriter
from tools.metrics import *
from tools.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
np.random.seed(0)


def train():
    if config['loss'] == 'focal':
        config['learning_rate'] *= 4

    train_loader, val_loader = datasets[config['dataset']](
        split, dataroot,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        ood=is_ood,
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss'],
        weights=weights
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    print("Using scheduler")

    if 'pretrained' in config:
        model.load(torch.load(config['pretrained']))
        print(f"Loaded pretrained weights: {config['pretrained']}")
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt,
            div_factor=10,
            pct_start=.3,
            final_div_factor=10,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader.dataset) // config['batch_size']
        )

    if 'gamma' in config:
        model.gamma = config['gamma']
        print(f"GAMMA: {model.gamma}")

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))

    step = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        for images, intrinsics, extrinsics, labels, ood in train_loader:
            t_0 = time()

            if is_ood:
                outs, preds, loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood)
            else:
                outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(f"[{epoch}] {step}", loss.item())

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

                if is_ood:
                    save_unc(model.epistemic(outs), ood, config['logdir'])
                save_pred(preds, labels, config['logdir'])

            if step % 50 == 0:
                iou = get_iou(preds.cpu(), labels)

                print(f"[{epoch}] {step}", "IOU: ", iou)

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        model.eval()

        predictions, ground_truth, oods, aleatoric, epistemic = run_loader(model, val_loader)

        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

        print(f"Validation mIOU: {iou}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-e', '--num_epochs', required=False, type=int)
    parser.add_argument('--loss', default="ce", required=False, type=str)
    parser.add_argument('--gamma', required=False, type=float)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)
    is_ood = args.ood

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True

    split = args.split
    dataroot = f"../data/{config['dataset']}"

    train()
