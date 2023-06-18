
from train import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)


def eval():
    n_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    train_loader, val_loader = compile_data(
        split,
        DATAROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    model = models[config['type']](config['gpus'], n_classes=n_classes)
    model.load(torch.load(config['pretrained']))

    print("--------------------------------------------------")
    print(f"Running eval on {split}")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Pretrained: {config['pretrained']} ")
    print("--------------------------------------------------")

    model.eval()

    predictions, ground_truth, aleatoric, epistemic = get_loader_info(model, val_loader)

    iou = get_iou(predictions, ground_truth)

    print(f"mIOU: {iou}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="mini", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    split = args.split
    DATAROOT = "../data/nuscenes"

    eval()
