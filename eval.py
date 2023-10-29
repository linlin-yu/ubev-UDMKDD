import seaborn as sns
import torch.nn.functional

from train import *

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
np.random.seed(0)


def eval(config, is_ood, set, split, dataroot):
    train_loader, val_loader = datasets[config['dataset']](
        split, dataroot,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        ood=is_ood
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes
    )

    if set == 'val':
        print("Using validation set")
        loader = val_loader
    elif set == 'train':
        print("Using train set")
        loader = train_loader
    else:
        raise NotImplementedError()

    model.load(torch.load(config['pretrained']))
    model.eval()
    model.training = False

    print("--------------------------------------------------") 
    print(f"Running eval on {split}")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Loader: {len(loader.dataset)}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Pretrained: {config['pretrained']} ")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    predictions = []
    ground_truth = []
    oods = []
    aleatoric = []
    epistemic = []
    logits = []

    with torch.no_grad():
        for images, intrinsics, extrinsics, labels, ood in tqdm(loader, desc="Running validation"):
            outs = model(images, intrinsics, extrinsics).detach().cpu()

            predictions.append(model.activate(outs))
            ground_truth.append(labels)
            oods.append(ood)
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))
            logits.append(outs)

            if is_ood:
                save_unc(model.epistemic(outs)/model.epistemic(outs).max(), ood, config['logdir'])
            else:
                save_unc(model.aleatoric(outs), model.activate(outs).argmax(dim=1) != labels.argmax(dim=1), config['logdir'])

            save_pred(model.activate(outs), labels, config['logdir'])

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(oods, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0),
            torch.cat(logits, dim=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="mini", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-m', '--metric', default="rocpr", required=False)
    parser.add_argument('-r', '--save', default=False, action='store_true')
    parser.add_argument('--set', default="val", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    split = args.split
    is_ood = args.ood
    metric = args.metric
    set = args.set
    dataroot = f"../data/{config['dataset']}"
    name = f"{config['backbone']}_{config['type']}"

    predictions, ground_truth, oods, aleatoric, epistemic, logits = eval(config, is_ood, set, split, dataroot)

    if args.save:
        torch.save(predictions, os.path.join(config['logdir'], 'preds.pt'))
        torch.save(ground_truth, os.path.join(config['logdir'], 'ground_truth.pt'))
        torch.save(oods, os.path.join(config['logdir'], 'oods.pt'))
        torch.save(aleatoric, os.path.join(config['logdir'], 'aleatoric.pt'))
        torch.save(epistemic, os.path.join(config['logdir'], 'epistemic.pt'))
        torch.save(logits, os.path.join(config['logdir'], 'logits.pt'))

    if is_ood:
        uncertainty_scores = epistemic.squeeze(1)
        uncertainty_labels = oods
    else:
        iou = get_iou(predictions, ground_truth)
        print(f"mIOU: {iou}")

        uncertainty_scores = aleatoric.squeeze(1)
        uncertainty_labels = torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()

    if metric == 'patch':
        pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores, uncertainty_labels)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.plot(thresholds, agc, 'g.-', label=f"AU-p(accurate|certain): {au_agc:.3f}")
        ax1.set_xlabel('Uncertainty Percentiles')
        ax1.set_ylabel('p(accurate|certain)')
        ax1.legend(frameon=True)

        ax2.plot(thresholds, ugi, 'r.-', label=f"AU-p(uncertain|inaccurate): {au_ugi:.3f}")
        ax2.set_xlabel('Uncertainty Percentiles')
        ax2.set_ylabel('p(uncertain|inaccurate)')
        ax2.legend(frameon=True)

        ax3.plot(thresholds, pavpu,'b.-', label=f"AU-PAvPU: {au_pavpu:.3f}")
        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
        ax3.legend(frameon=True)

        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"patch_{'o' if is_ood else 'm'}_{name}.png")

        pm = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=uncertainty_scores.mean())
        print(f"AVG-PAvPU: {pm[0]:.3f}, AVG-p(accurate|certain): {pm[1]:.3f}, AVG-P(uncertain|inaccurate): {pm[2]:.3f}")
        print(f"AU-PAvPU: {au_pavpu:.3f}, AU-p(accurate|certain): {au_agc:.3f}, AU-P(uncertain|inaccurate): {au_ugi:.3f}")
    elif metric == "rocpr":
        fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(fpr, tpr, 'b-', label=f'AUROC - {auroc:.3f}')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.tick_params(axis='x', which='both', bottom=True)
        ax1.tick_params(axis='y', which='both', left=True)
        ax1.legend()

        ax2.plot(rec, pr, 'r-', label=f'AUPR - {aupr:.3f}')
        ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', which='both', bottom=True)
        ax2.tick_params(axis='y', which='both', left=True)
        ax2.legend()

        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"rocpr_{'o' if is_ood else 'm'}_{name}.png")

        print(f"AUROC: {auroc:.3f} AUPR: {aupr:.3f}")
    else:
        raise ValueError("Please pick a valid metric.")

    fig.savefig(save_path, bbox_inches='tight')
