# Import Statements
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets
from modules.MonuSeg_model import MonuSegModel
from other_unet.unet_model import UNetSegmentationModel
import warnings
warnings.filterwarnings("ignore")


def run(data_path):

    # make results reproducible
    np.random.seed(42)
    torch.manual_seed(42)

    # define hyperparameters
    batch_size = 1
    epochs = 10
    lrates = [0.01]
    n_classes = 2
    in_channels = 3
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # apply pos weight to make positive class more important as image is imbalanced to background classes
    pos_weight = torch.tensor([3.0]).to(device)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    save_dir = "MonuSeg_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define transforms

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ]),
    }

    print("Loading dataset...")

    # paths
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    # UNet model
    model = UNetSegmentationModel(in_channels=3, out_channels=1)
    model.to(device)

    # create train and validation datasets
    datasets = {
        'train': MonuSegDataset(train_path, data_transforms['train']),
        'valid': MonuSegDataset(valid_path, data_transforms['valid']),
        'test': MonuSegDataset(test_path, data_transforms['valid'])
    }

    datasets['train'].generate_masks()
    datasets['valid'].generate_masks()
    datasets['test'].generate_masks()

    print("Number of samples:")
    print(f"Train: {len(datasets['train'])}")
    print(f"Validation: {len(datasets['valid'])}")
    print(f"Test: {len(datasets['test'])}")

    datasets["train"].transform = data_transforms["train"]
    datasets["valid"].transform = data_transforms["valid"]
    datasets["test"].transform = data_transforms["test"]

    # create train and validation dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    }

    best_model = {"model": None, "param": None,
                  "epoch": None, "measure": None, "weights": None,
                  "iou": None
                  }

    for lr in lrates:
        print(f"\nTraining model... (lr: {lr})")

        # clear gpu cache
        torch.cuda.empty_cache() if device == 'cuda' else None

        # load model
        monuseg_model = MonuSegModel(model,
                                     device=device,
                                     n_classes=n_classes,
                                     epochs=epochs,
                                     criterion=loss,
                                     lr=lr)

        # fit model
        best_epoch, best_measure, best_weights, best_iou = monuseg_model.fit(
            dataloaders['train'], dataloaders['valid'])

        if best_model["measure"] is None or best_measure > best_model["measure"]:
            best_model["model"] = monuseg_model
            best_model["param"] = lr
            best_model["epoch"] = best_epoch
            best_model["measure"] = best_measure
            best_model["weights"] = best_weights
            best_model['iou'] = best_iou
        print("Best Model IOU", best_model['measure'])
        # save best model
        torch.save(best_model["weights"], os.path.join(
            save_dir, "monuseg_model.pt"))

        with open(os.path.join(save_dir, "monuseg_params.txt"), "w+") as file:
            file.write("parameter,epoch\n")
            file.write(
                ",".join([str(best_model["param"]), str(best_model["epoch"])]))
        # save losses
        with open(os.path.join(save_dir, "monuseg_train_losses.txt"), "w+") as file:
            file.write(",".join(map(str, best_model["model"].train_losses)))

        with open(os.path.join(save_dir, "monuseg_valid_losses.txt"), "w+") as file:
            file.write(",".join(map(str, best_model["model"].valid_losses)))


if __name__ == "__main__":
    run("data")
