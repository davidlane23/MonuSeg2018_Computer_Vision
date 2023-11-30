# Import Statements
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets
from modules.MonuSeg_model import MonuSegModel
from Model.unet_model import UNetSegmentationModel
import warnings
from config import *

warnings.filterwarnings("ignore")


def run(data_path):

    # apply pos weight to make positive class more important as image is imbalanced to background classes
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    save_dir = "MonuSeg_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define transforms

    data_transforms = \
        {
            'train':
            [
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.ColorJitter(1, 1, 1, 0.5),
                    transforms.RandomAdjustSharpness(0, 0.2)]),

                transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.ColorJitter(1, 1, 1, 0.5),
                    transforms.GaussianBlur(3)]),

                transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.ColorJitter(1, 1, 1, 0.5),
                    transforms.RandomSolarize(0.2)])
            ],
            'valid': transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(128),
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
    model.to(DEVICE)

    # create train and validation datasets
    datasets = \
        {
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

    best_model = {"model": None, "param": None,
                  "epoch": None, "measure": None, "weights": None,
                  "pix_accuracy": None, "trg_transform": None
                  }
    for transform_idx in range(len(data_transforms['train'])):
        datasets["train"].transform = data_transforms["train"][transform_idx]
        datasets["valid"].transform = data_transforms["valid"]
        datasets["test"].transform = data_transforms["test"]

        # create train and validation dataloaders
        dataloaders = \
            {
                'train': torch.utils.data.DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
                'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=BATCH_SIZE, shuffle=False),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
            }
        for lr in LRATES:
            print(
                f"\nTraining model... (lr: {lr}), (transform: {transform_idx})")

            # clear gpu cache
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

            # load model
            monuseg_model = MonuSegModel(model,
                                         device=DEVICE,
                                         n_classes=N_CLASSES,
                                         epochs=EPOCH,
                                         criterion=loss,
                                         lr=lr)

            # fit model
            best_epoch, best_measure, best_weights, best_pix_acc = monuseg_model.fit(
                dataloaders['train'], dataloaders['valid'])

            if best_model["measure"] is None or best_measure > best_model["measure"]:
                best_model["model"] = monuseg_model
                best_model["param"] = lr
                best_model["epoch"] = best_epoch
                best_model["measure"] = best_measure
                best_model["weights"] = best_weights
                best_model['pix_accuracy'] = best_pix_acc
                best_model['trg_transform'] = transform_idx
    print("Chosen Model Trained with transform: ", best_model['trg_transform'])
    print("Chosen Model Trained with lr: ", best_model['param'])
    print(f"Chosen Model achieved {best_model['measure']} mIOU")
    # save best model
    torch.save(best_model["weights"], os.path.join(
        save_dir, "monuseg_model.pt"))

    with open(os.path.join(save_dir, "monuseg_params.txt"), "w+") as file:
        file.write("parameter,epoch,measure,pix_accuracy\n")
        file.write(
            ",".join([str(best_model["param"]), str(best_model["epoch"]), str(best_model["measure"]), str(best_model["pix_accuracy"])]))

    # save losses
    with open(os.path.join(save_dir, "monuseg_train_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].train_losses)))

    with open(os.path.join(save_dir, "monuseg_valid_losses.txt"), "w+") as file:
        file.write(",".join(map(str, best_model["model"].valid_losses)))

        # save accuracies
    with open(os.path.join(save_dir, "monuseg_valid_accuracies.txt"), "w+") as file:
        file.write(
            ",".join(map(str, best_model["model"].valid_accuracies)))

    with open(os.path.join(save_dir, "monuseg_pixel_accuracies.txt"), "w+") as file:
        file.write(
            ",".join(map(str, best_model["model"].pixel_accuracies)))


if __name__ == "__main__":
    run("data")
