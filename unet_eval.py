import os

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as T

from other_unet.unet_model import UNetSegmentationModel
from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets
from modules.MonuSeg_model import MonuSegModel
from modules.MonuSeg_evaluator import IOU_Evaluator
import warnings
from config import *


warnings.filterwarnings("ignore")


def run(data_path):
    """
        Setup for evaluation.
    """
    # make results reproducible
    weights = torch.load(os.path.join(
        SAVE_DIR, "monuseg_model.pt"), map_location=DEVICE)
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    data_transform = T.Compose([
        T.Resize(128),
        # T.CenterCrop(60),
        T.ToTensor(),
    ])

    print("Loading dataset...")

    # paths
    valid_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    # UNet model
    model = UNetSegmentationModel(in_channels=3, out_channels=1)
    model.to(DEVICE)

    datasets = {
        'valid': MonuSegDataset(valid_path, data_transform),
        'test': MonuSegDataset(test_path, data_transform)
    }

    datasets['valid'].generate_masks()
    datasets['test'].generate_masks()

    print("Number of samples:")
    print(f"Validation: {len(datasets['valid'])}")
    print(f"Test: {len(datasets['test'])}")

    dataloaders = {
        "valid": torch.utils.data.DataLoader(datasets["valid"], batch_size=BATCH_SIZE, shuffle=False),
        "test": torch.utils.data.DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)
    }

    """
        Initialise model and evaluator.
    """
    with open(os.path.join(SAVE_DIR, "monuseg_params.txt")) as file:
        lines = file.readlines()
        lr, epoch = lines[1].strip().split(",")  # Add more columns

    print("\nLoading best model...")
    print(f"Learning Rate: {lr}")
    print(f"Epoch: {epoch}")

    iou_eval = IOU_Evaluator(N_CLASSES)
    monuseg_model = MonuSegModel(model,
                                 device=DEVICE,
                                 n_classes=N_CLASSES,
                                 epochs=EPOCH,
                                 criterion=loss,
                                 lr=lr,
                                 weights=weights
                                 )

    print(f"\nEvaluating best model...")
    for dl_name, dataloader in dataloaders.items():
        predictions, ground_truth = monuseg_model.predict_batches(
            dataloader=dataloader)
        iou_eval.update(predictions, ground_truth)
        mean_iou = iou_eval.get_mean_iou()

        print("Mean IOU is: ", mean_iou)


if __name__ == "__main__":
    run("data")
