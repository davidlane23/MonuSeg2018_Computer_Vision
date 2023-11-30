import os

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as T

from other_unet.unet_model import UNetSegmentationModel
from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets
from modules.MonuSeg_model import MonuSegModel
from modules.MonuSeg_evaluator import IOU_Evaluator, MonuSegEvaluator
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
        T.Resize(256),
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
        lr, epoch, measure, pix_accuracy = lines[1].strip().split(
            ",")  # Add more columns

    print("\nLoading best model...")
    print(f"Learning Rate: {lr}")
    print(f"Epoch: {epoch}")

    monuseg_evaluator = MonuSegEvaluator(N_CLASSES)
    iou_evaluator, pixel_accuracy_evaluator = monuseg_evaluator.auc_evaluators()

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
        iou_evaluator.update(predictions.to(DEVICE), ground_truth.to(DEVICE))
        mean_iou = iou_evaluator.get_mean_iou()
        print("Mean IOU is: ", mean_iou)

        predictions = (predictions > 0.5).float()
        correct_pixels, total_pixels = pixel_accuracy_evaluator.total_pixels(
            predictions=predictions, masks=ground_truth)
        pixel_acc = pixel_accuracy_evaluator.get_pixel_accuracy(
            correct_pixels, total_pixels)
        print("Pixel Acc: ", pixel_acc)

    """
        Plot training and validation losses from saved files
    """
    # read in losses data
    with open(os.path.join(SAVE_DIR, "monuseg_train_losses.txt")) as file:
        train_losses = [float(loss)
                        for loss in file.readline().strip().split(",")]
    with open(os.path.join(SAVE_DIR, "monuseg_valid_losses.txt")) as file:
        valid_losses = [float(loss)
                        for loss in file.readline().strip().split(",")]

    with open(os.path.join(SAVE_DIR, "monuseg_valid_accuracies.txt")) as file:
        valid_accuracies = [float(acc)
                            for acc in file.readline().strip().split(",")]

    with open(os.path.join(SAVE_DIR, "monuseg_pixel_accuracies.txt")) as file:
        pixel_accuracies = [float(acc)
                            for acc in file.readline().strip().split(",")]

    # Plot losses
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))

    # Plot train and validation losses
    sns.lineplot(x=range(len(train_losses)),
                 y=train_losses, ax=ax1, label="Train loss")
    sns.lineplot(x=range(len(valid_losses)),
                 y=valid_losses, ax=ax1, label="Valid loss")
    ax1.set_title("Best Model's Average Losses Over Epochs")

    # Plot accuracies
    sns.lineplot(x=range(len(valid_accuracies)),
                 y=valid_accuracies, ax=ax2, label="Valid accuracy")
    sns.lineplot(x=range(len(pixel_accuracies)),
                 y=pixel_accuracies, ax=ax2, label="Pixel accuracy")
    ax2.set_title("Best Model's Accuracies Over Epochs")

    plt.show()


if __name__ == "__main__":
    run("data")
