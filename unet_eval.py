import os

import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torchvision
from PIL import Image

from other_unet.unet_model import UNetSegmentationModel
from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets
from modules.MonuSeg_model import MonuSegModel
from modules.MonuSeg_evaluator import IOU_Evaluator, MonuSegEvaluator
import warnings
from config import *


warnings.filterwarnings("ignore")


def numpy_array_to_image(array):
    # Assuming array is a 2D or 3D NumPy array
    if len(array.shape) == 2:
        # For grayscale images
        img_array = array
    elif len(array.shape) == 3:
        # For single-channel images (e.g., (H, W, C))
        img_array = array.squeeze()
    else:
        raise ValueError("Unsupported array shape")

    # Rescale values to the range [0, 255]
    img_array = (img_array * 255).astype(np.uint8)

    # Create a PIL Image from the array
    img = Image.fromarray(img_array)

    return img


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
    metric_evaluators = monuseg_evaluator.multi_metric_evaluators()

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

        df = pd.DataFrame(columns=['ImageFile', 'Prediction', 'GroundTruth'])

        for idx, pred in enumerate(predictions):
            label = ground_truth[idx]
            if dl_name == "valid":
                img_file = datasets['valid'].img_list[idx]
            else:
                img_file = datasets['test'].img_list[idx]

            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()
            df = df._append({'ImageFile': img_file, 'Prediction': pred_np, 'GroundTruth': label_np,
                             }, ignore_index=True)

        np.save(os.path.join(
            SAVE_DIR, f"monuseg_{dl_name}_predictions.npy"), df.to_numpy())

        # Evaluate the AUC metrics
        print(f"\nAUC Evaluation Results on {dl_name} dataset:")
        print(f"{'Mean IOU':<25} {'Pixel Accuracy':<15}\n")

        iou_evaluator.update(predictions.to(DEVICE), ground_truth.to(DEVICE))
        predictions = predictions.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)
        iou_evaluator.update(predictions, ground_truth)
        mean_iou = iou_evaluator.get_mean_iou()

        predictions = (predictions > 0.5).float()
        correct_pixels, total_pixels = pixel_accuracy_evaluator.total_pixels(
            predictions=predictions, masks=ground_truth)
        pixel_acc = pixel_accuracy_evaluator.get_pixel_accuracy(
            correct_pixels, total_pixels)

        print(f"{mean_iou:<25} {pixel_acc:<15}")

        print(f"\nOther Evaluation Results on {dl_name} dataset:")

        print(f"{'Metric':<25} {'Value':<25}\n")

        for metric_name, metric_evaluator in metric_evaluators.items():
            metric_value = metric_evaluator(predictions.cpu().numpy(), ground_truth.cpu().numpy())

            # Format the metric value based on the metric name
            formatted_value = f"{metric_value:.4f}" if metric_name != 'accuracy' else f"{metric_value:.2%}"

            print(f"{metric_name.capitalize():<25} {formatted_value:<25}")

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
                 y=valid_accuracies, ax=ax2, label="Valid accuracy (mIOU)")
    sns.lineplot(x=range(len(pixel_accuracies)),
                 y=pixel_accuracies, ax=ax2, label="Pixel accuracy")
    ax2.set_title("Best Model's Accuracies Over Epochs")

    plt.show()

    # Show a sample prediction
    loaded_data = np.load(os.path.join(
        SAVE_DIR, "monuseg_valid_predictions.npy"), allow_pickle=True)

    loaded_df = pd.DataFrame(loaded_data, columns=[
        'ImageFile', 'Prediction', 'GroundTruth'])

    # Assuming you want to visualize the first row of the DataFrame
    row_index = 0
    img_file = loaded_df.loc[row_index, 'ImageFile']
    prediction = loaded_df.loc[row_index, 'Prediction']
    ground_truth = loaded_df.loc[row_index, 'GroundTruth']

    prediction_img = numpy_array_to_image(prediction)
    ground_truth_img = numpy_array_to_image(ground_truth)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Assuming img_file is a path to the image
    axes[0].imshow(plt.imread(img_file))
    axes[0].set_title('Original Image')

    axes[1].imshow(prediction_img, cmap='gray')
    axes[1].set_title('Prediction')

    axes[2].imshow(ground_truth_img, cmap='gray')
    axes[2].set_title('Ground Truth')

    plt.show()


if __name__ == "__main__":
    run("data")
