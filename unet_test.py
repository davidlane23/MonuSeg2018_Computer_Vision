import os
import shutil
import xml.etree.ElementTree as et

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
from other_unet.unet_model import UNetSegmentationModel
import warnings
from pixelAccuracy_Evaluator import PixelAccuracyEvaluator
from iou_Evaluator import IOU_Evaluator

warnings.filterwarnings("ignore")
# from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex
from ImageSegmentationDataset import ImageSegmentationDataset

# Set random seed for reproducibility
state = 42

# from UNET import UNetSegmentationModel
trg_transforms1 = transforms.Compose([
    transforms.Resize(128),
    # transforms.RandomCrop(128),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [1/255],std=[1/255])
])

val_transforms1 = transforms.Compose([
    transforms.Resize(128),
    # transforms.CenterCrop(128),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [1/255],std=[1/255])
])


def train_modelcv(dataloader_cvtrain, dataloader_cvval, model, criterion, optimizer, scheduler, num_epochs, device,
                  iou_eval):
    best_measure = 0
    best_epoch = -1
    best_loss = 100000
    val_losses = []
    train_losses = []
    best_IOU = 0.0
    best_val_loss = 10000
    for epoch in range(num_epochs):
        print(f'------------------------CURRENT EPOCH: {epoch + 1}--------------------------')
        train_loss = train_model(model, dataloader_cvtrain, criterion, optimizer, device)
        val_loss, epoch_iou, pixel_accuracy = evaluate(model, dataloader_cvval, loss_crit, device)
        # Get the train and val losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # reset the mean IOU for each epoch for new calculation
        iou_eval.reset()
        # use meanIOU as the best measure
        if (best_IOU < epoch_iou):
            # if(best_val_loss > val_loss):
            best_epoch = epoch
            best_val_loss = val_loss
            best_IOU = epoch_iou
            bestweights = model.state_dict()
            print(f"Current best Epoch: {epoch + 1}, Val Loss at Epoch: {val_loss}")
        print(f"Epoch {epoch + 1} Val Loss: {val_loss}")
        print(f"Epoch {epoch + 1} Train Loss: {train_loss}")
        print(f"Epoch {epoch + 1} Pixel Acc: {pixel_accuracy}")

    return train_losses, val_losses, best_epoch, bestweights, best_IOU


def evaluate(model, dataloader, criterion, device):
    model.eval()
    datasize = 0
    avgloss = 0
    mean_iou = []

    correct_pixels = 0
    total_pixels = 0

    # Calculate mean IOU using torchvision Metrics JaccardIndex
    jaccard = JaccardIndex(task='binary', num_classes=1, ignore_index=0).to(device)
    # jaccard = BinaryJaccardIndex(threshold=0.35).to(device)
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)

            masks[masks <= 0.65] = 0
            masks[masks > 0.65] = 1

            masks = masks.to(device)
            outputs = model(inputs)

            predictions = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary segmentation

            outputs = torch.sigmoid(outputs)

            # mean_iou.append(jaccard(outputs, masks).item())
            # Convert output to binarized predicted mask
            outputs[outputs <= 0.65] = 0
            outputs[outputs > 0.65] = 1

            # get the IoU of each sample
            iou_evaluator.update(outputs, masks)

            # get the correct and total number of pixels of each sample
            correct_pixels, total_pixels = pixel_accuracy_evaluator.total_pixels(masks, predictions)

            if criterion is not None:
                curloss = criterion(outputs, masks)
                avgloss += curloss.item()
                # avgloss = (avgloss * datasize + curloss)  / (datasize + inputs.shape[0])
            datasize += inputs.shape[0]
            # get probabilities
            # out_tensor = torch.argmax(outputs, 0)
            # normalized_masks = torch.log_softmax(out_tensor, dim=1)
            # print(out_tensor)
    avgloss /= datasize
    # divide by number of samples in batch
    mean_iou = iou_evaluator.getMeanIOU()
    pixel_acc = pixel_accuracy_evaluator.get_pixel_accuracy(correct_pixels, total_pixels)
    # mean_iou = sum(mean_iou)/len(mean_iou)
    print("Mean IOU of epoch: ", mean_iou)
    print("Pixel Accuracy: ", pixel_acc)
    return avgloss, mean_iou, pixel_acc


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    data_size = 0
    avg_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        out_tensor = outputs
        loss = criterion(out_tensor, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = (avg_loss * data_size + loss.item()) / (data_size + images.size(0))
        data_size += images.size(0)

    return avg_loss


def compute_dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def map_files_to_tissue_types(directory_path, tissue_types):
    # Get all file names in the specified directory
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Create a mapping between file names and tissue types
    file_tissue_mapping = dict(zip(file_names, tissue_types))

    return file_tissue_mapping


def split_data(file_tissue_mapping, seed=state):
    # This function splits data into an estimated 0.7 , 0.15, 0.15
    # Get a list of all file names and tissue types
    file_names = list(file_tissue_mapping.keys())

    # Split into training, validation, and test sets
    train_data, test_data = train_test_split(file_names, test_size=5, random_state=seed)

    # Ensure 'Colon' files are in the test set
    colon_files = [file_name for file_name, tissue_type in file_tissue_mapping.items() if tissue_type == 'Colon']
    test_data += colon_files[:3]

    # Remove 'Colon' files from the training and val set
    train_data = [file_name for file_name in train_data if file_name not in test_data]

    # Select random files for validation (Further split test data)
    val_data = train_test_split(train_data, test_size=8, random_state=seed)[1]

    # Remove val data from train data
    train_data = [file_name for file_name in train_data if file_name not in val_data]

    return train_data, val_data, test_data


def generate_datasets(train_root, val_root, test_root, trg_transforms, val_transforms):
    # Print out all file names in tissue_image
    source_root = 'MoNuSegData'
    tissue_image_dir = os.path.join(source_root, "tissue_image")
    all_tissue_images = os.listdir(tissue_image_dir)

    print("All file names in tissue_image:")
    for img_name in all_tissue_images:
        print(img_name)

    tissue_types = [
        'Liver',  # TCGA-18-5592-01Z-00-DX1.tif C
        'Liver',  # TCGA-21-5784-01Z-00-DX1.tif C
        'Liver',  # TCGA-21-5786-01Z-00-DX1.tif C
        'Kidney',  # TCGA-2Z-A9J9-01A-01-TS1.tif C
        'Liver',  # TCGA-38-6178-01Z-00-DX1.tif C
        'Lung',  # TCGA-44-2665-01B-06-BS6.tif C
        'Liver',  # TCGA-49-4488-01Z-00-DX1.tif C
        'Liver',  # TCGA-50-5931-01Z-00-DX1.tif C
        'Lung',  # TCGA-69-7764-01A-01-TS1.tif C
        'Colon',  # TCGA-A6-6782-01A-01-BS1.tif C
        'Breast',  # TCGA-A7-A13E-01Z-00-DX1.tif C
        'Breast',  # TCGA-A7-A13F-01Z-00-DX1.tif C
        'Breast',  # TCGA-AC-A2FO-01A-01-TS1.tif C
        'Breast',  # TCGA-AO-A0J2-01A-01-BSA.tif C
        'Breast',  # TCGA-AR-A1AK-01Z-00-DX1.tif C
        'Breast',  # TCGA-AR-A1AS-01Z-00-DX1.tif C
        'Colon',  # TCGA-AY-A8YK-01A-01-TS1.tif C
        'Kidney',  # TCGA-B0-5698-01Z-00-DX1.tif C
        'Kidney',  # TCGA-B0-5710-01Z-00-DX1.tif C
        'Kidney',  # TCGA-B0-5711-01Z-00-DX1.tif C
        'Unlabelled',  # TCGA-BC-A217-01Z-00-DX1.tif W
        'Prostate',  # TCGA-CH-5767-01Z-00-DX1.tif C
        'Bladder',  # TCGA-ZF-A9R5-01A-01-TS1.tif C
        'Bladder',  # TCGA-DK-A2I6-01A-01-TS1.tif C
        'Breast',  # TCGA-E2-A1B5-01Z-00-DX1.tif W
        'Breast',  # # TCGA-E2-A14V-01Z-00-DX1.tif W
        'Prostate',  # TCGA-EJ-A46H-01A-03-TSC.tif C
        'Unlabelled',  # TCGA-F9-A8NY-01Z-00-DX1.tif W
        'Brain',  # TCGA-FG-A4MU-01B-01-TS1.tif C
        'Unlabelled',  # TCGA-FG-A87N-01Z-00-DX1.tif W
        'Bladder',  # TCGA-G2-A2EK-01A-02-TSB.tif W
        'Prostate',  # TCGA-G9-6336-01Z-00-DX1.tif C
        'Prostate',  # TCGA-G9-6348-01Z-00-DX1.tif C
        'Prostate',  # TCGA-G9-6356-01Z-00-DX1.tif C
        'Prostate',  # TCGA-G9-6362-01Z-00-DX1.tif C
        'Prostate',  # TCGA-G9-6363-01Z-00-DX1.tif C
        'Kidney',  # TCGA-HT-8564-01Z-00-DX1.tif C
        'Prostate',  # TCGA-IZ-8196-01A-01-BS1.tif C
        'Kidney',  # TCGA-HE-7128-01Z-00-DX1.tif C
        'Kidney',  # TCGA-HE-7129-01Z-00-DX1.tif C
        'Kidney',  # TCGA-HE-7130-01Z-00-DX1.tif C
        'Brain',  # TCGA-GL-6846-01A-01-BS1.tif C
        'Kidney',  # TCGA-HC-7209-01A-01-TS1.tif C
        'Stomach',  # TCGA-KB-A93J-01A-01-TS1.tif C
        'Unlabelled',  # TCGA-MH-A561-01Z-00-DX1.tif W
        'Colon',  # TCGA-NH-A8F7-01A-01-TS1.tif C
        'Stomach',  # TCGA-RD-A8N9-01A-01-TS1.tif C
        'Unlabelled',  # TCGA-UZ-A9PJ-01Z-00-DX1.tif W
        'Unlabelled',  # TCGA-UZ-A9PN-01Z-00-DX1.tif W
        'Unlabelled',  # TCGA-XS-A8TJ-01Z-00-DX1.tif W
        'Bladder',  # TCGA-CU-A0YN-01A-02-BSB.tif C
    ]

    # Identify unique tissue types
    unique_tissue_types = set(tissue_types)
    tissue_type_count = {tissue_type: tissue_types.count(tissue_type) for tissue_type in unique_tissue_types}
    print(unique_tissue_types)
    print(tissue_type_count)

    directory_path = 'MoNuSegData\\tissue_image'
    directory_path_annot = 'MoNuSegData\\annotations'
    file_tissue_mapping = map_files_to_tissue_types(directory_path, tissue_types)

    # Split data into train val and test
    train_data, val_data, test_data = split_data(file_tissue_mapping)

    # Ensure data is split properly
    print("Train Data", len(train_data))
    print("Val Data", len(val_data))
    print("Test Data", len(test_data))

    # Define new folders to store train and val data
    new_train_img_path = os.path.join(train_root, "train_tissue_image")
    new_train_annot_path = os.path.join(train_root, "train_tissue_annot")

    new_val_img_path = os.path.join(val_root, "val_tissue_image")
    new_val_annot_path = os.path.join(val_root, "val_tissue_annot")

    new_test_img_path = os.path.join(test_root, "test_tissue_image")
    new_test_annot_path = os.path.join(test_root, "test_tissue_annot")

    new_train_mask_path = os.path.join(train_root, "train_tissue_mask")
    new_val_mask_path = os.path.join(val_root, "val_tissue_mask")
    new_test_mask_path = os.path.join(test_root, "test_tissue_mask")

    # Create these folders
    os.makedirs(new_train_img_path, exist_ok=True)
    os.makedirs(new_train_annot_path, exist_ok=True)

    os.makedirs(new_test_img_path, exist_ok=True)
    os.makedirs(new_test_annot_path, exist_ok=True)

    os.makedirs(new_val_img_path, exist_ok=True)
    os.makedirs(new_val_annot_path, exist_ok=True)

    os.makedirs(new_train_mask_path, exist_ok=True)
    os.makedirs(new_val_mask_path, exist_ok=True)
    os.makedirs(new_test_mask_path, exist_ok=True)

    # Define a dictionary to map data types to their corresponding destination paths
    data_type_paths = {
        'train': (new_train_img_path, new_train_annot_path),
        'val': (new_val_img_path, new_val_annot_path),
        'test': (new_test_img_path, new_test_annot_path)
    }

    # Loop through each data type (train, val, test)
    for data_type, data_files in {'train': train_data, 'val': val_data, 'test': test_data}.items():
        # Get the destination paths for the current data type
        dest_img_path, dest_annot_path = data_type_paths[data_type]

        # Loop through the data files and move them into image and annot folders
        for file_name in data_files:
            # Construct the full path to the source file
            source_path = os.path.join(directory_path, file_name)

            # Construct the full path to the destination file
            dest_path = os.path.join(dest_img_path, file_name)
            source_xml_path = os.path.join(directory_path_annot, f"{os.path.splitext(file_name)[0]}.xml")

            # Construct the full path to the destination XML file
            dest_xml_path = os.path.join(dest_annot_path, f"{os.path.splitext(file_name)[0]}.xml")

            # Move the file to the new directory
            shutil.copy(source_path, dest_path)
            shutil.copy(source_xml_path, dest_xml_path)

    # Define paths for masks in both training and validation datasets

    train_dataset = ImageSegmentationDataset(new_train_img_path, new_train_annot_path, new_train_mask_path,
                                             trg_transforms)
    train_dataset.generate_masks()

    val_dataset = ImageSegmentationDataset(new_val_img_path, new_val_annot_path, new_val_mask_path, val_transforms)
    val_dataset.generate_masks()

    test_dataset = ImageSegmentationDataset(new_test_img_path, new_test_annot_path, new_test_mask_path, val_transforms)
    test_dataset.generate_masks()

    # Return both datasets
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_root = "Train_Data"
    val_root = "Val_Data"
    test_root = "Test_Data"

    train_dataset, val_dataset, test_dataset = generate_datasets(train_root, val_root, test_root, trg_transforms1,
                                                                 val_transforms1)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    best_hyperparameter = None
    weights_chosen = None
    bestmeasure = None
    best_model_epoch = None

    lr_list = [0.001, 0.01]

    for lr in lr_list:
        iou_evaluator = IOU_Evaluator(num_classes=1)
        # Instantiate the U-Net segmentation model
        # Finetune the last layer of the model to output only 2 classes, either background class or cell
        model = UNetSegmentationModel(in_channels=3, out_channels=1)
        model.to(device)

        # Evaluate with pixel accuracy
        pixel_accuracy_evaluator = PixelAccuracyEvaluator(model, val_loader, device)

        # apply pos weight to make positive class more important as image is imbalanced to background classes
        pos_weight = torch.tensor([3.0]).to(device)
        print("##################### NEW RUN ###########################")
        # optimizer = optim.RMSprop(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # loss_crit = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight)
        loss_crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_losses, val_losses, best_epoch, weights_chosen, mean_iou = train_modelcv( \
            model=model,
            dataloader_cvtrain=dataloaders['train'],
            dataloader_cvval=dataloaders['val'],
            criterion=loss_crit,
            scheduler=None,
            optimizer=optimizer,
            num_epochs=20,
            iou_eval=iou_evaluator,
            device=device
        )
        # Measure chosen for determining best model: Mean IOU (Jaccard Index)
        if (bestmeasure == None):
            best_hyperparameter = lr
            bestmeasure = mean_iou
            bestweights = weights_chosen
            best_model_epoch = best_epoch
        else:
            if (mean_iou >= bestmeasure):
                best_hyperparameter = lr
                bestmeasure = mean_iou
                bestweights = weights_chosen
                best_model_epoch = best_epoch

        # Calculate pixel accuracy on the validation set
        pixel_accuracy = pixel_accuracy_evaluator.calculate_pixel_accuracy()
        print(f"Pixel Accuracy on Validation Set: {pixel_accuracy}")

    print(f"Model Chosen: Best LR:{best_hyperparameter},Best mIOU: {bestmeasure},Best model epoch {best_model_epoch}")
    torch.save(bestweights, 'model\\best_model.pth')
    # Testing Phase-----------------------------------
    checkpoint = torch.load(f'model\\best_model.pth', map_location=device)
    model = UNetSegmentationModel(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint)
    model.to(device)
    test_img = Image.open("MoNuSegTestData\\tissue_image\\TCGA-44-2665-01B-06-BS6.tif").convert('RGB')
    test_img = torchvision.transforms.ToTensor()(test_img)
    test_img = torchvision.transforms.Resize(256)(test_img)
    test_img = test_img.to(device)

    # Testint purpose to see if able to get proper segmentation mas     k
    output = model(test_img.unsqueeze(0))
    output = output.squeeze(0)
    output_img = torchvision.transforms.ToPILImage()(output.type(torch.uint8)).convert('RGB').show()
