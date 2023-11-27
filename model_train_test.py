import cv2
import os
import xml.etree.ElementTree as et

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from joblib._multiprocessing_helpers import mp
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, annot_path, mask_path, transforms):
        self.transform = transforms
        self.img_path = img_path
        self.annotations_path = annot_path
        self.masks_path = mask_path

        self.img_list = []
        self.annotations_list = []
        self.masks_list = []
        for img in os.listdir(img_path):
            self.img_list.append(os.path.join(img_path, img))
        for annot in os.listdir(self.annotations_path):
            self.annotations_list.append(os.path.join(annot_path, annot))

    def generate_masks(self):
        # print(self.annotations_path)
        if (os.listdir(self.masks_path) == []):
            annot_path = self.annotations_path
            for annot in os.listdir(annot_path):
                # Randomly appearing desktop.ini, need to skip
                if (annot == "desktop.ini"):
                    pass
                else:
                    print(os.path.join(annot_path, annot))
                    tree = et.parse(os.path.join(annot_path, annot))
                    root = tree.getroot()
                    maskes = []
                    img_size = (1000, 1000)
                    # Create empty image
                    masked_image = Image.new("LA", img_size, color=(0, 0))
                    # Get the Mask of the images
                    for region in root.findall(".//Region"):
                        vertices = region.findall(".//Vertex")
                        mask = [(float(vertex.get("X")), float(vertex.get("Y"))) for vertex in vertices]
                        maskes.append(mask)
                    # Generate Mask for each image from the annotations
                    #
                    print(self.masks_path)
                    for mask in maskes:
                        draw = ImageDraw.Draw(masked_image)
                        draw.polygon(mask, fill=(255, 255))
                    image_array = np.array(masked_image)
                    masked_image.save(f'{self.masks_path}\\{annot[:-4]}.png')
                    self.masks_list.append(f'{self.masks_path}\\{annot[:-4]}.png')
                    # self.masks_list.append(maskes)
        else:
            mask_path = self.masks_path
            # print(mask_path)
            for mask in os.listdir(mask_path):
                self.masks_list.append(os.path.join(mask_path, mask))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.masks_list[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as grayscale
        mask[mask <= 0] = 0
        mask[mask >= 1] = 1

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        # Convert mask to tensor
        mask = torch.tensor(np.array(mask), dtype=torch.float32)

        # Assuming that the image has shape (C, H, W)
        _, h, w = img.shape

        # Dummy bounding box
        target = {
            'boxes': torch.tensor([[0, 0, w, h]], dtype=torch.float32),  # Adjust as needed
            'labels': torch.tensor([1], dtype=torch.int64),  # Adjust as needed
            'masks': mask
        }

        return img, target


# # Image augmentation (Extra augmentations to consider)
# augmentation = iaa.SomeOf((0, 2), [
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.OneOf([iaa.Affine(rotate=90),
#                 iaa.Affine(rotate=180),
#                 iaa.Affine(rotate=270)]),
#     iaa.GaussianBlur(sigma=(0.0, 5.0))
# ])

trg_transforms1 = (transforms.Compose
    ([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

val_transforms1 = (transforms.Compose
    ([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    data_size = 0
    avg_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Iterate over the batch dimension
        for i in range(images.size(0)):
            # Get individual images from the batch
            image = images[i:i + 1]  # Slice to get a tensor of shape [1, C, H, W]
            mask = masks[i:i + 1]  # Slice to get a tensor of shape [1, H, W]

            # Rest of your code for processing individual images goes here

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = (avg_loss * data_size + loss.item()) / (data_size + images.size(0))
        data_size += images.size(0)

    return avg_loss


def generate_datasets(train_root, val_root, trg_transforms, val_transforms):
    train_img_path = os.path.join(train_root, "tissue_image")
    train_annot_path = os.path.join(train_root, "annotations")
    train_mask_path = os.path.join(train_root, "masks")
    val_img_path = os.path.join(val_root, "tissue_image")
    val_annot_path = os.path.join(val_root, "annotations")
    val_mask_path = os.path.join(val_root, "masks")
    train_dataset = ImageSegmentationDataset(train_img_path, train_annot_path, train_mask_path, trg_transforms)
    train_dataset.generate_masks()
    val_dataset = ImageSegmentationDataset(val_img_path, val_annot_path, val_mask_path, val_transforms)
    val_dataset.generate_masks()
    return train_dataset, val_dataset


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)

    # Modify the number of output classes in the box predictor
    num_classes = 2  # Change this based on your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Popular dim_reducd value

    model.roi_heads.box_predictor = MaskRCNNPredictor(in_channels=in_features, num_classes=num_classes, dim_reduced=256)

    # Create the training dataset and dataloader
    train_root = "MonuSegTrainData"
    val_root = "MonuSegTestData"

    train_dataset, val_dataset = generate_datasets(train_root, val_root, trg_transforms1, val_transforms1)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

    # Set up optimizer and criterion
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = BCEWithLogitsLoss(reduction='mean')

    # Train the Mask R-CNN model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{"masks": mask.to(device)} for mask in masks]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
