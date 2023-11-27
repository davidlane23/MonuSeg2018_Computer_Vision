import os
import xml.etree.ElementTree as et
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

# Define the U-Net model for segmentation
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder(x)
        # Middle
        middle = self.middle(enc1)
        # Decoder
        dec1 = self.decoder(middle)

        return dec1


# Define the U-Net model for segmentation
class UNetSegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSegmentationModel, self).__init__()
        self.unet = UNet(in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)


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
        if os.listdir(self.masks_path) == []:
            annot_path = self.annotations_path
            for annot in os.listdir(annot_path):
                if annot == "desktop.ini":
                    pass
                else:
                    tree = et.parse(os.path.join(annot_path, annot))
                    root = tree.getroot()
                    mask_list = []
                    img_size = (1000, 1000)
                    masked_image = Image.new("LA", img_size, color=(0, 0))
                    for region in root.findall(".//Region"):
                        vertices = region.findall(".//Vertex")
                        mask = [(float(vertex.get("X")), float(vertex.get("Y"))) for vertex in vertices]
                        mask_list.append(mask)
                    for mask in mask_list:
                        draw = ImageDraw.Draw(masked_image)
                        draw.polygon(mask, fill=(255, 255))
                    image_array = np.array(masked_image)
                    masked_image.save(f'{self.masks_path}\\{annot[:-4]}.png')
                    self.masks_list.append(f'{self.masks_path}\\{annot[:-4]}.png')
        else:
            mask_path = self.masks_path
            for mask in os.listdir(mask_path):
                self.masks_list.append(os.path.join(mask_path, mask))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        masks = self.masks_list[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        mask[mask <= 0] = 0
        mask[mask >= 1] = 1

        # Resize the mask to match the output size of the model
        resize_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        mask = resize_transform(Image.fromarray(mask))

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

trg_transforms1 = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
])

val_transforms1 = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])


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


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_root = "MonuSegTrainData"
    val_root = "MonuSegTestData"

    # Instantiate the U-Net segmentation model
    model = UNetSegmentationModel(in_channels=3, out_channels=1)
    model.to(device)

    train_dataset, val_dataset = generate_datasets(train_root, val_root, trg_transforms1, val_transforms1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    lr_list = [0.001]
    for lr in lr_list:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        loss_crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
        avg_loss = train_model(model, dataloaders['train'], loss_crit, optimizer, device)
        print(avg_loss)
