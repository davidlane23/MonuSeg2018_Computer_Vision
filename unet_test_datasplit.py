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

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# from UNET import UNetSegmentationModel
# Define the U-Net model for segmentation


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

        # Normalize the images
        img_mean, img_std = img.mean(), img.std()
        img = (img - img_mean) / img_std

        mask[mask <= 0] = 0
        mask[mask > 0] = 255

        # Resize the mask to match the output size of the model
        resize_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        mask = resize_transform(Image.fromarray(mask))

        img = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            img = self.transform(img)

        return img, mask


trg_transforms1 = transforms.Compose([
    transforms.Resize(128),
    # transforms.RandomCrop(250),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [1/255],std=[1/255])
])

val_transforms1 = transforms.Compose([
    transforms.Resize(128),
    # transforms.CenterCrop(250),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [1/255],std=[1/255])
])


def train_modelcv(dataloader_cvtrain, dataloader_cvval, model, criterion, optimizer, scheduler, num_epochs, device):
    best_measure = 0
    best_epoch = -1
    best_loss = 100000
    val_losses = []
    train_losses = []
    best_val_loss = 10000
    for epoch in range(num_epochs):
        print(f'------------------------CURRENT EPOCH: {epoch + 1}--------------------------')
        train_loss = train_model(model, dataloader_cvtrain, criterion, optimizer, device)
        val_loss = evaluate(model, dataloader_cvval, loss_crit, device)
        # Get the train and val losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (best_val_loss > val_loss):
            best_epoch = epoch
            best_val_loss = val_loss
            # best_dice_loss = dice_loss
            bestweights = model.state_dict()
            print(f"Current best Epoch: {epoch + 1}, Val Loss at Epoch: {val_loss}")
        print(f"Epoch {epoch + 1} Val Loss: {val_loss}")
        print(f"Epoch {epoch + 1} Train Loss: {train_loss}")
    return train_losses, val_losses, best_epoch, bestweights


def evaluate(model, dataloader, criterion, device):
    model.eval()
    datasize = 0
    accuracy = 0
    avgloss = 0
    idx = 0
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            outputs[outputs <= 0.65] = 0
            outputs[outputs > 0.65] = 255
            # print(outputs)
            if criterion is not None:
                curloss = criterion(outputs, masks)
                avgloss = (avgloss * datasize + curloss) / (datasize + inputs.shape[0])
            # accuracy = (accuracy * datasize + torch.sum(preds == labels_idx)) / (datasize + inputs.shape[0])
            datasize += inputs.shape[0]
            # get probabilities
            # out_tensor = torch.argmax(outputs, 0)
            # normalized_masks = torch.log_softmax(out_tensor, dim=1)
            # print(out_tensor)
            # for i in outputs:
            #     torchvision.transforms.ToPILImage()(i.type(torch.uint8)).convert('RGB').show()
            # preds = torch.argmax(softmax_pred, dim=1).data
    return avgloss


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


def split_data(file_tissue_mapping, seed=42):
    # Get a list of all file names and tissue types
    file_names = list(file_tissue_mapping.keys())
    tissue_types = list(file_tissue_mapping.values())

    # Split into training, validation, and test sets
    train_data, test_data = train_test_split(file_names, test_size=3, random_state=seed)

    # Ensure 'Colon' files are in the test set
    colon_files = [file_name for file_name, tissue_type in file_tissue_mapping.items() if tissue_type == 'Colon']
    test_data += colon_files[:2]

    # Remove 'Colon' files from the training set
    train_data = [file_name for file_name in train_data if file_name not in test_data]

    # Select random files for validation
    val_data = train_test_split(train_data, test_size=6, random_state=seed)[1]

    # Remove val data from train data
    train_data = [file_name for file_name in train_data if file_name not in val_data]

    print("Train Data", len(train_data))
    print("Val Data", len(val_data))
    print("Test Data", len(test_data))

    return train_data, val_data, test_data


def generate_datasets(train_root, val_root, trg_transforms, val_transforms):
    # Print out all file names in tissue_image
    tissue_image_dir = os.path.join(train_root, "tissue_image")
    all_tissue_images = os.listdir(tissue_image_dir)

    # print("All file names in tissue_image:")
    # for img_name in all_tissue_images:
    #     print(img_name)

    # File paths = All the files in the directory /tissue_image
    file_paths = []

    tissue_types = [
        'Liver',  # TCGA-18-5592-01Z-00-DX1.tif
        'Liver',  # TCGA-21-5784-01Z-00-DX1.tif
        'Liver',  # TCGA-21-5786-01Z-00-DX1.tif
        'Liver',  # TCGA-38-6178-01Z-00-DX1.tif
        'Liver',  # TCGA-49-4488-01Z-00-DX1.tif
        'Liver',  # TCGA-50-5931-01Z-00-DX1.tif
        'Breast',  # TCGA-A7-A13E-01Z-00-DX1.tif
        'Breast',  # TCGA-A7-A13F-01Z-00-DX1.tif
        'Breast',  # TCGA-AR-A1AK-01Z-00-DX1.tif
        'Breast',  # TCGA-AR-A1AS-01Z-00-DX1.tif
        'Colon',  # TCGA-AY-A8YK-01A-01-TS1.tif
        'Kidney',  # TCGA-B0-5698-01Z-00-DX1.tif
        'Kidney',  # TCGA-B0-5710-01Z-00-DX1.tif
        'Kidney',  # TCGA-B0-5711-01Z-00-DX1.tif
        'Kidney',  # TCGA-BC-A217-01Z-00-DX1.tif
        'Prostate',  # TCGA-CH-5767-01Z-00-DX1.tif
        'Bladder',  # TCGA-DK-A2I6-01A-01-TS1.tif
        'Liver',  # TCGA-E2-A14V-01Z-00-DX1.tif
        'Liver',  # TCGA-E2-A1B5-01Z-00-DX1.tif
        'Liver',  # TCGA-F9-A8NY-01Z-00-DX1.tif
        'Liver',  # TCGA-FG-A87N-01Z-00-DX1.tif
        'Bladder',  # TCGA-G2-A2EK-01A-02-TSB.tif
        'Prostate',  # TCGA-G9-6336-01Z-00-DX1.tif
        'Prostate',  # TCGA-G9-6348-01Z-00-DX1.tif
        'Prostate',  # TCGA-G9-6356-01Z-00-DX1.tif
        'Prostate',  # TCGA-G9-6362-01Z-00-DX1.tif
        'Prostate',  # TCGA-G9-6363-01Z-00-DX1.tif
        'Kidney',  # TCGA-HE-7128-01Z-00-DX1.tif
        'Kidney',  # TCGA-HE-7129-01Z-00-DX1.tif
        'Kidney',  # TCGA-HE-7130-01Z-00-DX1.tif
        'Stomach',  # TCGA-KB-A93J-01A-01-TS1.tif
        'Liver',  # TCGA-MH-A561-01Z-00-DX1.tif
        'Colon',  # TCGA-NH-A8F7-01A-01-TS1.tif
        'Stomach',  # TCGA-RD-A8N9-01A-01-TS1.tif
        'Liver',  # TCGA-UZ-A9PJ-01Z-00-DX1.tif
        'Liver',  # TCGA-UZ-A9PN-01Z-00-DX1.tif
        'Liver'  # TCGA-XS-A8TJ-01Z-00-DX1.tif
    ]

    # Identify unique tissue types
    unique_tissue_types = set(tissue_types)
    tissue_type_count = {tissue_type: tissue_types.count(tissue_type) for tissue_type in unique_tissue_types}
    print(unique_tissue_types)
    print(tissue_type_count)

    directory_path = 'MoNuSegTrainData\\tissue_image'
    file_tissue_mapping = map_files_to_tissue_types(directory_path, tissue_types)

    print("File-to-Tissue-Type Mapping:")
    for file_name, tissue_type in file_tissue_mapping.items():
        print(f"{file_name}: {tissue_type}")

    # Split data
    train_data, val_data, test_data = split_data(file_tissue_mapping)

    new_train_img_path = os.path.join(train_root, "new_tissue_image")
    os.makedirs(new_train_img_path, exist_ok=True)

    train_img_path = os.path.join(train_root, "tissue_image")
    train_data_path_mapping = {file_name: os.path.join(train_img_path, file_name) for file_name in train_data}
    # stop here

    train_annot_path = os.path.join(train_root, "annotations")
    train_mask_path = os.path.join(train_root, "masks")

    val_img_path = os.path.join(val_root, "tissue_image")
    valid_data_path_mapping = {file_name: os.path.join(val_img_path, file_name) for file_name in train_data}

    val_annot_path = os.path.join(val_root, "annotations")
    val_mask_path = os.path.join(val_root, "masks")

    # Define paths for masks in both training and validation datasets
    train_mask_path = os.path.join(train_root, "masks")
    val_mask_path = os.path.join(val_root, "masks")

    train_dataset = ImageSegmentationDataset(train_data_path_mapping, train_annot_path, train_mask_path, trg_transforms)

    train_dataset.generate_masks()
    val_dataset = ImageSegmentationDataset(valid_data_path_mapping, val_annot_path, val_mask_path, val_transforms)

    val_dataset.generate_masks()
    # Return both datasets
    return train_dataset, val_dataset


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_root = "MonuSegTrainData"
    val_root = "MonuSegTestData"

    # Instantiate the U-Net segmentation model
    model = UNetSegmentationModel(in_channels=3, out_channels=1)
    model.to(device)

    train_dataset, val_dataset = generate_datasets(train_root, val_root, trg_transforms1, val_transforms1)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    lr_list = [0.001]
    # weighted_loss = torch.tensor([0.1, 0.9])

    for lr in lr_list:
        print("##################### NEW RUN ###########################")
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

        loss_crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # loss_crit = compute_dice_loss()
        # loss_crit = compute_dice_loss()

        best_hyperparameter = None
        weights_chosen = None
        bestmeasure = None
        best_epoch = None

        train_losses, val_losses, best_epoch, weights_chosen = train_modelcv( \
            model=model,
            dataloader_cvtrain=dataloaders['train'],
            dataloader_cvval=dataloaders['val'],
            criterion=loss_crit,
            scheduler=None,
            optimizer=optimizer,
            num_epochs=10,
            device=device
        )
        model.load_state_dict(weights_chosen)
        test_img = Image.open("MoNuSegTestData\\tissue_image\\TCGA-44-2665-01B-06-BS6.tif").convert('RGB')
        test_img = torchvision.transforms.ToTensor()(test_img)
        test_img = torchvision.transforms.Resize(256)(test_img)
        test_img = test_img.to(device)

        output = model(test_img.unsqueeze(0))
        print(type(output))
        print(output)
        output = output.squeeze(0)
        # output = output.detach().cpu().numpy()
        output_img = torchvision.transforms.ToPILImage()(output.type(torch.uint8)).convert('RGB').show()
        # output_img.show()

        # print(avg_loss)
