import os
import xml.etree.ElementTree as et

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torchvision
from other_unet.unet_model import UNetSegmentationModel
import warnings
warnings.filterwarnings("ignore")
# from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex,JaccardIndex

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
        img_mean,img_std = img.mean(),img.std()
        img = (img-img_mean)/img_std

        # mask_mean,mask_std = mask.mean(),mask.std()
        # mask = (mask-mask_mean)/mask_std

        # Binarize mask truth labels for black and white pixels
        # mask = np.float32(mask)
        mask[mask <= 0] = 0
        mask[mask > 0] = 255


        img = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            img = self.transform(img)
            # Resize the mask to match the output size of the model
            resize_transform = transforms.Compose([transforms.Resize((128, 128),interpolation=Image.NEAREST), transforms.ToTensor()])
            mask = resize_transform(Image.fromarray(mask))
        # print(mask)
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
    best_IOU = 0.0
    best_val_loss = 10000
    for epoch in range(num_epochs):
        print(f'------------------------CURRENT EPOCH: {epoch+1}--------------------------')
        train_loss = train_model(model,dataloader_cvtrain,criterion,optimizer,device)
        val_loss,epoch_iou = evaluate(model,dataloader_cvval,loss_crit,device)
        # Get the train and val losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mean_IOU.append(epoch_iou)
        if(best_val_loss > val_loss):
            best_epoch = epoch
            best_val_loss = val_loss
            best_IOU = epoch_iou
            # best_dice_loss = dice_loss
            bestweights = model.state_dict()
            print(f"Current best Epoch: {epoch+1}, Val Loss at Epoch: {val_loss}")
        print(f"Epoch {epoch+1} Val Loss: {val_loss}")
        print(f"Epoch {epoch+1} Train Loss: {train_loss}")
    return train_losses,val_losses,best_epoch,bestweights,mean_IOU

def evaluate(model, dataloader, criterion, device):
    model.eval()
    datasize = 0
    accuracy = 0
    avgloss = 0
    dice_loss = 0
    mean_iou = []
    # Calculate mean IOU using torchvision Metrics JaccardIndex
    jaccard = JaccardIndex(task='binary',num_classes=2,ignore_index=0).to(device)
    # jaccard = BinaryJaccardIndex(threshold=0.35).to(device)
    with torch.no_grad():
        for inputs,masks in dataloader:
            inputs = inputs.to(device)

            masks[masks <= 0.65] = 0
            masks[masks > 0.65] = 1

            masks = masks.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            mean_iou.append(jaccard(outputs, masks).item())
            # Convert output to binarized predicted mask
            outputs[outputs <= 0.65] = 0
            outputs[outputs > 0.65] = 1

            # get the IoU of each sample

            if criterion is not None:
                curloss = criterion(outputs, masks)
                avgloss = (avgloss * datasize + curloss)  / (datasize + inputs.shape[0])
            datasize += inputs.shape[0]
            # get probabilities
            # out_tensor = torch.argmax(outputs, 0)
            # normalized_masks = torch.log_softmax(out_tensor, dim=1)
            # print(out_tensor)
            # for i in outputs:
            #     torchvision.transforms.ToPILImage()(i.type(torch.uint8)).convert('RGB').show()
            # preds = torch.argmax(softmax_pred, dim=1).data
    # divide by number of samples in batch
    mean_iou = sum(mean_iou)/len(mean_iou)
    print("Mean IOU of epoch: ", mean_iou)
    return avgloss,mean_iou

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

    lr_list = [0.001,0.01]
    # weighted_loss = torch.tensor([0.1, 0.9])

    for lr in lr_list:
        print("##################### NEW RUN ###########################")
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

        loss_crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # loss_crit = compute_dice_loss()
        # loss_crit = compute_dice_loss()

        train_losses, val_losses, best_epoch,weights_chosen,mean_iou = train_modelcv( \
            model=model,
            dataloader_cvtrain=dataloaders['train'],
            dataloader_cvval=dataloaders['val'],
            criterion=loss_crit,
            scheduler=None,
            optimizer=optimizer,
            num_epochs=20,
            device=device
        )
        # Measure chosen for determining best model: Mean IOU (Jaccard Index)
        if(bestmeasure == None):
            best_hyperparameter = lr
            bestmeasure = mean_iou
            bestweights = weights_chosen
            best_model_epoch = best_epoch
        else:
            if(mean_iou >= bestmeasure):
                best_hyperparameter = lr
                bestmeasure = mean_iou
                bestweights = weights_chosen
                best_model_epoch = best_epoch

    print(f"Model Chosen: Best LR:{best_hyperparameter},Best mIOU: {bestmeasure},Best model epoch {best_model_epoch}")
    # Testing
    model.load_state_dict(bestweights)
    test_img = Image.open("MoNuSegTestData\\tissue_image\\TCGA-44-2665-01B-06-BS6.tif").convert('RGB')
    test_img = torchvision.transforms.ToTensor()(test_img)
    test_img = torchvision.transforms.Resize(256)(test_img)
    test_img = test_img.to(device)

    # Testint purpose to see if able to get proper segmentation mask
    output = model(test_img.unsqueeze(0))
    output = output.squeeze(0)
    output_img = torchvision.transforms.ToPILImage()(output.type(torch.uint8)).convert('RGB').show()

        # print(avg_loss)
