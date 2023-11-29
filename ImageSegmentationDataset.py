import torch
import os
import xml.etree.ElementTree as et
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torchvision.transforms as transforms

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
                    masked_image.save(f'{self.masks_path}/{annot[:-4]}.png')
                    self.masks_list.append(f'{self.masks_path}/{annot[:-4]}.png')
        else:
            mask_path = self.masks_path
            for mask in os.listdir(mask_path):
                self.masks_list.append(os.path.join(mask_path, mask))

        print("Masks generated")
        print("Length of masks list: ", len(self.masks_list))
        print("Length of images list: ", len(self.img_list))


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