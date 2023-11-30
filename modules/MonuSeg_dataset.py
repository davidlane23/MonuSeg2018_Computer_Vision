import os
import csv

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import xml.etree.ElementTree as et
import cv2
import torchvision.transforms as transforms
from .utils import tissue_types
from sklearn.model_selection import train_test_split
import shutil


# Define the MonuSeg dataset
class MonuSegDataset(Dataset):
    def __init__(self, root, transforms):
        self.transform = transforms
        self.img_path = os.path.join(root, "tissue_image")
        self.annotations_path = os.path.join(root, "annotations")
        self.masks_path = os.path.join(root, "masks")

        self.img_list = []
        self.annotations_list = []
        self.masks_list = []
        for img in os.listdir(self.img_path):
            self.img_list.append(os.path.join(self.img_path, img))
        for annot in os.listdir(self.annotations_path):
            self.annotations_list.append(
                os.path.join(self.annotations_path, annot))

    def generate_masks(self):
        print("Generating masks...")
        # Check if directory exists, if not create it
        if not os.path.exists(self.masks_path):
            os.makedirs(self.masks_path)

        if len(os.listdir(self.masks_path)) == 0:
            for annot in os.listdir(self.annotations_path):
                if annot == "desktop.ini":
                    pass
                else:
                    tree = et.parse(os.path.join(self.annotations_path, annot))
                    root = tree.getroot()
                    mask_list = []
                    img_size = (1000, 1000)
                    masked_image = Image.new("LA", img_size, color=(0, 0))
                    for region in root.findall(".//Region"):
                        vertices = region.findall(".//Vertex")
                        mask = [(float(vertex.get("X")), float(vertex.get("Y")))
                                for vertex in vertices]
                        mask_list.append(mask)
                    for mask in mask_list:
                        draw = ImageDraw.Draw(masked_image)
                        draw.polygon(mask, fill=(255, 255))
                    image_array = np.array(masked_image)
                    masked_image.save(os.path.join(
                        self.masks_path, annot[:-4] + ".png"))

                    self.masks_list.append(os.path.join(
                        self.masks_path, annot[:-4] + ".png"))
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
        img_mean, img_std = img.mean(), img.std()
        img = (img - img_mean) / img_std

        # Binarize mask truth labels for black and white pixels
        # mask = np.float32(mask)
        mask[mask <= 0] = 0
        mask[mask > 0] = 255

        img = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            img = self.transform(img)
            # Resize the mask to match the output size of the model
            resize_transform = transforms.Compose(
                [transforms.Resize((256, 256), interpolation=Image.NEAREST),
                 transforms.ToTensor()])
            mask = resize_transform(Image.fromarray(mask))

        return img, mask


def map_files_to_tissue_types(directory_path, tissue_types):
    # Get all file names in the specified director
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(
        os.path.join(directory_path, f))]

    # Create a mapping between file names and tissue types
    file_tissue_mapping = dict(zip(file_names, tissue_types))

    return file_tissue_mapping


def split_data(file_tissue_mapping, seed=42):
    # This function splits data into an estimated 0.7 , 0.15, 0.15
    # Get a list of all file names and tissue types
    file_names = list(file_tissue_mapping.keys())

    # Split into training, validation, and test sets
    train_data, test_data = train_test_split(
        file_names, test_size=5, random_state=seed)

    # Ensure 'Colon' files are in the test set
    colon_files = [file_name for file_name,
                   tissue_type in file_tissue_mapping.items() if tissue_type == 'Colon']
    test_data += colon_files[:3]

    # Remove 'Colon' files from the training and val set
    train_data = [
        file_name for file_name in train_data if file_name not in test_data]

    # Select random files for validation (Further split test data)
    val_data = train_test_split(train_data, test_size=8, random_state=seed)[1]

    # Remove val data from train data
    train_data = [
        file_name for file_name in train_data if file_name not in val_data]

    return train_data, val_data, test_data


def generate_datasets(root, save_dir):
    # Print out all file names in tissue_image
    source_root = os.path.join(root, "MoNuSegAllData")
    tissue_image_dir = os.path.join(source_root, "tissue_image")
    annotation_dir = os.path.join(source_root, "annotations")
    all_tissue_images = os.listdir(tissue_image_dir)

    # Define roots
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")
    test_root = os.path.join(root, "test")
    if not os.path.exists(train_root):
        # Identify unique tissue types
        unique_tissue_types = set(tissue_types)  # imported from utils
        tissue_type_count = {tissue_type: tissue_types.count(
            tissue_type) for tissue_type in unique_tissue_types}
        print(unique_tissue_types)
        print(tissue_type_count)

        file_tissue_mapping = map_files_to_tissue_types(
            tissue_image_dir, tissue_types)

        # Split data into train val and test
        train_data, val_data, test_data = split_data(file_tissue_mapping)

        # save splits
        with open(os.path.join(save_dir, "monuseg_datasets_split.txt"), "w+") as file:
            file.write(",".join([file for file in train_data]) + "\n")
            file.write(",".join([file for file in val_data]) + "\n")
            file.write(",".join([file for file in test_data]))

        # Save the splits into into txt

        # Ensure data is split properly
        print("Train Data", len(train_data))
        print("Val Data", len(val_data))
        print("Test Data", len(test_data))

        # Define new folders to store train and val data
        new_train_img_path = os.path.join(train_root, "tissue_image")
        new_train_annot_path = os.path.join(train_root, "annotations")

        new_val_img_path = os.path.join(val_root, "tissue_image")
        new_val_annot_path = os.path.join(val_root, "annotations")

        new_test_img_path = os.path.join(test_root, "tissue_image")
        new_test_annot_path = os.path.join(test_root, "annotations")

        new_train_mask_path = os.path.join(train_root, "masks")
        new_val_mask_path = os.path.join(val_root, "masks")
        new_test_mask_path = os.path.join(test_root, "masks")

        # Create these folders if does not exists
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
                source_path = os.path.join(tissue_image_dir, file_name)

                # Construct the full path to the destination file
                dest_path = os.path.join(dest_img_path, file_name)
                source_xml_path = os.path.join(
                    annotation_dir, f"{os.path.splitext(file_name)[0]}.xml")

                # Construct the full path to the destination XML file
                dest_xml_path = os.path.join(
                    dest_annot_path, f"{os.path.splitext(file_name)[0]}.xml")

                # Check if file already exists before copying
                if not os.path.exists(dest_path):
                    shutil.copy(source_path, dest_path)

                if not os.path.exists(dest_xml_path):
                    shutil.copy(source_xml_path, dest_xml_path)
