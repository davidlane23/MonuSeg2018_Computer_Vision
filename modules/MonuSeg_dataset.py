import os
import csv

import torch
import rasterio
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import xml.etree.ElementTree as et
import cv2
import torchvision.transforms as transforms


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
			self.annotations_list.append(os.path.join(self.annotations_path, annot))

	def generate_masks(self):
		print("Generating masks...")
		if len(os.listdir(self.masks_path)) == 0:
			for annot in os.listdir(self.annotations_path):
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
					masked_image.save(os.path.join(self.masks_path, annot[:-4] + ".png"))

					self.masks_list.append(os.path.join(self.masks_path, annot[:-4] + ".png"))
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

		# Binarize mask truth labels for black and white pixels
		# mask = np.float32(mask)
		mask[mask <= 0] = 0
		mask[mask > 0] = 255

		img = Image.fromarray(img.astype('uint8'))
		if self.transform is not None:
			img = self.transform(img)
			# Resize the mask to match the output size of the model
			resize_transform = transforms.Compose(
				[transforms.Resize((128, 128), interpolation=Image.NEAREST), transforms.ToTensor()])
			mask = resize_transform(Image.fromarray(mask))

		return img, mask

