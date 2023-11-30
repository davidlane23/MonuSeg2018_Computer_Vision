import torch
from tqdm import tqdm
import torch.nn as nn
# from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex
from .MonuSeg_evaluator import IOU_Evaluator
from .MonuSeg_evaluator import PixelAccuracyEvaluator
from sklearn.metrics import accuracy_score, jaccard_score
import numpy as np


class MonuSegModel:
    def __init__(self, model, device, n_classes=2, weights=None, criterion=None, lr=None, epochs=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.iou_evaluator = IOU_Evaluator(num_classes=2)
        self.pixAcc_evaluator = PixelAccuracyEvaluator()
        self.config_model(weights)
        if lr is not None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=float(lr))  # TODO: Try other optimizers
        else:
            self.optimizer = None

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []  # mIOU
        self.pixel_accuracies = []

        self.thresholds = [0.1, 0.15, 0.20, 0.25, 0.30,
                           0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7]

    def config_model(self, weights):
        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)

    def train(self, dataloader):
        self.model.train()

        epoch_losses = []

        data_size = 0
        avg_loss = 0
        num_of_batches = 0

        for i, (images, masks) in tqdm(enumerate(dataloader)):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # self.optimizer.zero_grad()
            outputs = self.model(images)
            out_tensor = outputs
            loss = self.criterion(out_tensor, masks)

            # calculate gradient and back propagation
            self.optimizer.zero_grad()  # reset accumulated gradients
            loss.backward()  # compute new gradients
            self.optimizer.step()  # apply new gradients to change model parameters

            avg_loss = (avg_loss * num_of_batches + loss) / \
                       (num_of_batches + 1)
            epoch_losses.append(float(avg_loss))

            # update number of batches
            num_of_batches += 1

        return avg_loss, epoch_losses

    def evaluate(self, dataloader):
        self.model.eval()

        # Do not record gradients
        with torch.no_grad():
            epoch_losses = []
            epoch_iou = []
            epoch_pixel_accuracies = []

            num_of_batches = 0
            data_size = 0
            avg_loss = 0
            mean_iou = 0

            for i, (images, masks) in tqdm(enumerate(dataloader)):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)

                outputs[outputs <= 0.65] = 0
                outputs[outputs > 0.65] = 1

                self.iou_evaluator.update(outputs, masks)

                mean_iou = self.iou_evaluator.get_mean_iou()

                # Assuming a threshold of 0.5 for binary segmentation
                predictions = (outputs > 0.5).float()

                # get the correct and total number of pixels of each sample
                correct_pixels, total_pixels = self.pixAcc_evaluator.total_pixels(
                    predictions=predictions, masks=masks)

                pixel_acc = self.pixAcc_evaluator.get_pixel_accuracy(
                    correct_pixels=correct_pixels, total_pixels=total_pixels)

                loss = self.criterion(outputs, masks)

                avg_loss = (avg_loss * num_of_batches + loss) / \
                           (num_of_batches + 1)

                epoch_losses.append(avg_loss)
                epoch_iou.append(mean_iou)
                epoch_pixel_accuracies.append(pixel_acc)

                data_size += images.size(0)
                # update data size
                num_of_batches += 1

            # avg_loss /= data_size
            mean_iou = self.iou_evaluator.get_mean_iou()
            pixel_acc = self.pixAcc_evaluator.get_pixel_accuracy(
                correct_pixels=correct_pixels, total_pixels=total_pixels)
            print(f'\nPixel Acc: {pixel_acc}')
            print(f'Mean IOU: {mean_iou}')

            return avg_loss, mean_iou, pixel_acc, epoch_losses, epoch_iou, epoch_pixel_accuracies

    def fit(self, train_dataloader, valid_dataloader):
        best_measure = -1
        best_epoch = -1
        best_pixel_acc = -1

        if self.epochs is None or self.criterion is None or self.optimizer is None:
            raise ValueError(
                "Missing parameters \"epochs/criterion/optimizer\"")

        for epoch in range(self.epochs):
            self.iou_evaluator.reset()

            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # train and evaluate model performance
            train_loss, _ = self.train(train_dataloader)
            valid_loss, measure, pixel_acc, epoch_losses, epoch_iou, epoch_pixel_accuracies = self.evaluate(
                valid_dataloader)

            print(f"\nTrain Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")
            print(f'Measure (mIOU): {measure}')
            print(f'Measure (Pixel Accuracy): {pixel_acc}')

            # save metrics
            self.train_losses.append(float(train_loss))
            self.valid_losses.append(float(valid_loss))
            self.valid_accuracies.append(float(measure))
            self.pixel_accuracies.append(float(pixel_acc))

            if measure > best_measure:
                print(f'Updating best measure: {best_measure} -> {measure}')
                best_epoch = epoch
                best_weights = self.model.state_dict()
                best_measure = measure
                best_pixel_acc = pixel_acc

        return best_epoch, best_measure, best_weights, best_pixel_acc

    def predict_batches(self, dataloader):
        self.model.eval()

        # Lists to store predictions and ground truth for all batches
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(dataloader)):
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)

                # Apply a threshold for binary segmentation
                outputs[outputs <= 0.65] = 0
                outputs[outputs > 0.65] = 1

                # # Assuming a threshold of 0.5 for binary segmentation
                # predictions = (outputs > 0.5).float()

                # Append predictions and ground truth to the lists
                all_predictions.append(outputs)
                all_ground_truth.append(masks)

        # Concatenate predictions and ground truth along batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)

        return all_predictions, all_ground_truth
