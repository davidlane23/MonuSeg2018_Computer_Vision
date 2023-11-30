import torch


class IOU_Evaluator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, pred_mask, true_mask):
        for i in range(self.num_classes):
            pred_mask_i = (pred_mask == i)
            true_mask_i = (true_mask == i)
            # calculates the intersection and the union of the predicted and true masks
            intersection = torch.logical_and(
                pred_mask_i, true_mask_i).sum().item()
            union = torch.logical_or(pred_mask_i, true_mask_i).sum().item()
            self.intersection[i] += intersection
            self.union += union

    def compute_iou(self):
        class_iou = torch.zeros(self.num_classes)
        # IOU = Jaccard Dist =  A INTERSECT B / A UNION B
        for i in range(self.num_classes):
            # apply epsilon to ensure no division over 0
            class_iou[i] = self.intersection[i] / self.union[i] + 2e-12
        return class_iou

    def get_mean_iou(self):
        class_iou = self.compute_iou()
        # total iou / num class
        mean_iou = class_iou.sum().item() / self.num_classes
        return mean_iou


class PixelAccuracyEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def total_pixels(self, predictions, masks):
        correct_pixels = 0
        total_pixels = 0

        valid_masks = (masks == 1)
        correct_pixels += (predictions == valid_masks).sum().item()
        total_pixels += masks.numel()

        return correct_pixels, total_pixels  # Return the computed values

    def get_pixel_accuracy(self, correct_pixels, total_pixels):
        pixel_acc = correct_pixels / total_pixels
        return pixel_acc
