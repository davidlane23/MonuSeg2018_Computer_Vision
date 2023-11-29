import torch


class PixelAccuracyEvaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def calculate_pixel_accuracy(self):
        self.model.eval()
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for inputs, masks in self.dataloader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(inputs)
                predictions = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary segmentation

                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()

        pixel_accuracy = correct_pixels / total_pixels
        return pixel_accuracy
