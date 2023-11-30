class PixelAccuracyEvaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def total_pixels(self, predictions, masks, predicted_class=1):
        correct_pixels = 0
        total_pixels = 0

        valid_masks = (masks == predicted_class)
        correct_pixels += (predictions == valid_masks).sum().item()
        total_pixels += masks.numel()

        return correct_pixels, total_pixels  # Return the computed values

    def get_pixel_accuracy(self, correct_pixels, total_pixels):
        pixel_acc = correct_pixels / total_pixels
        return pixel_acc