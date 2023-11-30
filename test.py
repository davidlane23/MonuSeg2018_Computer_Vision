import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Load the npy file
loaded_data = np.load(os.path.join(
    "MonuSeg_results", "monuseg_valid_predictions.npy"), allow_pickle=True)
loaded_df = pd.DataFrame(loaded_data, columns=[
                         'ImageFile', 'Prediction', 'GroundTruth'])

# Assuming you want to visualize the first row of the DataFrame
row_index = 0
img_file = loaded_df.loc[row_index, 'ImageFile']
prediction = loaded_df.loc[row_index, 'Prediction']
ground_truth = loaded_df.loc[row_index, 'GroundTruth']

# Assuming you have a function to convert tensors to images
# Replace this with your actual function


def numpy_array_to_image(array):
    # Assuming array is a 2D or 3D NumPy array
    if len(array.shape) == 2:
        # For grayscale images
        img_array = array
    elif len(array.shape) == 3:
        # For single-channel images (e.g., (H, W, C))
        img_array = array.squeeze()
    else:
        raise ValueError("Unsupported array shape")

    # Rescale values to the range [0, 255]
    img_array = (img_array * 255).astype(np.uint8)

    # Create a PIL Image from the array
    img = Image.fromarray(img_array)

    return img


print(type(prediction))


# Convert tensors to images
prediction_img = numpy_array_to_image(prediction)
ground_truth_img = numpy_array_to_image(ground_truth)

# Plot the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Assuming img_file is a path to the image
axes[0].imshow(plt.imread(img_file))
axes[0].set_title('Original Image')

axes[1].imshow(prediction_img, cmap='gray')
axes[1].set_title('Prediction')

axes[2].imshow(ground_truth_img, cmap='gray')
axes[2].set_title('Ground Truth')

plt.show()
