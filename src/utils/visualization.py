import os

from grain.python import Batch
from tqdm.auto import tqdm
import numpy as np

import matplotlib.pyplot as plt

def print_directory_structure(root_dir, prefix="", exclude=None):
    if exclude is None:
        exclude = []

    items = [item for item in os.listdir(root_dir) if item not in exclude]
    item_count = len(items)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        connector = "├── " if index < item_count - 1 else "└── "
        if os.path.isdir(item_path):
            print(f"{prefix}{connector}{item}/")
            new_prefix = f"{prefix}│   " if index < item_count - 1 else f"{prefix}    "
            print_directory_structure(item_path, new_prefix, exclude)
        else:
            print(f"{prefix}{connector}{item}")

# print_directory_structure('./', exclude=['logs', '1.0.0', '.git', '__init__.py','__pycache__', 'test.py'])


def get_num_batches(iterator):
    dataloader = iterator._data_loader
    num_records = dataloader._sampler._num_records
    batch_op = next(op for op in dataloader._operations if isinstance(op, Batch))
    batch_size = batch_op.batch_size
    num_workers = dataloader._global_num_workers
    num_batches = (num_records // batch_size)
    # The progress bar gets messed up if num_workers > 0
    # num_batches = ( raw_num_batches // num_workers ) * num_workers + batch_size # Added the one batch because of the way I handle the end of epoch check
    return num_batches


def create_progress_bar(iterator, desc):
    num_batches = get_num_batches(iterator)
    return tqdm(total=num_batches, desc=desc, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

############################################################################################################################################################################

# Save exemple through the training

def visualize_rgb_nir(image_rgb_nir, save_name):
    """
    Visualizes the RGB and NIR images extracted from a 4-channel RGB-NIR image.

    Parameters:
    - image_rgb_nir: A NumPy array of shape (4, height, width) in uint16.
    """

    # Extract the BGR channels
    bgr_image = image_rgb_nir[:3, :, :].transpose(1, 2, 0)

    # Extract the NIR channel
    nir_image = image_rgb_nir[3, :, :]

    # Plotting the images
    plt.figure(figsize=(10, 5))

    # Plot the RGB image
    plt.subplot(1, 2, 1)
    plt.imshow(bgr_image)
    plt.title("RGB Image")
    plt.axis("off")

    # Plot the NIR image
    plt.subplot(1, 2, 2)
    plt.imshow(nir_image, cmap='gray')
    plt.title("NIR Image")
    plt.axis("off")

    plt.savefig(f'./outputs/visualization/{save_name}_output_image.png')  # Replace with your preferred path and filename

    # Optionally close the plot to free up resources (especially useful in scripts)
    plt.close()


def visualize_hist(image_rgb_nir, save_name):
    """
    Visualizes the histogram of channels in a 4-channel RGB-NIR image.

    Parameters:
    - image_rgb_nir: A NumPy array of shape (4, height, width) in uint16.
    """

    # Extract the BGR channels
    bgr_image = image_rgb_nir[:3, :, :].transpose(1, 2, 0)

    # Extract the NIR channel
    nir_image = image_rgb_nir[3, :, :]

    # Plotting the histograms
    plt.figure(figsize=(10, 5))

    # Plot the RGB histograms
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.hist(bgr_image[:, :, i].ravel(), bins=256, range=(0, 1))
        plt.title(f"RGB Channel {i}")
        plt.xlim([0, 1])

    # Plot the NIR histogram
    plt.subplot(2, 3, 4)
    plt.hist(nir_image.ravel(), bins=256, range=(0, 1))
    plt.title("NIR Channel")
    plt.xlim([0, 1])

    plt.savefig(f'./outputs/visualization/{save_name}_output_hist.png')  # Replace with your preferred path and filename

    # Optionally close the plot to free up resources (especially useful in scripts)
    plt.close()

def visualize_mask(mask, save_name):
    """
    Visualizes the mask image.

    Parameters:
    - mask: A NumPy array of shape (height, width) in uint8.
    """

    # Plotting the mask
    plt.figure(figsize=(5, 5))
    plt.imshow(mask, cmap='tab20', vmin=0, vmax=19)
    plt.title("Mask Image")
    plt.axis("off")

    plt.savefig(f'./outputs/visualization/{save_name}_output_mask.png')  # Replace with your preferred path and filename

    # Optionally close the plot to free up resources (especially useful in scripts)
    plt.close()