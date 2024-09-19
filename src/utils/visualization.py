import os

from grain.python import Batch
from tqdm.auto import tqdm
import numpy as np

import matplotlib.pyplot as plt

from typing import List, Tuple, Union, Optional, Any, Dict
from tensorboardX import SummaryWriter

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


def visualize_mask(mask: np.ndarray, class_names: List[str]) -> np.ndarray:
    """
    Visualizes the mask image.

    Parameters:
    - mask: A NumPy array of shape (height, width) in uint8.
    - class_names: List of class names for better visualization.

    Returns:
    - Colored mask as a NumPy array.
    """
    # Define a colormap
    num_classes = len(class_names) 
    cmap = plt.get_cmap('tab20', num_classes)

    # Apply colormap to mask
    mask_color = cmap(mask)[:, :, :3]  # Shape: (H, W, 3)

    # Plotting the mask
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask_color)
    ax.set_title("Mask Image")
    ax.axis("off")

    plt.tight_layout()

    # Convert the matplotlib figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img

def visualize_rgb_nir(image_rgb_nir: np.ndarray) -> np.ndarray:
    """
    Visualizes the RGB and NIR images extracted from a 4-channel RGB-NIR image.

    Parameters:
    - image_rgb_nir: A NumPy array of shape (4, height, width) in uint16.

    Returns:
    - Combined RGB and NIR image as a NumPy array.
    """
    # Extract the BGR channels
    bgr_image = image_rgb_nir[:3, :, :].transpose(1, 2, 0)  # Shape: (H, W, 3)

    # Convert BGR to RGB for visualization purposes
    rgb_image = bgr_image[..., ::-1]  # Convert BGR to RGB by reversing the last channel

    # Extract the NIR channel
    nir_image = image_rgb_nir[3, :, :]  # Shape: (H, W)

    # Plotting the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Plot the NIR image
    axes[1].imshow(nir_image, cmap='gray')
    axes[1].set_title("NIR Image")
    axes[1].axis("off")

    plt.tight_layout()

    # Convert the matplotlib figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img


def log_sample_visualizations(
    writer: SummaryWriter,
    images_rgb_nir: np.ndarray,
    masks: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str] = None,
    step: int = 0,
    num_samples: int = 3
):
    """
    Logs sample visualizations of input images, predicted masks, and ground truth masks to TensorBoard.

    Args:
        writer (SummaryWriter): TensorboardX writer.
        images_rgb_nir (np.ndarray): Input images, shape (N, 4, H, W).
        masks (np.ndarray): Ground truth masks, shape (N, H, W).
        predictions (np.ndarray): Predicted masks, shape (N, H, W).
        class_names (List[str], optional): List of class names for better labeling. Defaults to None.
        step (int, optional): Current epoch or step number. Defaults to 0.
        num_samples (int, optional): Number of samples to visualize. Defaults to 3.
    """
    N = images_rgb_nir.shape[0]
    num_samples = min(num_samples, N)
    
    for i in range(num_samples):
        image_rgb_nir = images_rgb_nir[i]  # Shape: (4, H, W)
        mask = masks[i]  # Shape: (H, W)
        pred = predictions[i]  # Shape: (H, W)
        
        # Generate RGB+NIR visualization
        rgb_nir_img = visualize_rgb_nir(image_rgb_nir)
        
        # Generate Mask and Prediction visualizations
        mask_img = visualize_mask(mask, class_names)
        pred_img = visualize_mask(pred, class_names)

        # Log images to TensorBoard
        writer.add_image(f"Sample_{i}/RGB_NIR", rgb_nir_img, global_step=step, dataformats='HWC')
        writer.add_image(f"Sample_{i}/Ground_Truth_Mask", mask_img, global_step=step, dataformats='HWC')
        writer.add_image(f"Sample_{i}/Prediction_Mask", pred_img, global_step=step, dataformats='HWC')


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues"
) -> np.ndarray:
    """
    Plots the confusion matrix as a heatmap.

    Args:
        cm (np.ndarray): Confusion matrix, shape (C, C).
        class_names (List[str], optional): List of class names. Defaults to None.
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        cmap (str, optional): Colormap for the heatmap. Defaults to "Blues".

    Returns:
        np.ndarray: The plot as a numpy array (H, W, 3).
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
    else:
        cm_display = cm

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")

    fig.tight_layout()

    # Convert the matplotlib figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img
