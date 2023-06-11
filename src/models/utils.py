import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F

from torch import Tensor


def plot_predictions(
    images: Tensor,
    masks_true: Tensor,
    y_pred: Tensor,
    n_images: int = 4,
    title="predictions_plot",
    segm_threshold=0.5,
):
    """
    Images: Images tensor of batch size
    Masks tensor of batch size equal to images
    y_pred_sigmoid: predictions from a nn. Sigmoided inside the function
    n_images: how many images to plot
    """

    y_hat = F.sigmoid(y_pred)
    images = images.cpu()
    masks_true = masks_true.cpu().squeeze()
    y_hat = y_hat.cpu().squeeze()
    y_hat_ = torch.where(y_hat > segm_threshold, 1, 0)

    # Define the number of rows and columns
    num_rows = 3
    num_cols = n_images

    # Create a grid of subplots using GridSpec
    fig = plt.figure()
    grid = GridSpec(num_rows, num_cols + 1, figure=fig)

    # Create axes for each subplot
    axes = []
    titles = ["Image", "Mask", "Prediction\n@0.5 threshold"]
    for i in range(num_rows):
        # Add title on the left side of the row
        ax_title = fig.add_subplot(grid[i, 0])
        ax_title.set_axis_off()
        ax_title.text(0, 0.5, f"{titles[i]}", va="center")

        # Add the main subplot for each column
        row_axes = []
        data = images.permute((0, 2, 3, 1))
        cmap = None
        if i == 1:
            data = masks_true
            cmap = "gray"
        if i == 2:
            data = y_hat_
            cmap = "gray"

        for j in range(num_cols):
            ax = fig.add_subplot(grid[i, j + 1])
            ax.imshow(data[j], cmap=cmap)
            ax.set_axis_off()
            row_axes.append(ax)
        axes.append(row_axes)

    # Adjust the layout and spacing of subplots
    fig.tight_layout()

    plt.savefig(f"{title}.png")
    plt.show()


def prediction_accuracy(y_real, y_pred, segm_threshold=0.5):
    y_pred = torch.where(y_pred > segm_threshold, 1, 0)

    return (y_pred == y_real).sum().cpu().item()


def calculate_sensitivity_specificity():
    pass