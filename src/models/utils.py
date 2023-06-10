import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch import Tensor


def plot_predictions(
    images: Tensor,
    masks_true: Tensor,
    y_pred: Tensor,
    n_images: int = 4,
    title="predictions_plot",
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

    fig, ax = plt.subplots(3, n_images)
    for i in range(n_images):
        ax[0, i].imshow(images[i].permute((1, 2, 0)))
        ax[0, i].set_axis_off()

        ax[1, i].imshow(masks_true[i], cmap="gray")
        ax[1, i].set_axis_off()

        ax[2, i].imshow(y_hat[i], cmap="gray")
        ax[2, i].set_axis_off()

    plt.savefig(f"{title}.png")
    plt.show()
