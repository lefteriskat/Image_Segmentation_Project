import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

def plot_predictions(
    images: Tensor,
    masks_true: Tensor,
    y_pred: Tensor,
    n_images: int = 4,
    title:str="predictions_plot",
    segm_threshold:float=0.5,
)->None:
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










class Metrics:
    '''
    Class that stores functions that return different segmentation metrics'''

    def prediction_accuracy(y_real:Tensor, y_pred:Tensor, segm_threshold:float=0.5)->int:
        y_pred = torch.where(y_pred > segm_threshold, 1, 0)

        return (y_pred == y_real).sum().cpu().item()



    def get_IoU(tp:int, fp:int, fn:int)->float:
        return tp/(tp+fp+fn)
        

    def get_dice_coe(y_true:Tensor, y_pred_sigm:Tensor, segm_threshold=0.5, epsilon=1e-3)->float:
        y_pred_mask = torch.where(y_pred_sigm>segm_threshold, 1, 0)
        num = (2*y_pred_mask*y_true).sum()
        den = (y_pred_mask+y_true).sum()
        if den==0:
            den += epsilon
        return num/den
    

    def get_tp_tn_fp_fn(y_true:Tensor, y_pred_sigm:Tensor, segm_threshold:float=0.5)->Tuple[int]:
        y_true = y_true.cpu()
        y_true = y_true.type(torch.int64)

        y_pred = y_pred_sigm.cpu()
        y_pred = torch.where(y_pred>segm_threshold, 1, 0)
        y_pred = y_pred.type(torch.int64)

        true_positives = ((y_pred==1)&(y_true==1)).sum().item()
        true_negatives = ((y_pred==0)&(y_true==0)).sum().item()
        false_positives = ((y_pred==1)&(y_true==0)).sum().item()
        false_negatives = ((y_pred==0)&(y_true==1)).sum().item()

        return true_positives, true_negatives, false_positives, false_negatives

    
    def get_sensitivity_specificity(tp:int, tn:int, fp:int, fn:int)->Tuple[float]:
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)

        return sensitivity, specificity




class Losses:
    '''
    Class that holds loss functions that can be used to train segmentation models
    '''
    
    def focal_loss(gamma:float=0.5):
        def focal_loss__(y_pred:Tensor, y_real:Tensor):
            y_real_flat = y_real.view(y_real.size(0), -1)
            y_pred_flat = y_pred.view(y_pred.size(0), -1)        
     
            weight = (1-F.sigmoid(y_pred_flat)).pow(gamma)
            tmp = weight*y_real_flat*torch.log(F.sigmoid(y_pred_flat)) + (1-y_real_flat)*torch.log(1-F.sigmoid(y_pred_flat))
            return -torch.mean(tmp)
        
        return focal_loss__


    # def focal_loss(y_real:Tensor, y_pred:Tensor, gamma:float=0.5)->Tensor:
    #     ### y_real and y_pred is [batch_n, channels=1, h, w]: [6, 1, 128, 128]
    #     y_real_flat = y_real.view(y_real.size(0), -1)
    #     y_pred_flat = y_pred.view(y_pred.size(0), -1)
        
     
    #     weight = (1-F.sigmoid(y_pred_flat)).pow(gamma)
    #     tmp = weight*y_real_flat*torch.log(F.sigmoid(y_pred_flat)) + (1-y_real_flat)*torch.log(1-F.sigmoid(y_pred_flat))
    #     return -torch.mean(tmp)
    
    def bce_loss(y_pred:Tensor, y_real:Tensor)->Tensor:
        y_pred = torch.clip(y_pred, -10, 10)
        return torch.mean(y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred)))
    
    def dice_loss(y_pred:Tensor, y_real:Tensor)->Tensor:
        ### y_real and y_pred is [batch_n, channels=1, h, w]: [6, 1, 128, 128]
        y_real_flat = y_real.view(y_real.size(0), -1)
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        num = (2 * y_real_flat * F.sigmoid(y_pred_flat) + 1).mean()
        den = (y_real_flat + F.sigmoid(y_pred_flat)).mean() + 1
        return 1 - (num / den)
    
    def bce_total_variation(y_pred:Tensor, y_real:Tensor, lambda_:float=0.1)->Tensor:

        def total_variation_term():        
            y_pred_x = y_pred[:,:,:-1,:]
            y_pred_xp1 = y_pred[:,:,1:,:]
            
            y_pred_y = y_pred[:,:,:,:-1]
            y_pred_yp1 = y_pred[:,:,:,1:]
            
            y_pred_x_flat = torch.flatten(y_pred_x,start_dim=1)
            y_pred_xp1_flat = torch.flatten(y_pred_xp1,start_dim=1)
            
            y_pred_y_flat = torch.flatten(y_pred_y,start_dim=1)
            y_pred_yp1_flat = torch.flatten(y_pred_yp1,start_dim=1)
            
            term1 = torch.sum( torch.abs(F.sigmoid(y_pred_xp1_flat) - F.sigmoid(y_pred_x_flat)) )
            term2 = torch.sum( torch.abs(F.sigmoid(y_pred_yp1_flat) - F.sigmoid(y_pred_y_flat)) )
            return term1 + term2

        return Losses.bce_loss(y_real, y_pred) + lambda_*total_variation_term()