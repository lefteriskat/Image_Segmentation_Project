import numpy as np
import torch
from torchmetrics import Dice
from torchmetrics.classification import BinarySpecificity, BinaryJaccardIndex, BinaryRecall
import torchmetrics
def compute_iou(predicted_mask, ground_truth):
    iou = BinaryJaccardIndex(num_classes=2)
    iou_value = iou(predicted_mask, ground_truth)
    return iou_value

def compute_dice(predicted_mask, ground_truth):
    dice = Dice()
    dice_value = dice(predicted_mask, ground_truth)
    return dice_value

def compute_specificity(predicted_mask, ground_truth):
    metric = BinarySpecificity()
    specificity = metric(predicted_mask, ground_truth)
    return specificity

def compute_sensitivity(predicted_mask, ground_truth):
    sensitivity = torchmetrics.functional.classification.binary_recall(predicted_mask, ground_truth)
    return sensitivity