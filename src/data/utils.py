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

def calculate_class_weights(dataset):
    class_counts = [0,0]

    # Step 1: Compute class frequencies
    for _, label_batch in dataset:
        values, counts = np.unique(label_batch, return_counts=True)
        print(values)
        print(counts)
        class_counts = counts

    # Step 2: Calculate class weights
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]

    # Optional: Normalize the weights
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()

    return class_weights