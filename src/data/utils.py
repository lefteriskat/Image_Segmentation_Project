import numpy as np
import torch
from torchmetrics import Dice
from torchmetrics.classification import BinarySpecificity, BinaryJaccardIndex, BinaryRecall

# Generate ground truth segmentation mask (binary tensor)
ground_truth = torch.tensor([[0, 0, 1, 1],
                             [0, 1, 1, 1],
                             [0, 0, 1, 1]], dtype=torch.bool)

# Generate predicted segmentation mask (binary tensor)
predicted_mask = torch.tensor([[0, 0, 1, 1],
                               [0, 1, 1, 0],
                               [0, 0, 0, 0]], dtype=torch.bool)


### IoU

# Compute evaluation metrics
iou = BinaryJaccardIndex(num_classes=2)
IoU= iou(predicted_mask, ground_truth)

print("IoU:", IoU)



### DICE

met = Dice() # average='micro'
dice = met(predicted_mask, ground_truth)

# Print the Dice Overlap metric
print("Dice:", dice)



### Specificity
# 1os tropos
metric = BinarySpecificity()
specificity = metric(predicted_mask, ground_truth)

# Print the specificity metric
print("Specificity:", specificity)



### Sensitivity
metric = BinaryRecall()
sensitivity1 = metric(predicted_mask, ground_truth)

# Print the sensitivity metric
print("Sensitivity:", sensitivity1)

