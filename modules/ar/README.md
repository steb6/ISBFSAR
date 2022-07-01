# Skeleton-Based Few-Shot Open-Set Action-Recognition
This directory contains the implementation of the paper "Few-Shot Open-Set Skeleton-Based Action-Recognition".

## Getting ready
First, download the dataset from https://bit.ly/3OcD9yb, then set the variables inside the class TRXConfig in 'utils/params.py'.

## Training
Launch 'utils/train.py' to train with the split of NTURGBD120.

## Evaluation
To evaluate the model, first you need the exemplars.
You can download them from https://bit.ly/3a5bW1w, then set the path as done for the dataset.
Just launch 'utils/test_fs_os.py' to get the comparison with the baseline as done in the paper.
You can also build the confusion matrix of the discriminator with 'utils/create_confusion_matrix.py'.

## Pretrained model
If one doesn't want to do the full training, we provide checkpoints for both our model and the baseline here: https://bit.ly/3NBsPhZ.