# Model and evaluation

The baseline is ImageNet-initialized ResNet-18 trained with cross-entropy using the split provided to `ml/training/train.py`. The model saves only the best validation weighted F1 checkpoint and writes the actual validation metrics beside it. Checkpoints are ignored by Git. The inference service refuses inspection requests without this checkpoint rather than falling back to synthetic predictions.

Grad-CAM is calculated with gradients from the final ResNet block for the predicted class. Its normalized map is returned as an image and thresholded to create a coarse evidence bounding box. Evaluate classification on a held-out test split using accuracy, weighted precision/recall/F1 and a confusion matrix; compare the box with labeled masks using IoU if masks are available. Do not treat Grad-CAM as a replacement for a trained segmentation model.
