## Image Classification(FashionMNIST) with Checkpoint Recovery Method

## Project Overview 
-This project implements a Convolutional Neural Network (CNN) to classify images from the FashionMNIST dataset. The dataset contains 70,000 grayscale images of 10 different clothing categories (e.g., T-shirts, trousers, shoes) with dimensions of 28x28 pixels.
-The training proccess is the Checkpoint Recovery Method which can save and load the model's `state_dict` to continue the training with last last updated training
Reason for this method:-
* The model save_dict can be saved and loaded as the training conitnues
* Ensuring the reuseiblity
* Effectivness to tweaking the model in between the training
* Train on more epochs in batches
* Freedom of intruptions at will

## Key Feature of the model :-
1. Dataset: FashionMNIST, a popular benchmark dataset for image classification tasks.
2. Model: Custom CNN built with PyTorch.
   * Two convolutional blocks with Conv2D, ReLU, and MaxPooling.
   * A fully connected layer for classification.
3. Training:
   * Uses progressive checkpoints to save model weights during training.
   * Trained with different learning rates and augmentations.
4. Performance:
   * Achieved 90% accuracy on the test set after training for 100 epochs.
  
## Motivation 
- FashionMNIST is a simple dataset with high intra-class variance and low inter-class variability. The goal is to:

1. Build a CNN model from scratch to classify images accurately.
2. Experiment with data augmentations (e.g., trivial augmentations) to improve model performance.
3. Demonstrate model checkpointing to make training scalable and reproducible.
4. Turning the model's training conditions in between of the training

## Project Highlights
1. Progressive Checkpointing Approach:
* Saves the model state after every few epochs to prevent loss of progress during interruptions.
Data Augmentations:
Applied transformations like trivial augmentations to enhance generalization.
Training Pipeline:
The pipeline is modular and organized into separate files:
model.py: Defines the CNN architecture.
train.py: Handles training and validation loops.
utils.py: Contains utility functions (e.g., checkpoint saving and loading).
Evaluation:
Includes accuracy and loss curves, along with a few sample predictions.
