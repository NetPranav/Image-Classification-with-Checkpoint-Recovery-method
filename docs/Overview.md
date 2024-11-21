## Image Classification(FashionMNIST) with Checkpoint Recovery Method

## Project Overview 
This project implements a Convolutional Neural Network (CNN) to classify images from the FashionMNIST dataset. The dataset contains 70,000 grayscale images of 10 different clothing categories (e.g., T-shirts, trousers, shoes) with dimensions of 28x28 pixels.
The training proccess is the Checkpoint Recovery Method which can save and load the model's `state_dict` to continue the training with last last updated training
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
FashionMNIST is a simple dataset with high intra-class variance and low inter-class variability. The goal is to:

1. Build a CNN model from scratch to classify images accurately.
2. Experiment with data augmentations (e.g., trivial augmentations) to improve model performance.
3. Demonstrate model checkpointing to make training scalable and reproducible.
4. Turning the model's training conditions in between of the training

## Project Highlights
1. Progressive Checkpointing Approach:
  * Saves the model state after every few epochs to prevent loss of progress during interruptions.
2. Data Augmentations:
  * Applied transformations like trivial augmentations to enhance generalization.
3. Training Pipeline:
  - The pipeline is modular and organized into separate files:
    * `model.py`: Defines the CNN architecture.
    * `train_and_test.py`: Handles training and validation loops.
    * `Progressive_checkpoint_training.py`: Contains utility functions (e.g., checkpoint saving and loading).
4. Evaluation:
  * Includes accuracy and loss curves, along with a few sample predictions.

## Future Work
Experimentation:
* Test different CNN architectures like ResNet and MobileNet.
* Incorporate additional augmentations such as random rotation, flipping, and cropping.
Dataset Expansion:
* Train the model on datasets beyond FashionMNIST, such as CIFAR-10 or ImageNet (subset).
Visualization:
* Add Grad-CAM to visualize which parts of the image the model focuses on during predictions.

##  How to Run the Project
1. Clone the repository:
 *  `https://github.com/NetPranav/Image-Classification-with-Checkpoint-Recovery-method`
 *  `cd Image-Classification-with-Checkpoint-Recovery-method`
2. Install the dependencies
 * `pip install -r requirements.txt`
3. Open the notebook
 * Open the `jupyter notebook` in browser and open the `FashionMNIST_Classification.ipynb`.
 * Follow the cell Sequentially to run the training and testing properly without errors

 ## Note:
- the `traing_and_testing.py`,`model.py`,`Progressive_checkpoint_training.py` are the python file to understand about the code, They won't work individually only the main file: `FashionMNIST_Classification.ipynb` will work in certain notebook of prefrence. 
    
