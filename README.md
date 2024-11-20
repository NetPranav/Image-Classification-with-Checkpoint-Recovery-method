# Progressive Checkpoint Training for Image Classification(FashionMNIST)

## Project Overview
This project implements a **FashionMNIST image classification model** using PyTorch. The model uses the **progressive checkpointing approach**, which saves and resumes training dynamically to handle interruptions and ensure seamless progress.

## Features
- Incremental training with model checkpoints.
- Efficient handling of training interruptions.
- Accuracy tracking during training and testing phases.
- Visualization of results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Training and testing the model:
   ```bash
   python train_and_test.py
   ```
2. Load and test a saved checkpoint:
   ```bash
   python Progressive_checkpoint_training.py
   ```

## Results
- Achieved **90% accuracy** on the FashionMNIST test dataset.

## Note
- Results of the model can easily be increased by tweaking and playing with it but the project's approach was to keep it as simple as possible and showcase the technique of the and advantages of Progressive checkpoint training.

## Files in the Repository
- **train.py**: Contains the training loop with progressive checkpoint saving.
- **test.py**: Code to evaluate the saved model on test data.
- **model.py**: The CNN model definition.
- **README.md**: This documentation.
- **requirements.txt**: Dependencies for the project.

## Acknowledgments
This project was created for learning purposes and demonstrates best practices in deep learning model training with PyTorch.

<!---
NetPranav/NetPranav is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
