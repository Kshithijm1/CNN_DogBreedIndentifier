# CNN Image Classifier

## Overview

This Python project features a Convolutional Neural Network (CNN) designed to classify breeds of dogs into predefined categories based on the Oxford-IIIT Pet Dataset. The CNN is implemented using NumPy for numerical operations and the Python Imaging Library (PIL) for image handling. The project includes comprehensive methods for the forward pass, backward propagation, and training procedures using mini-batch gradient descent.

### Features

- Implementation of convolutional layers, ReLU activations, and max pooling.
- Fully connected layers for classification.
- Softmax loss calculation for output.
- Gradient descent optimization for learning the weights.
- Utility functions for loading and preprocessing images and labels.

## Dataset

The model trains on the [Oxford-IIIT Pet Dataset](https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1), which contains a diverse set of pet images labeled with their respective species. The dataset needs to be downloaded and prepared separately.

## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.8 or later
- NumPy (for matrix operations)
- Pillow (PIL, for image processing)

## Installation

### Step 1: Clone the Repository

To get started with the CNN Image Classifier, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Kshithijm1/CNN_DogBreedIndentifier.git
cd cnn-image-classifier
```

### Step 2: Install Dependencies

Install the necessary Python libraries using pip:

```bash
pip install numpy pillow
```

### Step 3: Download the Dataset

Download the Oxford-IIIT Pet Dataset from [here](https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1). Ensure you extract and organize the images and annotations in a directory accessible to the main script.

## Configuration

Modify the `main` function in `cnn_classifier.py` to point to the correct paths for your dataset:

- `images_path`: Directory containing the pet images.
- `annotations_path`: Path to the file listing image names and their corresponding labels.

## Usage

To run the classifier:

1. Navigate to the project directory.
2. Execute the script with Python:

```bash
python cnn_classifier.py
```

This command will initiate the training process and print out the loss during training. After training, it will predict the class for a specified test image.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Oxford-IIIT Pet Dataset for providing the images used in training and testing the classifier.
- Academic Torrents for hosting and distributing the dataset.
