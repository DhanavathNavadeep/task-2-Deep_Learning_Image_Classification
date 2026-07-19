# Task 2 – Deep Learning Image Classification

## Overview
This project is developed as part of the CodeAlpha Internship. It implements an image classification model using Deep Learning with TensorFlow/Keras. The model is trained on an image dataset to classify images into different categories with high accuracy.

## Features
- Image classification using Convolutional Neural Networks (CNN)
- Data preprocessing and augmentation
- Model training and validation
- Accuracy and Loss visualization
- Save and load trained model
- Image prediction (Inference)

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## Project Structure

```
task-2-Deep_Learning_Image_Classification/
│── dataset/
│── train.py
│── inference.py
│── model.keras
│── training_curves.png
│── report.pdf
│── requirements.txt
│── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/DhanavathNavadeep/task-2-Deep_Learning_Image_Classification.git
cd task-2-Deep_Learning_Image_Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python train.py
```

The trained model will be saved as:

```
model.keras
```

## Inference

Run:

```bash
python inference.py
```

## Results

- Training Accuracy: ~98%
- Validation Accuracy: ~97–99%
- Training and Validation Loss successfully minimized.

The training curves are available in:

```
training_curves.png
```

## Output

- Trained Model (`model.keras`)
- Training Curves (`training_curves.png`)
- Project Report (`report.pdf`)

## Author

**Dhanavath Navadeep**

B.Tech – Electronics and Communication Engineering (ECE)

Anurag Engineering College

## Internship

**CodeAlpha Internship**

Task 2 – Deep Learning Image Classification
