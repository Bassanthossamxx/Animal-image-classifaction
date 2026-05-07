# Animal Image Classification

End-to-end deep learning pipeline for visual recognition of five animal classes using PyTorch and transfer learning. The project follows the university-required three-tier workflow: training, validation, and final testing on unseen data.

## Project Overview

This repository contains:

- Local dataset extraction under `data/animals_dataset/`
- Three Jupyter notebooks for exploration, training, and evaluation
- A Gradio web app for image upload and prediction
- Saved plots and reports in `results/`

The bundled archive was normalized to the project convention:

- `validation` renamed to `val`
- `deep` renamed to `deer`

Dataset statistics are verified in `notebooks/1_data_exploration.ipynb` instead of being hardcoded in documentation.

## Project Structure

```text
Animal image classifaction/
├── app/
│   └── app.py
├── data/
│   ├── archive.zip
│   └── animals_dataset/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_model_training.ipynb
│   ├── 3_evaluation.ipynb
│   └── CV dataprocessing&cleaning.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Confirm dataset location

The project expects the dataset here:

```text
data/animals_dataset/
├── train/
├── val/
└── test/
```

The archive from `data/archive.zip` has already been extracted and normalized to this structure.

## Notebook Workflow

Run the notebooks in this order:

1. `notebooks/1_data_exploration.ipynb`
2. `notebooks/2_model_training.ipynb`
3. `notebooks/3_evaluation.ipynb`

### Notebook 1

- Verifies dataset folders and class names
- Counts images per class for each split
- Visualizes class distribution
- Displays sample images
- Checks actual image sizes and formats

### Notebook 2

- Loads train and validation splits with `ImageFolder`
- Applies data augmentation to the training split
- Trains a ResNet18 transfer-learning model
- Uses validation loss, LR scheduling, and early stopping
- Saves the best model to `models/best_model.pth`
- Saves training curves to `results/training_curves.png`

### Notebook 3

- Loads the saved model and test split
- Computes confusion matrix, accuracy, precision, recall, and F1-score
- Saves the classification report and visualizations into `results/`

## Web App

Launch the Gradio app from the project root:

```bash
python app/app.py
```

Open the local URL shown in the terminal, usually `http://127.0.0.1:7860`.

The app:

- Loads `models/best_model.pth`
- Accepts an uploaded image
- Returns the predicted class and class probabilities

## Evaluation Outputs

After training and evaluation, the project should produce:

- `results/class_distribution.png`
- `results/sample_images.png`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/sample_predictions.png`
- `results/classification_report.txt`

## University Requirements Mapping

### Training Phase

- Deep neural network training with transfer learning
- Feature learning from the labeled training split

### Validation Phase

- Separate validation split
- Early stopping to reduce overfitting
- Learning-rate scheduling and monitoring by epoch

### Testing Phase

- Final benchmark on unseen test images
- Performance report with required metrics

### Metrics Included

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score

## Notes

- The notebooks and app use the normalized class order: `cat`, `cow`, `deer`, `dog`, `lion`.
- Image resizing to `224x224` happens in preprocessing for model training and evaluation.
- The original archive contains mixed image sizes and uneven class counts, so actual dataset details are reported by Notebook 1.
