# Run Summary

This file summarizes the outputs generated after running the three project notebooks:

1. `notebooks/1_data_exploration.ipynb`
2. `notebooks/2_model_training.ipynb`
3. `notebooks/3_evaluation.ipynb`

## Notebook Coverage

### 1. Data Exploration

The first notebook completed the dataset verification stage and confirmed that the local project uses the normalized dataset structure:

- `data/animals_dataset/train`
- `data/animals_dataset/val`
- `data/animals_dataset/test`

It also confirmed the five working classes used throughout the project:

- `cat`
- `cow`
- `deer`
- `dog`
- `lion`

Generated outputs:

- `class_distribution.png`
- `sample_images.png`

Observed split counts from the local dataset:

| Split | Cat | Cow | Deer | Dog | Lion | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 91 | 86 | 87 | 111 | 89 | 464 |
| Val | 17 | 15 | 17 | 17 | 17 | 83 |
| Test | 17 | 16 | 16 | 16 | 17 | 82 |

Note:

- The local archive does not exactly match the original prompt assumptions of 500 images and fixed original size `224x224`.
- Images are resized during preprocessing for training and evaluation.

### 2. Model Training

The second notebook completed the training and validation stage using transfer learning with ResNet18.

Training setup highlights:

- Framework: PyTorch
- Model: ResNet18
- Strategy: transfer learning
- Validation monitoring: enabled
- Learning-rate scheduler: enabled
- Early stopping: enabled

Generated outputs:

- `models/best_model.pth`
- `training_curves.png`

This stage produced the trained model checkpoint used later by the evaluation notebook and the Gradio application.

### 3. Evaluation

The third notebook completed final testing on the unseen test split and generated the required performance report.

Generated outputs:

- `classification_report.txt`
- `confusion_matrix.png`
- `sample_predictions.png`

Final test metrics:

- Accuracy: `97.56%`
- Macro precision: `0.9778`
- Macro recall: `0.9765`
- Macro F1-score: `0.9757`

Per-class performance from the saved classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 1.0000 | 0.8824 | 0.9375 | 17 |
| Cow | 1.0000 | 1.0000 | 1.0000 | 16 |
| Deer | 0.8889 | 1.0000 | 0.9412 | 16 |
| Dog | 1.0000 | 1.0000 | 1.0000 | 16 |
| Lion | 1.0000 | 1.0000 | 1.0000 | 17 |

## Overall Assessment

The project run can be considered successful.

Strengths:

- All three notebooks completed their intended roles.
- The project produced every core academic deliverable: exploration outputs, trained model, confusion matrix, and classification report.
- Test accuracy is very high for this dataset.
- Most classes achieved perfect precision and recall on the test split.

Limitations:

- The test split is small at 82 images, so a small number of mistakes can noticeably change the final percentage.
- `cat` recall is lower than the other classes, which suggests a few cat images were confused with another class.
- `deer` precision is slightly lower than perfect, which suggests at least one image from another class was predicted as deer.

## Files Produced During the Run

- `class_distribution.png`
- `sample_images.png`
- `training_curves.png`
- `classification_report.txt`
- `confusion_matrix.png`
- `sample_predictions.png`
- `../models/best_model.pth`

## Conclusion

The notebook workflow successfully delivered a complete end-to-end deep learning pipeline:

- dataset verification
- training with validation control
- final benchmarking on unseen data
- artifacts ready for demonstration in the Gradio app
