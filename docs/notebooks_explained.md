# Notebooks Explained

This guide explains the notebook workflow as if you are studying it for a presentation, viva, or discussion with an instructor. It covers:

1. The idea behind each notebook
2. What each block of code is doing
3. Why the code is written that way
4. What output you should expect

The project uses three notebooks:

1. `1_data_exploration.ipynb`
2. `2_model_training.ipynb`
3. `3_evaluation.ipynb`

The old file `CV dataprocessing&cleaning.ipynb` is now only a redirect notebook that points to the three real deliverables.

---

## 1. `1_data_exploration.ipynb`

### Main Idea

This notebook is the dataset verification and understanding stage.

Before training any deep learning model, we need to confirm:

- the dataset path is correct
- the split folders exist
- the class folders exist
- the image counts make sense
- the images can be opened correctly
- the image sizes and formats are known

This is important because if the dataset is wrong, then training and evaluation will also be wrong.

---

### Markdown Title Cell

The first markdown cell introduces the notebook:

- it says this notebook verifies the dataset structure
- it explains that it will inspect class balance
- it mentions sample image display
- it mentions reporting real image sizes and formats

This cell does not execute code. It helps document the notebook purpose.

---

### First Code Cell: Imports and Paths

```python
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
```

#### What each line does

- `import os`
  - Imports Python’s built-in operating system utilities.
  - In this notebook it is not heavily used, but it is common in dataset scripts.

- `from collections import Counter`
  - Imports `Counter`, which is used to count repeated values.
  - Here it helps count image sizes and image formats.

- `from pathlib import Path`
  - Imports `Path`, which is a cleaner and safer way to work with file paths.
  - Instead of writing long string paths, we can combine folders using `/`.

- `import matplotlib.pyplot as plt`
  - Imports plotting tools.
  - Used for bar charts and image grids.

- `import pandas as pd`
  - Imports pandas for structured tables.
  - Used to display split counts neatly.

- `from PIL import Image`
  - Imports the Python Imaging Library interface.
  - Used to open and inspect image files.

---

Then the notebook defines paths:

```python
DATA_DIR = Path('../data/animals_dataset')
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
```

#### What this does

- `DATA_DIR = Path('../data/animals_dataset')`
  - Defines the dataset root.
  - `..` means “go one folder up” from `notebooks/` to the project root.

- `TRAIN_DIR = DATA_DIR / 'train'`
  - Creates the path to the training folder.

- `VAL_DIR = DATA_DIR / 'val'`
  - Creates the path to the validation folder.

- `TEST_DIR = DATA_DIR / 'test'`
  - Creates the path to the test folder.

- `RESULTS_DIR = Path('../results')`
  - Sets the folder where plots will be saved.

- `RESULTS_DIR.mkdir(exist_ok=True)`
  - Creates `results/` if it does not already exist.
  - `exist_ok=True` means Python should not crash if the folder already exists.

- `IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}`
  - Defines which file extensions count as valid images.
  - This is helpful because the dataset is not restricted to only one extension.

---

Then the notebook validates paths:

```python
assert DATA_DIR.exists(), f'Data not found at {DATA_DIR.resolve()}'
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    assert split_dir.exists(), f'Missing split directory: {split_dir}'

classes = sorted([item.name for item in TRAIN_DIR.iterdir() if item.is_dir()])
print(f'Found {len(classes)} classes: {classes}')
```

#### What this does

- `assert DATA_DIR.exists()`
  - Stops the notebook immediately if the dataset root is missing.
  - This protects the rest of the notebook from failing later in confusing ways.

- The loop over `[TRAIN_DIR, VAL_DIR, TEST_DIR]`
  - Checks that each required split folder exists.

- `TRAIN_DIR.iterdir()`
  - Lists everything inside the training folder.

- `if item.is_dir()`
  - Keeps only directories, which represent class folders.

- `item.name`
  - Extracts just the folder name, like `cat` or `dog`.

- `sorted(...)`
  - Sorts the class names alphabetically.
  - This gives a stable class order across runs.

- `print(...)`
  - Shows how many classes were found and their names.

#### Why this matters

If the class names are wrong at this stage, every later stage would use the wrong label mapping.

---

### Second Code Cell: Counting Images

```python
def image_files(folder: Path):
    return sorted([
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])
```

#### What this function does

This function collects valid image files from one folder.

- `folder.iterdir()`
  - Loops through everything inside the folder.

- `path.is_file()`
  - Keeps only files, not directories.

- `path.suffix.lower() in IMAGE_EXTENSIONS`
  - Keeps only image files with approved extensions.

- `sorted(...)`
  - Keeps the file list in a stable order.

This function is useful because many later steps need “all image files in a class folder.”

---

```python
def count_images(split_dir: Path):
    counts = {}
    for class_name in sorted([item.name for item in split_dir.iterdir() if item.is_dir()]):
        counts[class_name] = len(image_files(split_dir / class_name))
    return counts
```

#### What this function does

It counts how many image files exist inside each class folder for one split.

- `counts = {}`
  - Creates an empty dictionary.

- The loop goes through each class directory in the split.

- `split_dir / class_name`
  - Builds the folder path for that class.

- `image_files(...)`
  - Gets the valid image files.

- `len(...)`
  - Counts them.

- `counts[class_name] = ...`
  - Stores the result, for example:
    - `{'cat': 91, 'cow': 86, ...}`

- `return counts`
  - Gives back the dictionary.

---

Then the notebook applies the function:

```python
train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)
```

This creates one dictionary for each split.

---

Next:

```python
summary_df = pd.DataFrame([
    {'split': 'train', **train_counts},
    {'split': 'val', **val_counts},
    {'split': 'test', **test_counts},
]).set_index('split')
```

#### What this does

- `pd.DataFrame([...])`
  - Creates a table from a list of dictionaries.

- `{'split': 'train', **train_counts}`
  - Builds one row where:
    - `split` is `"train"`
    - the remaining keys and values come from `train_counts`

- `set_index('split')`
  - Makes the `split` column the row label.

The final table is easier to read than raw dictionaries.

---

Then:

```python
display(summary_df)
print('\nDataset totals:')
print(f"Train: {sum(train_counts.values())}")
print(f"Val:   {sum(val_counts.values())}")
print(f"Test:  {sum(test_counts.values())}")
```

#### What this does

- `display(summary_df)`
  - Shows the table nicely in Jupyter.

- `sum(train_counts.values())`
  - Adds all class counts in the train split.

- Similar logic is used for val and test.

#### Why this matters

This gives both class-level and split-level understanding of the dataset.

---

### Third Code Cell: Class Distribution Plot

This cell creates three side-by-side bar charts: one for train, one for validation, and one for test.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
```

- Creates one figure with three subplots in one row.
- `sharey=True` means all three charts use the same y-axis scale.

---

Then the notebook loops through split/count pairs:

```python
for idx, (split_name, counts) in enumerate([
    ('Train', train_counts),
    ('Validation', val_counts),
    ('Test', test_counts),
]):
```

- `enumerate(...)` gives an index and a value.
- `idx` tells the notebook which subplot to use.
- `split_name` is the chart title.
- `counts` is the dictionary for that split.

---

Inside the loop:

```python
axes[idx].bar(counts.keys(), counts.values(), color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
axes[idx].set_title(f'{split_name} Set')
axes[idx].set_xlabel('Class')
axes[idx].tick_params(axis='x', rotation=45)
```

- `bar(...)`
  - Creates the bar chart.
  - The x-axis is class names.
  - The y-axis is image counts.

- `set_title(...)`
  - Adds the subplot title.

- `set_xlabel('Class')`
  - Labels the x-axis.

- `tick_params(... rotation=45)`
  - Rotates class names so they are easier to read.

Then:

```python
axes[0].set_ylabel('Image Count')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

- `axes[0].set_ylabel(...)`
  - Adds a y-axis label to the first subplot.

- `plt.tight_layout()`
  - Reduces overlap between plot elements.

- `plt.savefig(...)`
  - Saves the figure to the `results/` folder.

- `plt.show()`
  - Displays it inside the notebook.

#### Output meaning

This plot helps you explain whether the classes are balanced or imbalanced across splits.

---

### Fourth Code Cell: Sample Images

This cell displays example images from each class.

```python
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
fig.suptitle('Sample Images from Training Set', fontsize=16)
```

- Creates a grid with:
  - one row per class
  - five columns for five images

- `suptitle(...)`
  - Adds one overall title above the figure.

---

```python
if len(classes) == 1:
    axes = [axes]
```

This is a safety check.

If there were only one class, matplotlib would return the axes in a different shape, so this line keeps later code consistent.

---

Then:

```python
for i, class_name in enumerate(classes):
    samples = image_files(TRAIN_DIR / class_name)[:5]
```

- Loops through each class.
- Takes the first five images from that class.

---

Inside the inner loop:

```python
for j in range(5):
    ax = axes[i][j] if len(classes) > 1 else axes[j]
    ax.axis('off')
    if j < len(samples):
        with Image.open(samples[j]) as img:
            ax.imshow(img.convert('RGB'))
    if j == 0:
        ax.set_title(class_name.upper(), fontsize=12, fontweight='bold')
```

#### What this does

- Chooses the correct subplot cell.
- `ax.axis('off')`
  - Hides axes lines and tick marks.

- `if j < len(samples)`
  - Avoids indexing past the available images.

- `Image.open(samples[j])`
  - Opens the image.

- `img.convert('RGB')`
  - Ensures a standard color format.

- `ax.imshow(...)`
  - Draws the image in the subplot.

- `if j == 0`
  - Only the first image in each row gets a class title.

Then it saves the figure:

```python
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'sample_images.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Output meaning

This lets you visually inspect whether the folders actually contain the correct animals and whether the dataset looks reasonable.

---

### Fifth Code Cell: Image Sizes and Formats

```python
size_counter = Counter()
format_counter = Counter()
```

- Creates two counters:
  - one for image sizes
  - one for file formats

---

```python
for class_name in classes:
    for image_path in image_files(TRAIN_DIR / class_name)[:15]:
        with Image.open(image_path) as img:
            size_counter[img.size] += 1
            format_counter[img.format] += 1
```

#### What this does

- Goes through each class.
- Takes up to 15 images per class.
- Opens each image.
- Records:
  - `img.size`, such as `(275, 183)`
  - `img.format`, such as `JPEG`

#### Why only 15 per class

This is a sample-based inspection, not a full dataset-wide expensive scan.

It is enough to understand the data without slowing the notebook too much.

---

Finally:

```python
print('Observed image sizes from training sample:')
for size, count in size_counter.most_common(10):
    print(f'  {size}: {count}')

print('\nObserved file formats from training sample:')
for file_format, count in format_counter.items():
    print(f'  {file_format}: {count}')

print('\nData exploration complete.')
```

#### What this does

- Prints the most common image sizes.
- Prints the observed file formats.
- Prints a completion message.

#### Output meaning

This proves that the original dataset is not necessarily already `224x224`.

That is why the model notebooks explicitly resize images during preprocessing.

---

## 2. `2_model_training.ipynb`

### Main Idea

This notebook handles the learning stage.

It:

- loads training and validation data
- applies preprocessing
- builds a transfer-learning model
- trains the classifier
- watches validation performance
- uses early stopping
- saves the best model

This notebook is the core deep learning stage of the project.

---

### Title Markdown Cell

This cell explains that the notebook trains a transfer-learning classifier and monitors validation performance.

Again, this is documentation, not executable code.

---

### First Code Cell: Setup

```python
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
```

#### What each import does

- `time`
  - Used to measure training duration.

- `Path`
  - Used for file paths.

- `matplotlib.pyplot`
  - Used later for training curve plots.

- `torch`
  - Main PyTorch library.

- `torch.nn as nn`
  - Neural network layers and loss functions.

- `torch.optim as optim`
  - Optimizers like Adam.

- `DataLoader`
  - Loads data in batches during training.

- `datasets, models, transforms`
  - `datasets.ImageFolder` loads folder-structured image datasets.
  - `models` provides pretrained architectures like ResNet18.
  - `transforms` handles image preprocessing and augmentation.

---

Then path and hyperparameter setup:

```python
TRAIN_DIR = Path('../data/animals_dataset/train')
VAL_DIR = Path('../data/animals_dataset/val')
MODELS_DIR = Path('../models')
RESULTS_DIR = Path('../results')
MODEL_SAVE_PATH = MODELS_DIR / 'best_model.pth'
```

- These point to the dataset, output model folder, and result folder.

---

```python
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
```

- Ensures the output folders exist.

---

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

#### What this does

- Checks if a GPU is available through CUDA.
- If yes, training uses the GPU.
- Otherwise, it falls back to CPU.

This makes the notebook portable across machines.

---

```python
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 0
```

#### Meaning of each hyperparameter

- `BATCH_SIZE = 16`
  - Number of images processed before each optimization step.

- `LEARNING_RATE = 1e-3`
  - Step size used by the optimizer.

- `NUM_EPOCHS = 20`
  - Maximum training passes over the full training set.

- `PATIENCE = 5`
  - Number of non-improving validation epochs allowed before early stopping.

- `NUM_WORKERS = 0`
  - Number of subprocesses used by `DataLoader`.
  - `0` is safer across Windows/Jupyter environments.

---

### Second Code Cell: Transforms and Data Loading

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### What this training transform does

- `Resize((224, 224))`
  - Resizes every image to the input size expected by ResNet18.

- `RandomHorizontalFlip(p=0.5)`
  - Randomly flips some images horizontally.
  - This helps the model generalize.

- `RandomRotation(15)`
  - Rotates images up to 15 degrees.

- `ColorJitter(...)`
  - Slightly changes brightness, contrast, and saturation.
  - This teaches the model to be robust to lighting variation.

- `ToTensor()`
  - Converts the image to a PyTorch tensor.

- `Normalize(...)`
  - Standardizes pixel values using ImageNet statistics.
  - This is important because the pretrained ResNet18 expects similarly normalized inputs.

---

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### Why validation transform is simpler

Validation data should not be augmented.

We want validation to reflect real model performance, not random transformed versions.

---

```python
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
```

#### What `ImageFolder` does

It expects the dataset structure:

```text
train/
  cat/
  cow/
  deer/
  dog/
  lion/
```

It automatically:

- reads images
- assigns numeric labels based on folder names
- stores class names in `dataset.classes`

---

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

#### Why two loaders

- `train_loader`
  - Uses `shuffle=True` so training order changes each epoch.

- `val_loader`
  - Uses `shuffle=False` because validation does not need randomness.

---

```python
class_names = train_dataset.classes
num_classes = len(class_names)

print(f'Classes ({num_classes}): {class_names}')
print(f'Train samples: {len(train_dataset)}')
print(f'Val samples: {len(val_dataset)}')
```

This shows:

- the class order
- the number of output classes
- the number of training and validation images

The class order is especially important because it defines the label mapping used later by the model and app.

---

### Third Code Cell: Model Setup

```python
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

#### What this means

- Loads a pretrained ResNet18 model.
- The weights come from ImageNet pretraining.

This is called transfer learning.

Instead of training from scratch, the model starts with useful visual features already learned from a huge dataset.

---

```python
for param in model.parameters():
    param.requires_grad = False
```

#### What this does

It freezes all pretrained layers.

That means:

- the convolutional backbone will not update during training
- only the final classification layer will be trained

This is useful when:

- the dataset is small
- you want faster training
- you want to reduce overfitting risk

---

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
```

#### What this does

- Replaces the final fully connected layer.
- The old layer was built for ImageNet’s original classes.
- The new layer outputs exactly `num_classes`, which is 5 here.

`model.to(device)` moves the model to GPU or CPU.

---

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

#### What each item means

- `criterion = nn.CrossEntropyLoss()`
  - Standard classification loss for multi-class problems.

- `optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)`
  - Uses Adam to update only the final layer parameters.

- `ReduceLROnPlateau(...)`
  - Reduces the learning rate when validation loss stops improving.
  - `mode='min'` means lower validation loss is better.
  - `factor=0.5` halves the learning rate when triggered.
  - `patience=2` waits for two bad epochs first.

---

### Fourth Code Cell: `run_epoch` Function

This is the core training/validation function.

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
```

If `optimizer` is provided, the function behaves as training.

If `optimizer` is `None`, it behaves as evaluation.

---

```python
is_training = optimizer is not None
model.train() if is_training else model.eval()
```

- Detects whether the current epoch is training or validation.
- `model.train()`
  - Enables training behavior.
- `model.eval()`
  - Switches the model into evaluation mode.

This matters especially for layers like dropout or batch normalization.

---

```python
running_loss = 0.0
correct = 0
total = 0
```

These variables accumulate statistics across all batches.

---

```python
context = torch.enable_grad() if is_training else torch.no_grad()
with context:
```

#### Why this is smart

- During training, gradients must be tracked.
- During validation, gradients are unnecessary.
- `torch.no_grad()` saves memory and speeds up validation.

---

Inside the loop:

```python
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
```

- Gets one batch of images and labels.
- Moves both to the same device as the model.

---

```python
if is_training:
    optimizer.zero_grad()
```

- Clears old gradients before the new backward pass.

Without this, gradients would accumulate incorrectly.

---

```python
outputs = model(images)
loss = criterion(outputs, labels)
```

- `model(images)` performs the forward pass.
- `loss = criterion(...)` computes classification loss.

---

```python
if is_training:
    loss.backward()
    optimizer.step()
```

#### What this means

- `loss.backward()`
  - Computes gradients.

- `optimizer.step()`
  - Updates the trainable weights.

These happen only during training, not validation.

---

```python
running_loss += loss.item()
predictions = outputs.argmax(dim=1)
total += labels.size(0)
correct += (predictions == labels).sum().item()
```

#### What this does

- Adds the batch loss to the total.
- Gets the predicted class index for each image.
- Adds batch size to total sample count.
- Adds number of correct predictions.

---

After the loop:

```python
epoch_loss = running_loss / max(len(loader), 1)
epoch_acc = 100.0 * correct / max(total, 1)
return epoch_loss, epoch_acc
```

- Computes average loss per batch.
- Computes accuracy percentage.
- Returns both values.

`max(..., 1)` avoids division by zero if something unexpected happens.

---

### Fifth Code Cell: Training Loop

```python
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0
```

#### What these variables do

- `history`
  - Stores values for later plotting.

- `best_val_loss`
  - Tracks the best validation loss seen so far.

- `patience_counter`
  - Counts how many epochs in a row failed to improve validation loss.

---

```python
start_time = time.time()
```

- Records training start time.

---

Then the epoch loop:

```python
for epoch in range(NUM_EPOCHS):
```

This repeats up to 20 epochs unless early stopping stops training sooner.

---

```python
train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
```

- First line runs one training epoch.
- Second line runs one validation epoch.

Notice validation does not pass an optimizer.

---

```python
history['train_loss'].append(train_loss)
history['train_acc'].append(train_acc)
history['val_loss'].append(val_loss)
history['val_acc'].append(val_acc)
```

This stores the metrics after each epoch.

---

```python
scheduler.step(val_loss)
```

This allows the scheduler to watch validation loss and reduce learning rate if improvement stalls.

---

```python
print(
    f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
)
```

This prints clean progress after each epoch.

---

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'best_val_loss': best_val_loss,
            'history': history,
        },
        MODEL_SAVE_PATH,
    )
    print('  Saved best model checkpoint.')
```

#### What this does

If current validation loss is the best so far:

- update `best_val_loss`
- reset early stopping counter
- save the current model checkpoint

The checkpoint includes:

- model weights
- class names
- best validation loss
- training history

Saving `class_names` is very useful because the app and evaluation notebook need the same label order.

---

```python
else:
    patience_counter += 1
    print(f'  No improvement ({patience_counter}/{PATIENCE})')
```

If validation loss does not improve:

- increase the counter
- print a warning message

---

```python
if patience_counter >= PATIENCE:
    print(f'Early stopping triggered at epoch {epoch + 1}.')
    break
```

This is early stopping.

If the model stops improving for several epochs, training ends early instead of wasting time and risking overfitting.

---

After training:

```python
training_minutes = (time.time() - start_time) / 60
print(f'Training completed in {training_minutes:.2f} minutes')
print(f'Best validation loss: {best_val_loss:.4f}')
```

This reports total runtime and the best validation loss achieved.

---

### Sixth Code Cell: Training Curves

This cell visualizes training history.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

- Creates two plots side by side.

---

Loss plot:

```python
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('Training and Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
```

This shows whether:

- loss is decreasing
- validation tracks training reasonably well
- overfitting may be happening

---

Accuracy plot:

```python
axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
axes[1].set_title('Training and Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
```

This shows how classification performance improves over epochs.

---

Saving the output:

```python
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'Best model saved to {MODEL_SAVE_PATH}')
```

This stores the plot in `results/` and confirms the checkpoint path.

---

## 3. `3_evaluation.ipynb`

### Main Idea

This notebook is the final benchmarking stage.

It evaluates the saved model using the unseen test split and produces:

- overall accuracy
- per-class precision
- per-class recall
- per-class F1-score
- confusion matrix
- sample predictions

This is the notebook that produces the final performance report required academically.

---

### Title Markdown Cell

This cell documents that the notebook loads the saved model and evaluates it on the test split.

---

### First Code Cell: Imports and Setup

The imports are similar to training, but now there are extra metric libraries:

- `numpy`
  - For numerical arrays and random sample selection.

- `seaborn`
  - For nicer confusion matrix visualization.

- `sklearn.metrics`
  - For standard evaluation metrics.

---

The path variables:

```python
TEST_DIR = Path('../data/animals_dataset/test')
MODEL_PATH = Path('../models/best_model.pth')
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)
```

These point to:

- test images
- trained model
- output folder

---

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

Same idea as training: use GPU if possible, CPU otherwise.

---

### Second Code Cell: Test Data and Model Loading

```python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This is the same preprocessing logic used for validation.

No random augmentation is used, because test evaluation should be stable and unbiased.

---

```python
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
```

- Loads the test dataset.
- Creates batches.
- Keeps order fixed by using `shuffle=False`.

---

```python
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
```

#### Why this matters

Instead of hardcoding class names again, the notebook reads them from the saved checkpoint.

This prevents label mismatch between:

- training
- evaluation
- app

---

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```

#### What this does

- Builds the same architecture shape used during training.
- Sets the last layer to the correct number of classes.
- Loads trained weights from the checkpoint.
- Moves the model to the correct device.
- Switches the model to evaluation mode.

---

### Third Code Cell: Prediction Loop

```python
all_predictions = []
all_labels = []
```

These lists will store all predicted and true labels for the entire test set.

---

```python
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
```

#### Step-by-step meaning

- `torch.no_grad()`
  - No gradients are needed during testing.

- Loop over batches from the test loader.

- `images = images.to(device)`
  - Moves the input batch to the device.

- `outputs = model(images)`
  - Runs the forward pass.

- `argmax(dim=1)`
  - Takes the class with highest score for each image.

- `predictions.cpu().numpy()`
  - Moves predictions back to CPU and converts to NumPy.

- `extend(...)`
  - Adds the batch results to the overall list.

At the end, you have predictions for the whole test set.

---

Then:

```python
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
```

This converts lists to NumPy arrays for metric functions.

---

### Fourth Code Cell: Metric Calculation

```python
test_accuracy = accuracy_score(all_labels, all_predictions)
```

- Computes overall accuracy.
- Accuracy is:
  - correct predictions / total predictions

---

```python
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    labels=range(len(class_names)),
    average=None,
)
```

#### Meaning

This calculates class-by-class metrics.

- `precision`
  - Of all items predicted as a class, how many were correct?

- `recall`
  - Of all true items in a class, how many were found?

- `f1`
  - Harmonic mean of precision and recall.

- `support`
  - Number of true examples in that class.

- `average=None`
  - Means “return values for each class separately.”

---

```python
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average='macro',
)
```

#### Meaning

This calculates macro averages:

- each class contributes equally
- useful when classes are not perfectly balanced

---

Then the notebook prints metrics clearly so you can read them in the notebook output.

This printed block is often useful during debugging and for report writing.

---

### Fifth Code Cell: Classification Report File

```python
report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
report_path = RESULTS_DIR / 'classification_report.txt'
```

- Creates a formatted multi-line report string.
- Sets the output file path.

---

```python
with report_path.open('w', encoding='utf-8') as file:
    file.write('ANIMAL CLASSIFICATION - TEST SET RESULTS\n')
    file.write('=' * 60 + '\n\n')
    file.write(f'Overall Test Accuracy: {test_accuracy * 100:.2f}%\n\n')
    file.write(report)
```

#### What this does

- Opens the output file in write mode.
- Writes a custom header.
- Writes overall accuracy.
- Writes the full classification report.

This file becomes a permanent text summary of performance.

---

### Sixth Code Cell: Confusion Matrix

```python
cm = confusion_matrix(all_labels, all_predictions)
```

#### What this means

A confusion matrix shows:

- rows = true class
- columns = predicted class

Correct predictions are usually on the diagonal.

Wrong predictions appear off the diagonal.

---

Then the plotting block:

```python
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - Accuracy: {test_accuracy * 100:.2f}%')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### What each key argument does

- `annot=True`
  - Writes the actual numbers inside the heatmap cells.

- `fmt='d'`
  - Displays integers.

- `cmap='Blues'`
  - Uses a blue color map.

- `xticklabels`, `yticklabels`
  - Uses class names on both axes.

This plot is very useful for explaining specific error patterns.

---

### Seventh Code Cell: Sample Predictions

```python
indices = np.random.choice(len(test_dataset), size=min(9, len(test_dataset)), replace=False)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()
```

#### What this does

- Randomly selects up to 9 test images.
- Creates a 3x3 plot grid.
- Flattens the axes array so it is easy to index.

---

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
```

These tensors are used to reverse normalization for display.

When images are normalized for the model, colors no longer look natural to humans.

So the notebook denormalizes them before plotting.

---

Inside the loop:

```python
img, true_label = test_dataset[sample_idx]
display_img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
```

#### What this does

- `img` is already transformed.
- `img * std + mean`
  - Reverses normalization.
- `clamp(0, 1)`
  - Keeps pixel values valid.
- `permute(1, 2, 0)`
  - Converts tensor shape from channel-first to image format.
- `.numpy()`
  - Converts to NumPy for plotting.

---

Then prediction:

```python
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    probs = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(probs, dim=1)
```

#### What this means

- `unsqueeze(0)`
  - Adds a batch dimension so the single image becomes a batch of size 1.

- `softmax(...)`
  - Converts raw output scores into probabilities.

- `torch.max(...)`
  - Finds the most probable class and its confidence.

---

Then:

```python
color = 'green' if pred_label == true_label else 'red'
```

- Green title means prediction is correct.
- Red title means prediction is wrong.

---

Finally:

```python
axes[plot_idx].imshow(display_img)
axes[plot_idx].axis('off')
axes[plot_idx].set_title(
    f'True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence * 100:.1f}%)',
    color=color,
    fontsize=10,
    fontweight='bold',
)
```

This displays:

- the image
- true class
- predicted class
- confidence score
- green or red correctness indicator

Then the plot is saved as `sample_predictions.png`.

---

## 4. `CV dataprocessing&cleaning.ipynb`

### Main Idea

This file is no longer the main implementation notebook.

It now acts as a simple redirect notebook so anyone opening the older filename can immediately see:

- the new notebook order
- the correct dataset convention
- the normalized class names

This avoids confusion and keeps backward compatibility with your earlier project structure.

---

## How the Three Notebooks Work Together

### Step 1: Exploration

Notebook 1 checks the data.

Without this, training could fail because of wrong paths, missing folders, or bad assumptions.

### Step 2: Training

Notebook 2 learns from the train split and uses the validation split to decide when to stop.

This creates the trained checkpoint.

### Step 3: Evaluation

Notebook 3 loads the saved model and measures final performance on the unseen test split.

This produces the final academic metrics and visuals.

---

## What You Should Be Able to Explain to Your Instructor

After studying these notebooks, you should be able to explain:

- why the dataset must be checked before training
- why images are resized to `224x224`
- why augmentation is used only for training
- what transfer learning means
- why ResNet18 was selected
- what validation loss is used for
- why early stopping helps avoid overfitting
- how the best model is saved
- why test data must remain unseen during training
- what accuracy, precision, recall, F1-score, and confusion matrix mean

If you can explain those clearly, you will already understand the codebase at a strong level.
