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
  - Imports Python's built-in operating system utilities.
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
  - `..` means "go one folder up" from `notebooks/` to the project root.

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
def image_files(folder):
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
```

#### What this function does

This function collects valid image files from one folder.

- `folder.iterdir()`
  - Goes through every file and folder inside.

- `f.suffix.lower() in IMAGE_EXTENSIONS`
  - Keeps only files whose extension is `.jpg`, `.jpeg`, or `.png`.
  - `.lower()` handles cases like `.JPG` or `.PNG`.

- `sorted(...)`
  - Returns the file list in alphabetical order, so results are consistent every run.

This function is useful because many later steps need "all image files in a class folder."

---

```python
def count_images(split_dir):
    return {d.name: len(image_files(d)) for d in sorted(split_dir.iterdir()) if d.is_dir()}
```

#### What this function does

It counts how many image files exist inside each class folder for one split.

This is written as a **dictionary comprehension**, which is a compact Python pattern for building a dictionary in one line.

Breaking it down:

- `for d in sorted(split_dir.iterdir())`
  - Loops through each item in the split folder in alphabetical order.

- `if d.is_dir()`
  - Keeps only folders (each folder is a class).

- `d.name`
  - Gets the class name, like `cat` or `dog`.

- `len(image_files(d))`
  - Calls the function above to get all images in that class folder and counts them.

- The result looks like: `{'cat': 91, 'cow': 86, 'deer': 87, 'dog': 111, 'lion': 89}`

---

Then the notebook applies the function:

```python
train_counts = count_images(TRAIN_DIR)
val_counts   = count_images(VAL_DIR)
test_counts  = count_images(TEST_DIR)
```

This creates one dictionary for each split.

---

Next:

```python
summary_df = pd.DataFrame([
    {'split': 'train', **train_counts},
    {'split': 'val',   **val_counts},
    {'split': 'test',  **test_counts},
]).set_index('split')
```

#### What this does

- `pd.DataFrame([...])`
  - Creates a table from a list of dictionaries.

- `{'split': 'train', **train_counts}`
  - Builds one row where:
    - `split` is `"train"`
    - `**train_counts` unpacks the dictionary so each class becomes its own column.

- `set_index('split')`
  - Makes the `split` column the row label instead of a regular column.

The final table is easier to read than raw dictionaries.

---

Then:

```python
display(summary_df)
print(f"\nTrain: {sum(train_counts.values())}  Val: {sum(val_counts.values())}  Test: {sum(test_counts.values())}")
```

#### What this does

- `display(summary_df)`
  - Shows the table nicely in Jupyter with proper formatting.

- The `print(...)` line
  - Adds all class counts in each split to give the total image count per split.
  - All three totals are printed on one line for brevity.

#### Why this matters

This gives both class-level and split-level understanding of the dataset at a glance.

---

### Third Code Cell: Class Distribution Plot

This cell creates three side-by-side bar charts: one for train, one for validation, and one for test.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
```

- Creates one figure with three subplots in one row.
- `sharey=True` means all three charts use the same y-axis scale, so you can compare heights visually.

---

Then the notebook loops through split/count pairs:

```python
for idx, (split_name, counts) in enumerate([
    ('Train', train_counts),
    ('Validation', val_counts),
    ('Test', test_counts),
]):
```

- `enumerate(...)` gives both an index number and a value at each step.
- `idx` tells the notebook which subplot to draw into.
- `split_name` is the chart title like `"Train"`.
- `counts` is the dictionary of class counts for that split.

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
  - `counts.keys()` gives the class names for the x-axis.
  - `counts.values()` gives the image counts for bar heights.
  - The color list assigns one color per class.

- `set_title(...)`
  - Adds the subplot title.

- `set_xlabel('Class')`
  - Labels the x-axis.

- `tick_params(... rotation=45)`
  - Rotates class names by 45 degrees so they do not overlap.

Then:

```python
axes[0].set_ylabel('Image Count')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

- `axes[0].set_ylabel(...)`
  - Adds a y-axis label to the first subplot only (since all share the same scale).

- `plt.tight_layout()`
  - Reduces overlap between plot elements automatically.

- `plt.savefig(...)`
  - Saves the figure to the `results/` folder as a high-quality PNG.

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
  - five columns for five images per class

- `suptitle(...)`
  - Adds one overall title above the entire figure.

---

```python
if len(classes) == 1:
    axes = [axes]
```

This is a safety check.

If there were only one class, matplotlib would return the axes in a different shape, so this line keeps later code consistent regardless.

---

Then:

```python
for i, class_name in enumerate(classes):
    samples = image_files(TRAIN_DIR / class_name)[:5]
```

- Loops through each class.
- Calls `image_files()` to get all images in that class folder.
- `[:5]` takes only the first five.

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

- `axes[i][j]`
  - Selects the subplot at row `i`, column `j`.

- `ax.axis('off')`
  - Hides the axis lines and tick marks so only the image is visible.

- `if j < len(samples)`
  - Avoids indexing past the end if a class has fewer than 5 images.

- `Image.open(samples[j])`
  - Opens the image file from disk.

- `img.convert('RGB')`
  - Forces a standard 3-channel color format so matplotlib can display it correctly.

- `ax.imshow(...)`
  - Draws the image into the subplot.

- `if j == 0`
  - Only the first image in each row gets a class name label.

Then it saves the figure:

```python
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'sample_images.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Output meaning

This lets you visually inspect whether the folders actually contain the correct animals and whether the dataset looks reasonable before training.

---

### Fifth Code Cell: Image Sizes and Formats

```python
size_counter = Counter()
format_counter = Counter()
```

- Creates two counters:
  - one for image sizes like `(275, 183)`
  - one for file formats like `JPEG`

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
- Takes up to 15 images per class as a sample.
- Opens each image.
- Records `img.size` (width, height) and `img.format` (JPEG, PNG, etc.).

#### Why only 15 per class

This is a sample-based inspection, not a full dataset-wide scan.

It is enough to understand the data without slowing the notebook.

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

- Prints the most common image dimensions.
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
  - Used to measure how long training takes.

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
  - `datasets.ImageFolder` loads folder-structured image datasets automatically.
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

- Ensures the output folders exist before trying to save anything.

---

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

#### What this does

- Checks if a GPU is available through CUDA.
- If yes, training uses the GPU which is much faster.
- Otherwise, it falls back to CPU.

This makes the notebook portable across machines with and without a GPU.

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
  - Number of images processed before each weight update.
  - Smaller batches use less memory but train slightly slower.

- `LEARNING_RATE = 1e-3`
  - This is `0.001`. It controls how big each update step is.

- `NUM_EPOCHS = 20`
  - Maximum number of passes over the full training set.

- `PATIENCE = 5`
  - If validation loss does not improve for 5 consecutive epochs, training stops early.

- `NUM_WORKERS = 0`
  - Number of background processes used by `DataLoader` to load images.
  - `0` means loading happens in the main process, which is safer on Windows and in Jupyter.

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

`transforms.Compose` chains multiple steps together. Each image passes through them in order.

- `Resize((224, 224))`
  - Resizes every image to 224×224 pixels, which is the input size ResNet18 expects.

- `RandomHorizontalFlip(p=0.5)`
  - Randomly flips some images left-to-right.
  - This teaches the model that the same animal flipped is still the same animal.

- `RandomRotation(15)`
  - Randomly rotates images up to 15 degrees.
  - This makes the model more robust to animals not perfectly centered.

- `ColorJitter(...)`
  - Slightly changes brightness, contrast, and saturation randomly.
  - This teaches the model to handle different lighting conditions.

- `ToTensor()`
  - Converts the image from a PIL Image into a PyTorch tensor.
  - Pixel values go from 0–255 to 0.0–1.0.

- `Normalize(...)`
  - Shifts pixel values to match ImageNet statistics (mean and standard deviation).
  - This is required because ResNet18 was pretrained on ImageNet with these exact values.

---

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### Why validation transform is simpler

Validation data should never be randomly augmented.

We want validation to reflect real model performance, not performance on randomly altered images.

So we only resize, convert, and normalize — no random flipping or color changes.

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

- reads images from each class folder
- assigns a numeric label based on alphabetical folder order (cat=0, cow=1, deer=2, dog=3, lion=4)
- stores class names in `dataset.classes`

---

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

#### Why two loaders

- `train_loader`
  - `shuffle=True` randomizes the order of images each epoch so the model does not memorize order.

- `val_loader`
  - `shuffle=False` keeps validation order consistent so results are reproducible.

---

```python
class_names = train_dataset.classes
num_classes = len(class_names)

print(f'Classes ({num_classes}): {class_names}')
print(f'Train samples: {len(train_dataset)}')
print(f'Val samples: {len(val_dataset)}')
```

This shows:

- the class order (which defines the label mapping)
- the number of output classes needed in the model
- how many images are in each split

The class order is especially important because every later step — evaluation, app — must use the same order.

---

### Third Code Cell: Model Setup

```python
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

#### What this means

- Loads a pretrained ResNet18 model.
- `DEFAULT` means use the best available pretrained weights, which come from ImageNet.

This is called **transfer learning**: instead of training from scratch, the model starts with useful visual features already learned from millions of images.

---

```python
for param in model.parameters():
    param.requires_grad = False
```

#### What this does

It **freezes** all pretrained layers.

That means:

- the convolutional backbone will not change during our training
- only the final classification layer will learn

This is the right choice when:

- the dataset is small (we have ~464 training images)
- we want faster training
- we want to reduce the risk of overfitting

---

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
```

#### What this does

- Replaces the final fully connected layer.
- The original layer was built for ImageNet's 1000 classes.
- The new layer outputs exactly `num_classes` values, which is 5 here.

`model.to(device)` moves the entire model to the selected device (GPU or CPU).

---

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

#### What each item means

- `criterion = nn.CrossEntropyLoss()`
  - The loss function for multi-class classification.
  - It measures how wrong the model's predictions are.

- `optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)`
  - Adam optimizer will update only the final layer's parameters.
  - `model.fc.parameters()` means "only touch the new classification layer."

- `ReduceLROnPlateau(...)`
  - Reduces the learning rate automatically when validation loss stops improving.
  - `mode='min'`: lower is better for loss.
  - `factor=0.5`: halves the learning rate when triggered.
  - `patience=2`: waits for 2 non-improving epochs before reducing.

---

### Fourth Code Cell: `run_epoch` Function

This is the most important function in the training notebook. It handles one full pass through the data, for either training or validation.

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
```

- `model`: the neural network.
- `loader`: either `train_loader` or `val_loader`.
- `criterion`: the loss function.
- `device`: CPU or GPU.
- `optimizer`: if provided, this is a training epoch. If `None`, this is a validation epoch.

The trick of using `optimizer=None` to signal "validation mode" keeps the code clean: one function handles both cases.

---

```python
is_training = optimizer is not None
model.train() if is_training else model.eval()
```

- `is_training` becomes `True` for training, `False` for validation.
- `model.train()` enables training behavior (e.g., dropout is active).
- `model.eval()` switches to evaluation behavior (e.g., dropout is disabled, batch norm uses running stats).

This distinction matters because some layers behave differently during training versus inference.

---

```python
running_loss, correct, total = 0.0, 0, 0
```

Three counters on one line:

- `running_loss`: accumulates the total loss across all batches.
- `correct`: accumulates the number of correct predictions.
- `total`: accumulates the total number of images seen.

---

```python
with torch.set_grad_enabled(is_training):
```

#### What this does

`torch.set_grad_enabled(is_training)` is a **context manager** that controls whether PyTorch tracks gradients.

- When `is_training=True`, gradients are tracked. This is required for backpropagation.
- When `is_training=False`, gradients are not tracked. This saves memory and makes validation faster.

This is simpler and more readable than writing:

```python
# old way — harder to read
context = torch.enable_grad() if is_training else torch.no_grad()
with context:
```

`torch.set_grad_enabled()` does exactly the same thing in one clean line.

---

Inside the loop:

```python
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
```

- Gets one batch of images and labels.
- Moves both to the same device as the model (GPU or CPU).

---

```python
if is_training:
    optimizer.zero_grad()
```

- Clears old gradients before computing new ones.
- Without this, gradients would accumulate from previous batches and corrupt the update.

---

```python
outputs = model(images)
loss = criterion(outputs, labels)
```

- `model(images)`: forward pass — produces one score per class for each image.
- `criterion(outputs, labels)`: computes how wrong those scores are compared to the true labels.

---

```python
if is_training:
    loss.backward()
    optimizer.step()
```

- `loss.backward()`: computes gradients (how to adjust each weight to reduce the loss).
- `optimizer.step()`: applies those adjustments to the model weights.

These two lines only happen during training. Validation skips them because we are only measuring performance, not updating anything.

---

```python
running_loss += loss.item()
correct += (outputs.argmax(dim=1) == labels).sum().item()
total += labels.size(0)
```

#### What this does

- `loss.item()`: converts the loss tensor to a plain Python number and adds it to the total.

- `outputs.argmax(dim=1)`: finds the class with the highest score for each image in the batch.
  - `dim=1` means "take the max along the class dimension."

- `== labels`: compares predicted class indices to true labels element-by-element.

- `.sum().item()`: counts how many predictions were correct in this batch.

- `labels.size(0)`: the number of images in this batch (batch size).

---

After all batches:

```python
return running_loss / len(loader), 100.0 * correct / total
```

- `running_loss / len(loader)`: average loss per batch across the epoch.
- `100.0 * correct / total`: accuracy as a percentage.

Both are returned together so the training loop can log and compare them.

---

### Fifth Code Cell: Training Loop

```python
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0
```

#### What these variables do

- `history`: stores metrics after each epoch for later plotting.
- `best_val_loss`: initialized to infinity. Any real loss will be smaller, so the first epoch always saves a checkpoint.
- `patience_counter`: counts how many epochs in a row had no improvement.

---

```python
start_time = time.time()
```

- Records the current time so we can measure total training duration.

---

Then the epoch loop:

```python
for epoch in range(NUM_EPOCHS):
```

This repeats up to 20 epochs unless early stopping ends training sooner.

---

```python
train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
```

- First line runs one full training epoch (optimizer is passed in, so weights are updated).
- Second line runs one full validation epoch (no optimizer, so weights are not changed).

---

```python
history['train_loss'].append(train_loss)
history['train_acc'].append(train_acc)
history['val_loss'].append(val_loss)
history['val_acc'].append(val_acc)
```

Stores the four metrics after each epoch so we can plot them later.

---

```python
scheduler.step(val_loss)
```

Gives the current validation loss to the scheduler. If it has not improved for 2 epochs, the scheduler halves the learning rate.

---

```python
print(
    f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
)
```

- `{epoch + 1:02d}` formats the epoch number with a leading zero if needed, like `01`, `02`.
- `:.4f` means four decimal places for loss values.
- `:.2f` means two decimal places for accuracy percentages.

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

If the current validation loss is the best seen so far:

- update `best_val_loss` to the new value
- reset `patience_counter` to zero (we have improvement, so start counting again)
- save a checkpoint with model weights, class names, best loss, and training history

Saving `class_names` inside the checkpoint is very important. It means the evaluation notebook and the app can always use the exact same label order as training — no risk of mismatch.

---

```python
else:
    patience_counter += 1
    print(f'  No improvement ({patience_counter}/{PATIENCE})')
```

If validation loss did not improve:

- increment the counter
- print a message showing how close we are to early stopping

---

```python
if patience_counter >= PATIENCE:
    print(f'Early stopping triggered at epoch {epoch + 1}.')
    break
```

This is **early stopping**.

If the model stops improving for 5 epochs in a row, training ends. This prevents wasting time and reduces overfitting risk.

---

After training:

```python
training_minutes = (time.time() - start_time) / 60
print(f'Training completed in {training_minutes:.2f} minutes')
print(f'Best validation loss: {best_val_loss:.4f}')
```

Reports total runtime and the best validation loss achieved.

---

### Sixth Code Cell: Training Curves

This cell visualizes the training history.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

- Creates two plots side by side: one for loss, one for accuracy.

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

- loss is decreasing over epochs (good)
- validation loss tracks training loss (good, means no overfitting)
- validation loss starts rising while training loss falls (bad, means overfitting)

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

Stores the plot in `results/` and confirms the checkpoint path.

---

## 3. `3_evaluation.ipynb`

### Main Idea

This notebook is the final benchmarking stage.

It evaluates the saved model using the unseen test split and produces:

- overall accuracy
- per-class precision, recall, and F1-score
- confusion matrix
- sample predictions with images

This is the notebook that produces the final performance report required academically.

---

### Title Markdown Cell

This cell documents that the notebook loads the saved model and evaluates it on the test split.

---

### First Code Cell: Imports and Setup

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
```

#### What each import does

- `numpy` — for numerical arrays and random sample selection.
- `seaborn` — for nicer confusion matrix visualization (the `heatmap` function).
- `sklearn.metrics` — provides three functions:
  - `accuracy_score`: overall accuracy.
  - `classification_report`: precision, recall, F1-score for each class, plus averages. Can return a formatted string or a dictionary.
  - `confusion_matrix`: the count matrix of true vs predicted labels.

Notice that `precision_recall_fscore_support` is **not** imported here. In this version, `classification_report` handles everything — it returns all the same numbers without needing a separate import.

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

Same idea as training: use GPU if available, CPU otherwise.

---

### Second Code Cell: Test Data and Model Loading

```python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This is the same preprocessing used for validation.

No random augmentation — test evaluation must be stable and unbiased.

---

```python
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
```

- Loads the test dataset.
- Creates batches of 16 images.
- `shuffle=False` keeps the order fixed so results are reproducible.

---

```python
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
```

#### Why this matters

Instead of hardcoding class names again, the notebook reads them from the saved checkpoint.

This guarantees the label mapping matches exactly what was used during training. If we hardcoded them separately, a typo or different order would cause wrong predictions silently.

---

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```

#### What this does

- `weights=None`: do not download pretrained weights — we are about to load our own trained weights.
- `model.fc = nn.Linear(...)`: rebuilds the same final layer shape used during training.
- `model.load_state_dict(...)`: loads the trained weights from the checkpoint.
- `model.to(device)`: moves the model to the correct device.
- `model.eval()`: switches to evaluation mode (disables dropout, etc.).

---

### Third Code Cell: Prediction Loop and Metrics

```python
all_predictions = []
all_labels = []
```

Two empty lists that will store all predicted and true labels for the entire test set.

---

```python
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
```

#### Step-by-step meaning

- `torch.no_grad()`: disables gradient tracking. We do not need gradients for evaluation.

- `images.to(device)`: moves the batch to the correct device.

- `model(images.to(device))`: forward pass — produces class scores for each image.

- `outputs.argmax(dim=1)`: picks the class with the highest score for each image.

- `.cpu().numpy()`: moves the predictions from the device back to CPU and converts to NumPy.

- `extend(...)`: adds each batch's results to the full list.

After the loop, `all_predictions` contains one predicted class index per test image, and `all_labels` contains the true class indices.

---

```python
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
```

Converts both lists to NumPy arrays, which is what the sklearn metric functions expect.

---

```python
test_accuracy = accuracy_score(all_labels, all_predictions)
```

Computes overall accuracy:

- correct predictions ÷ total predictions

---

```python
report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
```

#### What this does

`classification_report` normally produces a nicely formatted text block. With `output_dict=True`, it returns a **dictionary** instead.

The dictionary looks like this (simplified):

```python
{
    'cat':       {'precision': 1.0, 'recall': 0.88, 'f1-score': 0.94, 'support': 17},
    'cow':       {'precision': 1.0, 'recall': 1.0,  'f1-score': 1.0,  'support': 16},
    ...
    'macro avg': {'precision': 0.98, 'recall': 0.98, 'f1-score': 0.98, 'support': 82},
}
```

This is cleaner than calling `precision_recall_fscore_support` twice (once for per-class, once for macro averages). One call gives everything.

---

```python
print(f'Overall test accuracy: {test_accuracy * 100:.2f}%')
for class_name in class_names:
    m = report_dict[class_name]
    print(f"{class_name:<5} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1-score']:.4f} | Support: {int(m['support'])}")
macro = report_dict['macro avg']
print(f"Macro averages | Precision: {macro['precision']:.4f} | Recall: {macro['recall']:.4f} | F1: {macro['f1-score']:.4f}")
```

#### Breaking this down

- `test_accuracy * 100` converts the 0–1 accuracy to a percentage.
- `for class_name in class_names` loops through each of the 5 classes.
- `report_dict[class_name]` fetches the metrics dictionary for that class.
- `m['precision']`, `m['recall']`, `m['f1-score']`, `m['support']` extract each metric value.
- `{class_name:<5}` left-aligns the class name in a 5-character wide field so all rows line up.
- `report_dict['macro avg']` fetches the macro averages row.

**Macro average** means: compute each metric per class, then take the simple average across classes. Each class contributes equally regardless of size.

---

### Fourth Code Cell: Classification Report File

```python
report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
report_path = RESULTS_DIR / 'classification_report.txt'
```

- This time `classification_report` is called **without** `output_dict=True`.
- It returns a formatted text string suitable for saving to a file.
- `digits=4` means four decimal places.

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
- Writes a custom header line.
- Writes the overall accuracy.
- Writes the full classification report text.

This file becomes a permanent text record of performance you can include in your report.

---

### Fifth Code Cell: Confusion Matrix

```python
cm = confusion_matrix(all_labels, all_predictions)
```

#### What a confusion matrix is

A confusion matrix is a table where:

- rows = true class
- columns = predicted class

Correct predictions appear on the diagonal.
Wrong predictions appear off the diagonal.

For example, if the model predicted "deer" for 2 images that were actually "cat", those 2 would appear in the cat row, deer column.

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

- `annot=True`: writes the actual count numbers inside each cell.
- `fmt='d'`: formats the annotations as plain integers.
- `cmap='Blues'`: uses a blue color scale (darker = higher count).
- `xticklabels`, `yticklabels`: uses class names on both axes instead of numbers.

This plot is very useful for explaining specific error patterns to an instructor.

---

### Sixth Code Cell: Sample Predictions

```python
indices = np.random.choice(len(test_dataset), size=min(9, len(test_dataset)), replace=False)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()
```

#### What this does

- Randomly selects up to 9 test images.
- Creates a 3×3 grid of subplots.
- `axes.ravel()` flattens the 2D grid of axes into a 1D list so we can index easily.

---

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
```

These tensors reverse the normalization applied during preprocessing.

When images are normalized for the model, pixel values are no longer in the 0–1 range that matplotlib expects. So we must undo the normalization before displaying.

`view(3, 1, 1)` reshapes the tensors so they broadcast correctly across the image dimensions.

---

Inside the loop:

```python
img, true_label = test_dataset[sample_idx]
display_img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
```

#### What this does step by step

- `test_dataset[sample_idx]`: gets the transformed image tensor and its true label.
- `img * std + mean`: reverses the normalization (undoes `Normalize`).
- `.clamp(0, 1)`: ensures pixel values stay within the valid 0–1 range.
- `.permute(1, 2, 0)`: changes tensor shape from `(C, H, W)` to `(H, W, C)` which matplotlib requires.
- `.numpy()`: converts the tensor to a NumPy array.

---

Then prediction:

```python
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    probs = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(probs, dim=1)
```

#### What this means

- `img.unsqueeze(0)`: adds a batch dimension. The model expects input shape `(batch, C, H, W)` but a single image is `(C, H, W)`. `unsqueeze(0)` makes it `(1, C, H, W)`.

- `softmax(output, dim=1)`: converts raw class scores into probabilities that sum to 1.

- `torch.max(probs, dim=1)`: returns both the highest probability value (confidence) and the index of that class (predicted label).

---

Then:

```python
color = 'green' if pred_label == true_label else 'red'
```

- Green title = correct prediction.
- Red title = wrong prediction.

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

- the de-normalized image
- the true class name
- the predicted class name
- the confidence percentage
- green or red to indicate correctness

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

Without this, training could fail because of wrong paths, missing folders, or bad assumptions about the data.

### Step 2: Training

Notebook 2 learns from the train split and uses the validation split to decide when to stop.

This creates the trained checkpoint at `models/best_model.pth`.

### Step 3: Evaluation

Notebook 3 loads the saved model and measures final performance on the unseen test split.

This produces the final academic metrics and visualizations.

---

## What You Should Be Able to Explain to Your Instructor

After studying these notebooks, you should be able to explain:

- why the dataset must be checked before training
- why images are resized to `224x224`
- why augmentation is used only for training, not validation or testing
- what transfer learning means and why we froze the backbone
- why ResNet18 was selected
- what validation loss is used for
- why early stopping helps avoid overfitting
- what `torch.set_grad_enabled` does and why we disable gradients during validation
- how the best model checkpoint is saved and what it contains
- why test data must remain unseen during training
- what accuracy, precision, recall, F1-score, and confusion matrix mean
- why `classification_report(output_dict=True)` is used and what the dictionary contains

If you can explain those clearly, you will already understand the codebase at a strong level.
