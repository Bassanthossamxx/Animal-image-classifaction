# Notebooks Explained — Study Guide

Think of this document as your TA sitting next to you, walking through every line of code before your viva or discussion. Nothing is skipped. Every "why" is answered.

---

## The Big Picture First

Before diving into code, understand what the project is doing at a high level.

You have a folder of animal images. You want a computer program that looks at a new image and says "that is a dog" or "that is a lion." That is **image classification**.

To build this, the project follows three steps — one notebook per step:

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `1_data_exploration.ipynb` | Look at the data before touching it |
| 2 | `2_model_training.ipynb` | Teach the model using the data |
| 3 | `3_evaluation.ipynb` | Test the model on images it has never seen |

This order matters. You always check your data first, then train, then test. Skipping step 1 is like cooking without checking if you have all the ingredients.

---

## The Dataset Structure

The dataset is organized like this on disk:

```
data/animals_dataset/
├── train/          ← images the model learns from
│   ├── cat/
│   ├── cow/
│   ├── deer/
│   ├── dog/
│   └── lion/
├── val/            ← images used to check progress during training
│   ├── cat/ ...
└── test/           ← images kept hidden until final evaluation
    ├── cat/ ...
```

**Why three separate folders?**

Think of it like studying for an exam:
- `train` = your textbook and practice problems. You study from these.
- `val` = mock exams you take while studying to see how you are doing.
- `test` = the real exam. You only take it once, at the very end.

If the model ever sees `test` images during training, the final score is meaningless — it is like memorizing the real exam answers in advance.

---

---

# Notebook 1 — `1_data_exploration.ipynb`

## What This Notebook Is For

Before training, you need to confirm that the data is actually there and looks correct.

This notebook asks:
- Are the folders where we expect them?
- Are all five class folders present?
- How many images are in each class and split?
- Can the images actually be opened?
- What sizes and formats are the images?

**Why does this matter?** If any of those checks fail, training will either crash or silently produce wrong results. Discovering a problem here — before training — saves hours of wasted time.

---

## Cell 1: Imports and Paths

### The imports

```python
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
```

Go through each one:

**`import os`**
This gives access to operating system features like file paths and directory operations. It is a Python standard library — no installation needed. In this notebook it is not used heavily, but it is a common import in data scripts.

**`from collections import Counter`**
`Counter` is a special dictionary that counts things for you. If you give it a list like `['JPEG', 'JPEG', 'PNG', 'JPEG']`, it tells you `{'JPEG': 3, 'PNG': 1}`. Here it is used to count how many images have each size and format.

**`from pathlib import Path`**
This is the modern Python way to work with file paths. Instead of joining strings like `"../data" + "/" + "train"`, you write `Path('../data') / 'train'`. It handles Windows backslashes and Linux forward slashes automatically. Much safer than raw string paths.

**`import matplotlib.pyplot as plt`**
Matplotlib is the standard Python plotting library. `pyplot` is its main interface. `plt` is just a short alias so we do not have to type `matplotlib.pyplot` every time.

**`import pandas as pd`**
Pandas is used for data tables (called DataFrames). Here it displays the image count table in a nicely formatted grid inside Jupyter.

**`from PIL import Image`**
PIL stands for Python Imaging Library. The modern version is called Pillow. It opens image files and lets you inspect them — size, format, pixel data.

---

### The paths

```python
DATA_DIR = Path('../data/animals_dataset')
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
```

**`Path('../data/animals_dataset')`**
The `..` means "go one folder up." Since the notebooks are inside `notebooks/`, going up one level gets to the project root, and then we go into `data/animals_dataset/`.

**`DATA_DIR / 'train'`**
The `/` operator on Path objects joins folders together. This is not division — it is path joining. Result: `../data/animals_dataset/train`.

**`RESULTS_DIR.mkdir(exist_ok=True)`**
Creates the `results/` folder if it does not already exist. `exist_ok=True` means "if the folder already exists, do nothing — do not crash."

**`IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}`**
A Python `set` of valid image extensions. Sets use `{}` and do not allow duplicates. Using `in IMAGE_EXTENSIONS` to check a file extension is fast and clean.

> **TA might ask:** Why use a set instead of a list for extensions?
> **Answer:** Because checking `x in set` is faster than `x in list`. Also, a set makes it obvious these are unique values with no order.

---

### The path validation

```python
assert DATA_DIR.exists(), f'Data not found at {DATA_DIR.resolve()}'
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    assert split_dir.exists(), f'Missing split directory: {split_dir}'

classes = sorted([item.name for item in TRAIN_DIR.iterdir() if item.is_dir()])
print(f'Found {len(classes)} classes: {classes}')
```

**`assert DATA_DIR.exists(), "message"`**
`assert` checks if something is True. If it is False, Python immediately stops and prints the message. This is used as a quick sanity check — "if the data is not here, stop everything and tell me immediately."

> **TA might ask:** Why use `assert` and not `if/else`?
> **Answer:** `assert` is a fast way to say "this must be true or something is deeply wrong." It is for programmer-level safety checks, not user-facing error handling.

**`TRAIN_DIR.iterdir()`**
Lists every item (files and folders) inside a directory. Returns an iterator, not a list.

**`item.is_dir()`**
Returns `True` if the item is a folder. This filters out any stray files — we only want the class folders like `cat/`, `dog/`.

**`item.name`**
Gets just the final part of the path. So `../data/animals_dataset/train/cat` becomes just `cat`.

**`sorted([...])`**
Puts the class names in alphabetical order. This guarantees the same class order every time you run the notebook, regardless of how the operating system lists folders.

**Why class order matters so much:** The model assigns numbers to classes. `cat=0, cow=1, deer=2, dog=3, lion=4`. If the order ever changes between notebooks, the numbers change, and predictions become wrong. Sorting locks in the order.

---

## Cell 2: Counting Images

### The `image_files` function

```python
def image_files(folder):
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
```

This function answers: "give me all valid image files in this folder, in alphabetical order."

Breaking it down:

**`f for f in folder.iterdir()`**
This is a **generator expression** — a compact way to loop through items. Think of it as a loop that produces values one at a time instead of building a whole list in memory first.

**`f.suffix.lower()`**
`.suffix` gets the file extension including the dot, like `.jpg` or `.JPG`. `.lower()` converts to lowercase so `.JPG` and `.jpg` both match. Without `.lower()`, images with uppercase extensions would be silently ignored.

**`in IMAGE_EXTENSIONS`**
Checks if the extension is one of the valid ones we defined earlier.

**`sorted(...)`**
Wraps the generator and returns a sorted list of matching file paths.

> **TA might ask:** What does this function return for a folder with no images?
> **Answer:** An empty list `[]`. The `sorted()` of an empty generator is `[]`.

---

### The `count_images` function

```python
def count_images(split_dir):
    return {d.name: len(image_files(d)) for d in sorted(split_dir.iterdir()) if d.is_dir()}
```

This is a **dictionary comprehension** — Python's compact way to build a dictionary in one line.

To understand it, read it in plain English: "For every folder `d` inside `split_dir` (sorted alphabetically), if `d` is a directory, create a key-value pair where the key is the folder name and the value is the count of image files inside."

Result example:
```python
{'cat': 91, 'cow': 86, 'deer': 87, 'dog': 111, 'lion': 89}
```

> **TA might ask:** What is a dictionary comprehension and how is it different from a regular loop?
> **Answer:** A dictionary comprehension is a shorthand that builds a dictionary in one expression. The equivalent loop version would be:
> ```python
> result = {}
> for d in sorted(split_dir.iterdir()):
>     if d.is_dir():
>         result[d.name] = len(image_files(d))
> return result
> ```
> The comprehension is more concise and readable when the logic is simple.

---

### Applying the functions and displaying the table

```python
train_counts = count_images(TRAIN_DIR)
val_counts   = count_images(VAL_DIR)
test_counts  = count_images(TEST_DIR)

summary_df = pd.DataFrame([
    {'split': 'train', **train_counts},
    {'split': 'val',   **val_counts},
    {'split': 'test',  **test_counts},
]).set_index('split')

display(summary_df)
print(f"\nTrain: {sum(train_counts.values())}  Val: {sum(val_counts.values())}  Test: {sum(test_counts.values())}")
```

**`**train_counts`**
The `**` operator "unpacks" a dictionary into key-value pairs. So `{'split': 'train', **train_counts}` becomes `{'split': 'train', 'cat': 91, 'cow': 86, ...}`. This is how we build one row of the table per split.

**`pd.DataFrame([...])`**
Creates a table. Each dictionary in the list becomes one row.

**`.set_index('split')`**
Makes `split` the row label instead of a regular column. This makes the table look cleaner — the split names become row headers.

**`display(summary_df)`**
Jupyter's `display()` function renders the DataFrame as a formatted HTML table in the notebook output, much nicer than just printing it.

**`sum(train_counts.values())`**
`.values()` returns all the counts in the dictionary. `sum()` adds them up to get the total images in that split.

**Expected output from this cell:**

```
       cat  cow  deer  dog  lion
split
train   91   86    87  111    89
val     17   15    17   17    17
test    17   16    16   16    17

Train: 464  Val: 83  Test: 82
```

---

## Cell 3: Class Distribution Bar Charts

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
```

**`plt.subplots(1, 3, ...)`**
Creates one figure with 1 row and 3 columns of plots. Returns:
- `fig`: the overall figure object
- `axes`: an array of 3 subplot axes objects

**`figsize=(15, 4)`**
Sets the figure size in inches: 15 wide, 4 tall.

**`sharey=True`**
All three subplots share the same y-axis scale. This is important for comparison — if each had its own scale, you could not visually compare bar heights across splits.

---

```python
for idx, (split_name, counts) in enumerate([
    ('Train', train_counts),
    ('Validation', val_counts),
    ('Test', test_counts),
]):
```

**`enumerate([...])`**
When you iterate over a list normally you just get the values. `enumerate` also gives you the position (index). Here `idx` is 0, 1, or 2 and tells us which subplot to use.

**`(split_name, counts)`**
This is called **tuple unpacking**. Each item in the list is a tuple like `('Train', train_counts)`. Python automatically assigns `split_name = 'Train'` and `counts = train_counts`.

---

```python
axes[idx].bar(counts.keys(), counts.values(), color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
axes[idx].set_title(f'{split_name} Set')
axes[idx].set_xlabel('Class')
axes[idx].tick_params(axis='x', rotation=45)
```

**`axes[idx].bar(...)`**
Draws a bar chart in the `idx`-th subplot.
- `counts.keys()` → class names for the x-axis labels
- `counts.values()` → image counts for bar heights
- `color=[...]` → one hex color per bar

**`tick_params(axis='x', rotation=45)`**
Rotates the x-axis class name labels by 45 degrees. Without this they would overlap each other and be unreadable.

```python
axes[0].set_ylabel('Image Count')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

**`axes[0].set_ylabel(...)`**
Only the first subplot needs a y-axis label since all three share the same scale.

**`plt.tight_layout()`**
Automatically adjusts spacing between subplots so titles and labels do not overlap.

**`dpi=300`**
DPI = dots per inch. 300 is high resolution — good for a printed report.

**`bbox_inches='tight'`**
Trims whitespace around the figure before saving.

---

## Cell 4: Displaying Sample Images

```python
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
fig.suptitle('Sample Images from Training Set', fontsize=16)
```

With 5 classes, this creates a 5×5 grid. Each row is one animal class, each column is one sample image.

**`figsize=(15, 3 * len(classes))`**
Height scales with the number of classes. With 5 classes: `3 * 5 = 15` inches tall.

**`suptitle`**
"Super title" — one title for the whole figure, not just one subplot.

---

```python
if len(classes) == 1:
    axes = [axes]
```

This is a safety check. When you have only one row, matplotlib returns `axes` as a 1D array. When you have multiple rows, it returns a 2D array. This line forces a consistent 2D structure so the indexing `axes[i][j]` always works. In this project with 5 classes this line never triggers, but it is good defensive coding.

---

```python
for i, class_name in enumerate(classes):
    samples = image_files(TRAIN_DIR / class_name)[:5]
    for j in range(5):
        ax = axes[i][j] if len(classes) > 1 else axes[j]
        ax.axis('off')
        if j < len(samples):
            with Image.open(samples[j]) as img:
                ax.imshow(img.convert('RGB'))
        if j == 0:
            ax.set_title(class_name.upper(), fontsize=12, fontweight='bold')
```

**`image_files(TRAIN_DIR / class_name)[:5]`**
Gets all image files for that class, then `[:5]` takes the first five (list slicing).

**`ax.axis('off')`**
Hides the axis lines, tick marks, and labels. We want a clean image display, not a coordinate grid.

**`if j < len(samples)`**
Guards against a class that has fewer than 5 images. Without this, `samples[j]` would crash with an index error.

**`with Image.open(samples[j]) as img:`**
The `with` statement ensures the file is closed automatically after the block, even if an error occurs. Good practice when opening files.

**`img.convert('RGB')`**
Some images might be in grayscale, RGBA (with transparency), or other formats. Converting to RGB ensures matplotlib can always display them correctly.

**`if j == 0: ax.set_title(...)`**
Only add the class name label to the first image in each row. The other four images in the row do not need a label — it would clutter the display.

---

## Cell 5: Image Sizes and Formats

```python
size_counter = Counter()
format_counter = Counter()

for class_name in classes:
    for image_path in image_files(TRAIN_DIR / class_name)[:15]:
        with Image.open(image_path) as img:
            size_counter[img.size] += 1
            format_counter[img.format] += 1
```

**`Counter()`**
Starts as an empty counter-dictionary. When you do `counter[key] += 1`, if the key does not exist yet, `Counter` automatically treats it as 0 first. You do not need to initialize each key manually.

**`img.size`**
Returns a tuple like `(275, 183)` — width and height in pixels.

**`img.format`**
Returns the file format string, like `'JPEG'` or `'PNG'`.

**Why sample 15 images per class instead of all?**
Checking all images would take more time for no meaningful benefit at this stage. Sampling 15 per class (75 total) gives a reliable picture of what formats and sizes the dataset contains.

```python
for size, count in size_counter.most_common(10):
    print(f'  {size}: {count}')
```

**`.most_common(10)`**
Returns the 10 most frequent items, sorted from most to least common. This is a built-in `Counter` method.

**What this output tells you:**
The original images are all different sizes — `(275, 183)`, `(225, 225)`, etc. None of them are already `224x224`. This is why every preprocessing pipeline in this project includes `transforms.Resize((224, 224))`. Without resizing, the model cannot process images with different dimensions in the same batch.

---

---

# Notebook 2 — `2_model_training.ipynb`

## What This Notebook Is For

This is the most important notebook. It takes the dataset, feeds it to a neural network, and adjusts the network's internal parameters until it can correctly identify animals.

The key idea here is **transfer learning**. Instead of building a neural network from scratch and training it on our 464 images (which is very small for deep learning), we borrow a network that was already trained on millions of images from the internet. We then adapt the last layer of that network to recognize our 5 specific animals.

Think of it like this: you hire an experienced art critic who already knows how to look at shapes, textures, and colors in images. You just need to teach them the specific difference between a cat, cow, deer, dog, and lion — which takes much less effort than training someone from scratch to understand visual art.

---

## Cell 1: Setup

### Imports

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

**`import time`**
Python's standard time library. Used here to record when training starts and calculate total duration.

**`torch`**
PyTorch — the deep learning framework the entire project is built on.

**`torch.nn as nn`**
"Neural network" module. Contains building blocks like layers (`nn.Linear`) and loss functions (`nn.CrossEntropyLoss`).

**`torch.optim as optim`**
Optimization algorithms. The optimizer is what actually adjusts the model's weights during training.

**`DataLoader`**
A utility that feeds data to the model in batches. Without it, you would have to manually slice your dataset into batches.

**`datasets`**
Contains `ImageFolder`, a convenient class that reads a folder-structured image dataset and assigns labels automatically.

**`models`**
Contains pretrained architectures. We use `models.resnet18`.

**`transforms`**
Contains image preprocessing operations like resizing, flipping, and normalizing.

---

### Hyperparameters

```python
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 0
```

> **TA might ask:** What is a hyperparameter?
> **Answer:** A hyperparameter is a setting you choose before training that controls how training happens. It is different from a model parameter (like weights), which the model learns automatically during training. Examples: batch size, learning rate, number of epochs.

**`BATCH_SIZE = 16`**
Instead of showing the model all 464 images at once, we show 16 at a time. After each batch, weights are updated. Smaller batches = more updates per epoch but noisier gradients. Larger batches = smoother gradients but need more memory.

**`LEARNING_RATE = 1e-3`**
`1e-3` is scientific notation for `0.001`. The learning rate controls how big each weight update step is. Too high → training is unstable. Too low → training is very slow. 0.001 is a safe starting value for Adam optimizer.

**`NUM_EPOCHS = 20`**
One epoch = one full pass through the entire training dataset. 20 epochs means the model sees every training image up to 20 times.

**`PATIENCE = 5`**
If validation loss does not improve for 5 consecutive epochs, training stops early. This is the "patience" for early stopping.

**`NUM_WORKERS = 0`**
This controls how many background threads load images in parallel. 0 means the main Python process does it directly. On Windows and in Jupyter, values higher than 0 can cause errors, so 0 is the safe choice.

---

### Device selection

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

**`torch.cuda.is_available()`**
Returns `True` if a CUDA-capable GPU is available, `False` otherwise.

**`torch.device('cuda')`**
CUDA is NVIDIA's framework for GPU computation. Using a GPU can make training 10–50x faster than CPU for deep learning tasks.

**Why this line matters:** Moving tensors and models to the GPU requires explicitly specifying the device. This line makes the code portable — it uses GPU when available, CPU otherwise, without changing any other code.

---

## Cell 2: Transforms and Data Loading

### Why we need transforms at all

Raw images from disk are PIL objects with pixels in 0–255 range. Neural networks need:
1. A fixed input size (all images must be the same dimensions)
2. Values in a specific numerical range (not 0–255)
3. The tensor format (not PIL Image objects)
4. Statistics matching the pretrained model's training data (ImageNet normalization)

Transforms handle all of this.

---

### Training transform

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

**`transforms.Compose([...])`**
Chains transforms together. An image enters, passes through each transform in order, and comes out the other end ready for the model. Like an assembly line.

**`Resize((224, 224))`**
ResNet18 expects 224×224 pixel images. This step makes every image that exact size, regardless of original dimensions.

**`RandomHorizontalFlip(p=0.5)`**
Flips images left-to-right with 50% probability. A cat facing left is still a cat. This teaches the model not to rely on which direction an animal faces.

**`RandomRotation(15)`**
Randomly rotates images up to ±15 degrees. Teaches the model that animals at an angle are still the same animal.

**`ColorJitter(...)`**
Randomly tweaks brightness, contrast, and saturation. Teaches the model to recognize animals under different lighting conditions (sunny, dark, overcast).

**Why do we augment only training images?**
Augmentation artificially creates variety so the model generalizes better. But we never augment validation or test images because those need to represent real conditions fairly and consistently.

**`ToTensor()`**
Converts the PIL Image to a PyTorch tensor. Also scales pixel values from 0–255 to 0.0–1.0.

**`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`**
These specific numbers are the mean and standard deviation of the ImageNet dataset (the dataset ResNet18 was originally trained on). Normalizing with these values shifts pixel values so they match what the pretrained model expects. Think of it like translating to the same "language" the model already learned in.

> **TA might ask:** Why these exact numbers and not something else?
> **Answer:** These are the ImageNet statistics. Since ResNet18 was pretrained on ImageNet using these exact normalization values, we must use the same values during our training and inference. If we used different values, the pretrained features would not work correctly.

---

### Validation transform

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

No random augmentation — only resize, convert, normalize. Validation must give a fair and repeatable score. Randomly flipping validation images would make scores vary each run for no reason.

---

### Loading datasets

```python
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
```

**`datasets.ImageFolder`**
This is one of PyTorch's most useful utilities. Given a folder like:

```
train/
  cat/  image1.jpg, image2.jpg, ...
  cow/  image1.jpg, ...
  ...
```

It automatically:
- Finds all images in all subfolders
- Assigns numeric labels based on alphabetical subfolder order: cat=0, cow=1, deer=2, dog=3, lion=4
- Applies the specified transform to every image when it is loaded

---

### DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

A `DataLoader` wraps a dataset and delivers it in batches.

**`shuffle=True` for training:** Every epoch, the images are delivered in a different random order. This prevents the model from memorizing the sequence of images instead of learning the actual visual features.

**`shuffle=False` for validation:** Order does not matter for measuring accuracy. Keeping it fixed makes debugging easier.

---

## Cell 3: Model Setup

### Loading pretrained ResNet18

```python
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

**What is ResNet18?**
ResNet18 is a convolutional neural network with 18 layers, designed to classify images. It was trained on ImageNet — a dataset of 1.2 million images across 1000 categories. It already knows how to detect edges, shapes, textures, and complex visual patterns.

**`DEFAULT`**
Loads the best available pretrained weights. PyTorch downloads them automatically if not already cached.

---

### Freezing all layers

```python
for param in model.parameters():
    param.requires_grad = False
```

**`model.parameters()`**
Returns all the learnable values (weights and biases) in the model.

**`param.requires_grad = False`**
"Requires grad" means "should this parameter be updated during training?" Setting it to `False` freezes the parameter.

**Why freeze the backbone?**
ResNet18 has learned incredible visual features from 1.2 million images. With only 464 training images, if we tried to retrain all its layers, we would destroy those features (this is called "catastrophic forgetting"). By freezing the backbone, we keep all that knowledge and only teach the final layer the animal-specific differences.

> **TA might ask:** What is the difference between fine-tuning and feature extraction?
> **Answer:** Feature extraction (what we do here) = freeze the backbone, only train the last layer. Fine-tuning = unfreeze some or all layers and train them at a lower learning rate. Feature extraction is better when your dataset is small.

---

### Replacing the final layer

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
```

**`model.fc`**
`fc` stands for "fully connected" — the final classification layer. The original ResNet18 has `fc` as `Linear(512, 1000)` — 512 input features, 1000 output classes (one per ImageNet class).

**`nn.Linear(model.fc.in_features, num_classes)`**
Replaces it with `Linear(512, 5)`. Now the model outputs 5 scores — one per animal class.

**`model.fc.in_features`**
Reads the size of the layer's input (512) without hardcoding that number. If we ever changed the backbone to a different model, this would still work.

**`model.to(device)`**
Moves all model parameters to the device (GPU or CPU). This must happen before training.

---

### Loss, optimizer, scheduler

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

**`nn.CrossEntropyLoss()`**
The standard loss function for multi-class classification. It combines softmax (converting raw scores to probabilities) and negative log-likelihood loss into one step. Higher loss = worse predictions. The goal of training is to minimize this.

**`optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)`**
Adam is an optimizer that adjusts the learning rate automatically per parameter. `model.fc.parameters()` means "only optimize the final layer" — the frozen backbone layers are excluded because their `requires_grad=False`.

**`ReduceLROnPlateau(...)`**
This scheduler watches a metric (validation loss) and reduces the learning rate when it stops improving.
- `mode='min'` → we want loss to go down
- `factor=0.5` → multiply the learning rate by 0.5 (halve it)
- `patience=2` → wait 2 epochs of no improvement before reducing

**Why reduce the learning rate?**
When training is going well, a higher learning rate helps reach the minimum fast. When improvement slows down, a smaller learning rate helps the model take smaller steps to find a better minimum without overshooting it.

---

## Cell 4: The `run_epoch` Function

This is the heart of the training notebook. Read this section carefully.

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
```

This one function handles both training and validation. The trick: if `optimizer` is passed in, it trains. If not, it evaluates.

> **TA might ask:** Why write one function for both instead of two separate functions?
> **Answer:** The forward pass (running data through the model and computing loss) is identical for training and validation. The only difference is whether we also do a backward pass and weight update. One function with a flag avoids duplicating all that shared code.

---

```python
is_training = optimizer is not None
model.train() if is_training else model.eval()
```

**`is_training`**
A simple boolean: `True` if we passed an optimizer (training), `False` if we did not (validation).

**`model.train()`**
Puts the model in training mode. Some layers behave differently:
- **Dropout** is active (randomly zeros out some neurons to prevent overfitting)
- **BatchNorm** uses the current batch statistics

**`model.eval()`**
Puts the model in evaluation mode:
- **Dropout** is disabled (all neurons active)
- **BatchNorm** uses running statistics accumulated during training

**Always call the correct mode.** Forgetting `model.eval()` during validation is a common bug that makes validation metrics look artificially better (because dropout randomly removes neurons, it effectively averages many sub-networks, which can look like regularization).

---

```python
running_loss, correct, total = 0.0, 0, 0
```

Three variables on one line. Python allows multiple assignment separated by commas. These counters will accumulate values across all batches in the epoch.

---

```python
with torch.set_grad_enabled(is_training):
```

**What is a gradient and why do we care?**
A gradient is the direction and amount to change each weight to reduce the loss. Computing gradients takes memory and time. During validation, we do not update weights, so gradients are useless — we skip computing them.

**`torch.set_grad_enabled(is_training)`**
A context manager:
- When `is_training=True` → gradients are computed (normal PyTorch behavior)
- When `is_training=False` → gradients are not computed (same as wrapping in `torch.no_grad()`)

This is cleaner than the old way of writing:
```python
context = torch.enable_grad() if is_training else torch.no_grad()
with context: ...
```
`set_grad_enabled` does exactly the same thing in one readable line.

---

```python
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
```

**`for images, labels in loader`**
The DataLoader delivers batches. Each batch is a tuple of `(images_tensor, labels_tensor)`. Tuple unpacking assigns both at once.

**`.to(device)`**
Moves the tensors to the same device as the model. Model and input must always be on the same device. If the model is on GPU and the input is on CPU (or vice versa), PyTorch will crash with a device error.

---

```python
if is_training:
    optimizer.zero_grad()
```

**`optimizer.zero_grad()`**
PyTorch accumulates gradients by default — it adds new gradients onto old ones instead of replacing them. If you forget `zero_grad()`, every batch's gradients pile up and training goes completely wrong.

Think of it like erasing the whiteboard before writing a new calculation.

---

```python
outputs = model(images)
loss = criterion(outputs, labels)
```

**`model(images)`**
This is the **forward pass**. The images flow through all the layers of the network and produce raw scores (called "logits") — one score per class per image. Shape: `(batch_size, num_classes)` → `(16, 5)`.

**`criterion(outputs, labels)`**
Computes the loss by comparing the model's raw scores to the true class labels. CrossEntropyLoss internally applies softmax to convert scores to probabilities, then measures how wrong they are.

---

```python
if is_training:
    loss.backward()
    optimizer.step()
```

**`loss.backward()`**
Computes the gradient of the loss with respect to every trainable parameter. This is **backpropagation** — the key algorithm behind all neural network training. It uses the chain rule from calculus to propagate the error signal back through all layers.

**`optimizer.step()`**
Uses the computed gradients to update the trainable parameters (the weights of `model.fc`). Adam adjusts each weight using both the gradient and historical gradient information for a smarter update.

These two lines only run during training. During validation, we skip them because we are only measuring performance, not learning.

---

```python
running_loss += loss.item()
correct += (outputs.argmax(dim=1) == labels).sum().item()
total += labels.size(0)
```

**`loss.item()`**
Converts the PyTorch scalar tensor to a plain Python float. Required because you cannot do regular Python math with PyTorch tensors directly in accumulation.

**`outputs.argmax(dim=1)`**
`argmax` returns the index of the maximum value. `dim=1` means "find the max across the class dimension." So for each image, it picks the class with the highest score. Shape: `(batch_size,)` → `(16,)`.

**`== labels`**
Element-wise comparison. Returns a boolean tensor where `True` means prediction matched the true label.

**`.sum().item()`**
Counts the number of `True` values (correct predictions) and converts to Python int.

**`labels.size(0)`**
Returns the batch size (first dimension). Added to `total` to count all images seen.

---

```python
return running_loss / len(loader), 100.0 * correct / total
```

After all batches:

**`running_loss / len(loader)`**
Average loss per batch. Dividing by the number of batches normalizes for different dataset sizes.

**`100.0 * correct / total`**
Accuracy as a percentage. `correct / total` is a fraction (0.0 to 1.0), multiplying by 100 gives the percentage.

---

## Cell 5: The Training Loop

```python
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0
```

**`history`**
A dictionary of empty lists. After each epoch, the four metrics are appended. At the end of training, each list has one value per epoch — perfect for plotting.

**`float('inf')`**
Positive infinity. Any real number is smaller than infinity. This means the very first epoch will always update `best_val_loss`, which is what we want.

**`patience_counter`**
Starts at 0. Goes up by 1 each epoch with no improvement. Resets to 0 when improvement happens. When it reaches `PATIENCE`, training stops.

---

```python
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
```

Each epoch:
1. Run a full training pass → weights update
2. Run a full validation pass → no weight updates, just measurement

Notice `optimizer=optimizer` is passed to the training call and omitted from the validation call. This is how the single `run_epoch` function knows which mode to use.

---

```python
scheduler.step(val_loss)
```

Feeds the current validation loss to the scheduler. If `val_loss` has not improved for 2 epochs, the scheduler halves the learning rate automatically.

---

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'best_val_loss': best_val_loss,
        'history': history,
    }, MODEL_SAVE_PATH)
    print('  Saved best model checkpoint.')
else:
    patience_counter += 1
    print(f'  No improvement ({patience_counter}/{PATIENCE})')
```

**`model.state_dict()`**
Returns a dictionary of all model parameters (tensors). This is the standard way to save PyTorch models. When you want to load the model later, you call `model.load_state_dict(...)`.

**Why save `class_names` in the checkpoint?**
This is a very important design decision. The evaluation notebook and the app both need to know that class index 0 = "cat", 1 = "cow", etc. By saving `class_names` inside the checkpoint, they never need to re-discover this information — they just read it back from the file. This prevents label order mismatches.

**Why save only the best, not the last?**
The last epoch's model might have overfit slightly. The best validation loss checkpoint is the model that generalized best to unseen data, which is what we actually want to deploy.

---

```python
if patience_counter >= PATIENCE:
    print(f'Early stopping triggered at epoch {epoch + 1}.')
    break
```

**Early stopping** prevents the model from training past the point of useful learning. If validation loss has not improved for 5 epochs, continuing would likely just increase overfitting.

> **TA might ask:** What is overfitting?
> **Answer:** Overfitting means the model memorizes the training data instead of learning general patterns. It performs very well on training images but poorly on new images it has never seen. Early stopping and validation monitoring both help prevent this.

---

## Cell 6: Training Curves Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

Two side-by-side plots: loss on the left, accuracy on the right.

The loss plot should show:
- Both training and validation loss decreasing
- The two lines staying relatively close (not diverging)
- Divergence (training keeps falling while validation rises) = overfitting

The accuracy plot should show:
- Both lines increasing over epochs
- Validation accuracy reaching a high stable value

These plots are saved as `results/training_curves.png` for your report.

---

---

# Notebook 3 — `3_evaluation.ipynb`

## What This Notebook Is For

This is the final stage. The model has been trained. Now we measure how well it actually works on 82 images it has never seen before.

This notebook gives you the numbers for your academic report:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Visual sample predictions

---

## Cell 1: Imports and Setup

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Notice there is **no** `precision_recall_fscore_support` import. In this version, `classification_report` handles everything:
- Call it with `output_dict=True` → get a dictionary of all metrics
- Call it without `output_dict` → get a formatted text string for saving to a file

One function, two uses, no redundancy.

---

## Cell 2: Test Data and Model Loading

```python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Identical to the validation transform. No augmentation. Test evaluation must be clean and unbiased.

---

```python
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
```

**`map_location=device`**
If the model was saved on a GPU machine but you are loading on a CPU machine (or vice versa), `map_location` handles the translation automatically. Without this, loading would fail if devices do not match.

**`checkpoint['class_names']`**
Reads the class names that were saved during training. This ensures perfect consistency in the label-to-class mapping.

---

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```

**`weights=None`**
Do not download ImageNet weights — we are about to load our own trained weights immediately after.

**`model.load_state_dict(...)`**
Loads the saved parameter values into the model architecture. This restores the model to the exact state it was in when the best checkpoint was saved.

**`model.eval()`**
Very important. Switches to evaluation mode. If you forget this, layers like dropout will randomly drop neurons and your evaluation results will be wrong and unrepeatable.

---

## Cell 3: Prediction Loop and Metrics

```python
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
```

**Why collect all predictions first?**
The metric functions (accuracy_score, classification_report, etc.) work on complete arrays. We loop through batches to get individual predictions, collect them all, then compute metrics on the full set at once.

**`outputs.argmax(dim=1).cpu().numpy()`**
Chain of three operations:
1. `argmax(dim=1)` → pick the highest-scoring class for each image
2. `.cpu()` → move from GPU to CPU (numpy cannot handle GPU tensors)
3. `.numpy()` → convert to NumPy array for sklearn

**`extend(...)` vs `append(...)`**
`append` would add the entire batch array as one element. `extend` adds each element of the batch individually. We want a flat list of predictions, not a list of arrays.

---

```python
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
```

Converts the Python lists to NumPy arrays. sklearn metric functions require NumPy arrays, not plain Python lists.

---

```python
test_accuracy = accuracy_score(all_labels, all_predictions)
```

Overall accuracy = number of correct predictions ÷ total number of test images.

---

```python
report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
```

With `output_dict=True`, this returns a dictionary structured like:

```python
{
    'cat':       {'precision': 1.0,    'recall': 0.8824, 'f1-score': 0.9375, 'support': 17},
    'cow':       {'precision': 1.0,    'recall': 1.0,    'f1-score': 1.0,    'support': 16},
    'deer':      {'precision': 0.8889, 'recall': 1.0,    'f1-score': 0.9412, 'support': 16},
    'dog':       {'precision': 1.0,    'recall': 1.0,    'f1-score': 1.0,    'support': 16},
    'lion':      {'precision': 1.0,    'recall': 1.0,    'f1-score': 1.0,    'support': 17},
    'macro avg': {'precision': 0.9778, 'recall': 0.9765, 'f1-score': 0.9757, 'support': 82},
    ...
}
```

---

### Understanding the metrics

> **TA might ask:** What is the difference between precision and recall?

**Precision** answers: "Of all images the model *predicted* as cat, how many were actually cats?"
- High precision = when the model says cat, it is usually right
- Formula: True Positives ÷ (True Positives + False Positives)

**Recall** answers: "Of all the actual cats in the test set, how many did the model *find*?"
- High recall = the model finds most of the real cats
- Formula: True Positives ÷ (True Positives + False Negatives)

**F1-score** is the harmonic mean of precision and recall:
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Useful when you want a single number that balances both
- A score of 1.0 is perfect, 0.0 is worst

**Support** = how many test images exist for that class (the ground truth count).

**Macro average** = compute each metric per class, then take the simple unweighted average. Each class contributes equally, regardless of how many images it has.

---

```python
print(f'Overall test accuracy: {test_accuracy * 100:.2f}%')
for class_name in class_names:
    m = report_dict[class_name]
    print(f"{class_name:<5} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1-score']:.4f} | Support: {int(m['support'])}")
macro = report_dict['macro avg']
print(f"Macro averages | Precision: {macro['precision']:.4f} | Recall: {macro['recall']:.4f} | F1: {macro['f1-score']:.4f}")
```

**`{class_name:<5}`**
The `<5` part means "left-align this text in a field of width 5 characters." This pads shorter names with spaces so all the `|` separators line up in a neat column.

**`:.4f`**
Format as a decimal with exactly 4 digits after the decimal point.

**`int(m['support'])`**
`support` in the dictionary is stored as a float (e.g., `17.0`). `int(...)` converts it to a clean integer for display.

---

## Cell 4: Saving the Classification Report

```python
report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
```

This time called **without** `output_dict=True`. Returns a nicely formatted text block like:

```
              precision    recall  f1-score   support

         cat     1.0000    0.8824    0.9375        17
         cow     1.0000    1.0000    1.0000        16
        ...
```

```python
with report_path.open('w', encoding='utf-8') as file:
    file.write('ANIMAL CLASSIFICATION - TEST SET RESULTS\n')
    file.write('=' * 60 + '\n\n')
    file.write(f'Overall Test Accuracy: {test_accuracy * 100:.2f}%\n\n')
    file.write(report)
```

**`report_path.open('w', encoding='utf-8')`**
Opens the file in write mode (`'w'`). `utf-8` ensures correct character encoding across different operating systems.

**`'=' * 60`**
Repeats the `=` character 60 times to create a divider line. Python allows multiplying strings.

This file (`results/classification_report.txt`) is a permanent record of your final model performance.

---

## Cell 5: Confusion Matrix

```python
cm = confusion_matrix(all_labels, all_predictions)
```

The confusion matrix is a 5×5 grid. Row = true class, column = predicted class.

Example reading: If `cm[0][2] = 2`, it means 2 images that were truly "cat" (row 0) were predicted as "deer" (column 2).

Perfect predictions → all values on the diagonal, zeros everywhere else.

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
```

**`sns.heatmap`**
Seaborn's heatmap function visualizes the matrix as a color grid. Darker blue = higher count.

**`annot=True`**
Writes the actual number inside each cell.

**`fmt='d'`**
Formats annotations as integers (not floats like `17.0`).

> **TA might ask:** What does the confusion matrix tell you that accuracy alone does not?
> **Answer:** Accuracy is one number. The confusion matrix shows *which* classes are being confused with each other. For example, our model slightly confuses cats (some cats were predicted as another class), but never confuses cows with dogs. That level of detail cannot come from accuracy alone.

---

## Cell 6: Sample Predictions with Images

```python
indices = np.random.choice(len(test_dataset), size=min(9, len(test_dataset)), replace=False)
```

**`np.random.choice(..., replace=False)`**
Randomly selects 9 indices from the test dataset without replacement (same image cannot be selected twice).

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
```

The model receives normalized images. But normalized images look wrong to human eyes (colors are shifted). To display them, we must **undo** the normalization.

**`view(3, 1, 1)`**
Reshapes the 1D tensor `[0.485, 0.456, 0.406]` into shape `(3, 1, 1)`. This allows it to broadcast (apply separately) across the 3 color channels of an image shaped `(3, 224, 224)`.

---

```python
display_img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
```

Breaking this chain down:

**`img * std + mean`**
Reverses the `Normalize` transform. If normalization did `(x - mean) / std`, then the inverse is `x * std + mean`.

**`.clamp(0, 1)`**
Due to floating point rounding, some values may go slightly outside 0–1 after denormalization. Clamping keeps them in the valid range for display.

**`.permute(1, 2, 0)`**
PyTorch tensors use channel-first format: `(C, H, W)` = `(3, 224, 224)`. Matplotlib expects channel-last: `(H, W, C)` = `(224, 224, 3)`. `permute` reorders the dimensions.

**`.numpy()`**
Converts the PyTorch tensor to a NumPy array. Matplotlib requires NumPy arrays for `imshow`.

---

```python
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    probs = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(probs, dim=1)
```

**`img.unsqueeze(0)`**
`img` has shape `(3, 224, 224)`. The model expects batches: `(batch_size, 3, 224, 224)`. `unsqueeze(0)` inserts a new dimension at position 0, making it `(1, 3, 224, 224)` — a batch of one image.

**`torch.softmax(output, dim=1)`**
Converts raw scores to probabilities that all sum to 1.0 across the class dimension. For example: `[cat: 0.02, cow: 0.01, deer: 0.03, dog: 0.90, lion: 0.04]`.

**`torch.max(probs, dim=1)`**
Returns two tensors:
- `confidence` = the highest probability value (e.g., `0.90`)
- `pred_label` = the index of that class (e.g., `3` for dog)

---

```python
color = 'green' if pred_label == true_label else 'red'
```

Green title = model was correct. Red title = model made a mistake. This gives an instant visual of which predictions are wrong, and lets you look at the actual image to understand why.

---

---

## How the App (`app/app.py`) Connects

The Gradio app uses the same model and preprocessing as the evaluation notebook, loaded from the same checkpoint file.

```python
def load_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names
```

This is identical to what the evaluation notebook does. Same model architecture, same weights, same class names from the checkpoint. This guarantees that what you measured in evaluation is what the app actually does.

The app uses the same `Normalize` transform as training, so images uploaded by a user are preprocessed identically to how the model was trained.

---

## What You Should Be Able to Explain to Your TA

Go through this list before your viva. If you can answer all of these clearly, you understand the project at a strong level.

**About the data:**
- Why do we split into train, val, and test?
- Why is the test set kept hidden during training?
- Why are all images resized to 224×224?

**About preprocessing:**
- What does `transforms.Normalize` do and why those specific numbers?
- Why does training use random augmentation but validation does not?

**About transfer learning:**
- What is transfer learning and why is it used here?
- What does freezing the backbone mean?
- What is the difference between feature extraction and fine-tuning?

**About training:**
- What does `optimizer.zero_grad()` do and why is it necessary?
- What is a forward pass? What is a backward pass?
- What does `loss.backward()` compute?
- What is `torch.set_grad_enabled` and why do we disable gradients during validation?
- What is early stopping and why does it help?
- Why is the best checkpoint saved based on validation loss, not training loss?
- What is saved inside the checkpoint and why is `class_names` included?

**About evaluation:**
- What is the difference between precision and recall?
- What does the F1-score represent?
- What does a confusion matrix show that accuracy does not?
- What is the macro average?
- Why do we call `model.eval()` before evaluation?

**About the code patterns:**
- What is a dictionary comprehension?
- What does `enumerate` do?
- What does `unsqueeze(0)` do to a tensor?
- What does `permute(1, 2, 0)` do and why is it needed for display?
- What does `classification_report(output_dict=True)` return?
