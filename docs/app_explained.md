# App Explanation

This guide explains how `app/app.py` works from top to bottom.

The goal of the app is simple:

- load the trained model
- let the user upload an image
- preprocess the image
- run prediction
- show the predicted class and class probabilities

This file is written so you can study both the programming flow and the machine learning logic behind the app.

---

## 1. High-Level Idea

The app uses Gradio, which is a Python library for creating simple web interfaces.

Instead of building HTML, CSS, JavaScript, and backend routes manually, Gradio lets us define:

- inputs
- outputs
- the prediction function

Then it automatically creates the web UI.

In this project, the app does the following:

1. Finds the project root and important paths
2. Loads the saved PyTorch model checkpoint
3. Defines the same image preprocessing used during evaluation
4. Defines a function called when the user clicks the prediction button
5. Builds the web interface layout
6. Launches the local web server

---

## 2. Imports

```python
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
```

### What each import does

- `Path`
  - Handles filesystem paths.

- `gradio as gr`
  - Provides UI components like image upload, button, textbox, and labels.

- `torch`
  - Main PyTorch library.

- `torch.nn as nn`
  - Used for neural network layers, especially the final classification layer.

- `Image` from PIL
  - Helps ensure uploaded images are handled correctly as images.

- `models, transforms`
  - `models` gives access to ResNet18.
  - `transforms` handles preprocessing before prediction.

---

## 3. Path Setup

```python
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
DATASET_DIR = ROOT_DIR / "data" / "animals_dataset" / "test"
DEFAULT_CLASS_NAMES = ["cat", "cow", "deer", "dog", "lion"]
```

### What each line does

- `__file__`
  - Refers to the current file, `app.py`.

- `.resolve()`
  - Converts it to an absolute path.

- `.parents[1]`
  - Moves up two levels:
    - from `app/app.py`
    - to the project root

- `MODEL_PATH`
  - Points to the trained model checkpoint.

- `DATASET_DIR`
  - Points to the test split.
  - This is used only to automatically collect example images for the interface.

- `DEFAULT_CLASS_NAMES`
  - Provides a fallback class order if the checkpoint is missing class names.

This path setup makes the app runnable from the project root without hardcoded absolute paths.

---

## 4. Device Selection

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### What this means

- If a CUDA-capable GPU is available, prediction uses the GPU.
- Otherwise it uses the CPU.

This is the same logic used in the notebooks.

---

## 5. `build_model` Function

```python
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### Why this function exists

The saved checkpoint contains the trained weights, but before loading them, the app must recreate the same model architecture.

### Line by line

- `models.resnet18(weights=None)`
  - Creates a ResNet18 model without downloading pretrained weights.
  - We do not need pretrained weights now because we are loading our own trained checkpoint.

- `model.fc = nn.Linear(model.fc.in_features, num_classes)`
  - Replaces the final classification layer with one matching the project’s class count.

- `return model`
  - Returns the configured model object.

This function makes model creation reusable and easy to read.

---

## 6. `load_checkpoint` Function

```python
def load_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names
```

### Purpose

This function loads the saved trained model and returns:

- the model ready for prediction
- the class names in the correct label order

### Line by line

- `torch.load(MODEL_PATH, map_location=device)`
  - Reads the checkpoint file from disk.
  - `map_location=device` ensures it loads correctly whether it was trained on GPU or CPU.

- `checkpoint.get("class_names", DEFAULT_CLASS_NAMES)`
  - Gets the saved class names if present.
  - If not, it falls back to the default class list.

- `build_model(len(class_names))`
  - Recreates the model with the correct number of output classes.

- `model.load_state_dict(checkpoint["model_state_dict"])`
  - Loads the trained weights into the model.

- `model.to(device)`
  - Moves the model to the selected device.

- `model.eval()`
  - Switches the model into inference mode.
  - This is essential for stable prediction behavior.

- `return model, class_names`
  - Gives both objects back to the caller.

---

## 7. Model Loading Safety Logic

```python
model = None
class_names = DEFAULT_CLASS_NAMES
model_load_error = None
```

### Why this exists

The app should not crash immediately if the trained model file is missing or broken.

Instead, it should:

- start safely
- show a message in the UI
- tell the user what is missing

---

```python
if MODEL_PATH.exists():
    try:
        model, class_names = load_checkpoint()
    except Exception as exc:
        model_load_error = str(exc)
else:
    model_load_error = (
        f"Model not found at {MODEL_PATH}. Run notebooks/2_model_training.ipynb first."
    )
```

### What this does

- First checks whether the model file exists.

- If it exists:
  - tries to load it
  - if loading fails, stores the error message

- If it does not exist:
  - stores a user-friendly explanation

This makes the app robust and easier to debug.

---

## 8. Image Transform

```python
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
```

### Why this matters

The app must preprocess uploaded images the same way the model saw images during evaluation.

If preprocessing is inconsistent, predictions can become unreliable.

### Step by step

- `Resize((224, 224))`
  - Matches the ResNet18 input size used in training/evaluation.

- `ToTensor()`
  - Converts image pixels into a PyTorch tensor.

- `Normalize(...)`
  - Standardizes the pixel values using ImageNet normalization.

This keeps inference behavior aligned with the notebooks.

---

## 9. `predict_animal` Function

This is the core prediction function connected to the button in the interface.

```python
def predict_animal(image):
```

It receives the uploaded image from Gradio.

---

### First safety check

```python
if model is None:
    return "Model unavailable", "0.00%", {"ERROR": model_load_error or "Model not loaded"}
```

If the model failed to load:

- the app returns a clear status
- confidence becomes `0.00%`
- the label output shows the reason

This prevents a crash.

---

### Second safety check

```python
if image is None:
    return "No image uploaded", "0.00%", {}
```

If the user presses the button without uploading an image:

- the app returns a friendly message
- no probabilities are shown

---

### Image type normalization

```python
if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

image = image.convert("RGB")
```

### Why this is needed

Gradio may provide the image in different formats depending on settings.

- `Image.fromarray(image)`
  - Converts a NumPy array into a PIL image if needed.

- `.convert("RGB")`
  - Ensures a standard 3-channel image format.

This makes the rest of preprocessing consistent.

---

### Preprocessing and batch creation

```python
tensor = transform(image).unsqueeze(0).to(device)
```

#### Step by step

- `transform(image)`
  - Applies resize, tensor conversion, and normalization.

- `unsqueeze(0)`
  - Adds a batch dimension.
  - PyTorch models expect input shaped like:
    - `(batch_size, channels, height, width)`

- `.to(device)`
  - Moves the tensor to CPU or GPU.

---

### Model inference

```python
with torch.no_grad():
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
```

#### What this does

- `torch.no_grad()`
  - Disables gradient tracking because prediction does not need gradients.

- `outputs = model(tensor)`
  - Produces raw class scores.

- `torch.softmax(outputs, dim=1)`
  - Converts raw scores into probabilities across classes.

- `[0]`
  - Removes the batch dimension, leaving probabilities for the single image.

---

### Getting the final prediction

```python
confidence, predicted_idx = torch.max(probabilities, dim=0)
predicted_class = class_names[predicted_idx.item()].upper()
confidence_text = f"{confidence.item() * 100:.2f}%"
```

#### Meaning

- `torch.max(...)`
  - Finds the most probable class.

- `predicted_idx.item()`
  - Converts the tensor index into a normal Python integer.

- `class_names[...]`
  - Maps numeric class index back to a readable class name.

- `.upper()`
  - Displays the label in uppercase in the UI.

- `confidence_text`
  - Formats the score as a percentage string.

---

### Building the probability dictionary

```python
probability_map = {
    class_names[i].upper(): float(probabilities[i].item())
    for i in range(len(class_names))
}
```

### Why this is useful

The Gradio `Label` component can show a ranked probability distribution.

This dictionary maps each class name to its probability.

Example:

```python
{
    "CAT": 0.01,
    "COW": 0.02,
    "DEER": 0.95,
    "DOG": 0.01,
    "LION": 0.01
}
```

---

### Return values

```python
return predicted_class, confidence_text, probability_map
```

These three outputs feed directly into the three UI output components:

- predicted class textbox
- confidence textbox
- probability label

---

## 10. `collect_examples` Function

```python
def collect_examples():
    examples = []
    if not DATASET_DIR.exists():
        return examples
```

### Purpose

This function looks inside the test dataset and picks one example image per class for the app UI.

If the dataset path is missing, it returns an empty list safely.

---

```python
for class_name in class_names:
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        continue
    for image_path in sorted(class_dir.iterdir()):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            examples.append([str(image_path)])
            break
return examples
```

### Step by step

- Loops over the class names.
- Builds the folder path for each class.
- Skips missing folders safely.
- Sorts files in that folder.
- Takes the first valid image file.
- Adds it to `examples` in the format Gradio expects.

Why `examples.append([str(image_path)])` and not just the string?

Because Gradio expects example rows in list form for the linked input component.

---

## 11. Building the Gradio Interface

```python
with gr.Blocks(title="Animal Classifier") as demo:
```

### Meaning

This creates the main Gradio app container.

Everything inside it becomes part of the interface.

---

### Intro text

```python
gr.Markdown(
    """
    # Animal Image Classifier
    Upload an image to classify it as one of five animals:
    **Cat, Cow, Deer, Dog, or Lion**.
    """
)
```

This creates the heading and short description users see at the top.

---

### Model status message

```python
if model_load_error:
    gr.Markdown(
        f"**Model status:** {model_load_error}"
    )
```

If there is a problem loading the model, the UI shows it immediately.

This is better than leaving the user confused.

---

### Layout rows and columns

```python
with gr.Row():
    with gr.Column():
```

This splits the interface into two columns:

- left side for image input
- right side for prediction output

---

### Left column components

```python
image_input = gr.Image(type="pil", label="Upload Animal Image", height=360)
predict_button = gr.Button("Classify Animal", variant="primary")
```

#### What these do

- `gr.Image(...)`
  - Creates the upload area.
  - `type="pil"` tells Gradio to provide a PIL image object to the function.

- `gr.Button(...)`
  - Creates the main action button.

---

### Example images

```python
example_images = collect_examples()
if example_images:
    gr.Examples(
        examples=example_images,
        inputs=image_input,
        label="Sample test images",
    )
```

This adds clickable sample images to the UI if examples are available.

That helps during demo time because you can test predictions quickly.

---

### Right column components

```python
predicted_class = gr.Textbox(label="Predicted Animal", interactive=False)
confidence = gr.Textbox(label="Confidence", interactive=False)
probabilities = gr.Label(label="Class Probabilities", num_top_classes=5)
```

#### What they show

- `predicted_class`
  - The final predicted label.

- `confidence`
  - Confidence percentage of the top class.

- `probabilities`
  - Distribution over all classes.

`interactive=False` means the user cannot type into those outputs.

---

## 12. Wiring the Button to the Function

```python
predict_button.click(
    fn=predict_animal,
    inputs=image_input,
    outputs=[predicted_class, confidence, probabilities],
)
```

### What this means

When the button is clicked:

1. the uploaded image is sent to `predict_animal`
2. the function runs inference
3. the outputs are displayed in the three output widgets

This is the event binding that makes the app interactive.

---

## 13. Footer Markdown

```python
gr.Markdown(
    """
    **Model Details**

    - Architecture: ResNet18 transfer learning
    - Dataset path: `data/animals_dataset/`
    - Saved model: `models/best_model.pth`
    """
)
```

This provides helpful context to the user at the bottom of the interface.

It is especially useful in demos or presentations.

---

## 14. Launching the App

```python
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
```

### What this does

- Runs only when you execute `python app/app.py`
- Starts the Gradio local web server

#### Parameters

- `share=False`
  - Keeps the app local only.

- `server_name="127.0.0.1"`
  - Binds to localhost.

- `server_port=7860`
  - Uses port 7860, which is a common Gradio default.

---

## 15. How the App Connects to the Notebook Pipeline

The app depends on Notebook 2 and Notebook 3 indirectly.

### From Notebook 2

It needs:

- `models/best_model.pth`
- saved `class_names`

Without the training notebook, the app has no trained model to use.

### From Notebook 3

Notebook 3 does not directly feed the app, but it validates that the saved model actually performs well.

That means the app is not just technically working, it is backed by measured performance.

---

## 16. What You Should Be Able to Explain

After studying this file, you should be able to explain:

- how the app finds the saved model
- why the model architecture must be rebuilt before loading weights
- why preprocessing must match the training/evaluation pipeline
- how Gradio input and output components work
- how the button calls the prediction function
- why the app safely handles missing models
- how probability outputs are built and shown

If you can explain those clearly, you understand the app at both a code level and a system level.
