from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
DATASET_DIR = ROOT_DIR / "data" / "animals_dataset" / "test"
DEFAULT_CLASS_NAMES = ["cat", "cow", "deer", "dog", "lion"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names


model = None
class_names = DEFAULT_CLASS_NAMES
model_load_error = None

if MODEL_PATH.exists():
    try:
        model, class_names = load_checkpoint()
    except Exception as exc:
        model_load_error = str(exc)
else:
    model_load_error = (
        f"Model not found at {MODEL_PATH}. Run notebooks/2_model_training.ipynb first."
    )

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


def predict_animal(image):
    if model is None:
        return "Model unavailable", "0.00%", {"ERROR": model_load_error or "Model not loaded"}

    if image is None:
        return "No image uploaded", "0.00%", {}

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()].upper()
    confidence_text = f"{confidence.item() * 100:.2f}%"
    probability_map = {
        class_names[i].upper(): float(probabilities[i].item())
        for i in range(len(class_names))
    }
    return predicted_class, confidence_text, probability_map


def collect_examples():
    examples = []
    if not DATASET_DIR.exists():
        return examples

    for class_name in class_names:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                examples.append([str(image_path)])
                break
    return examples


with gr.Blocks(title="Animal Classifier") as demo:
    gr.Markdown(
        """
        # Animal Image Classifier
        Upload an image to classify it as one of five animals:
        **Cat, Cow, Deer, Dog, or Lion**.
        """
    )

    if model_load_error:
        gr.Markdown(
            f"**Model status:** {model_load_error}"
        )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Animal Image", height=360)
            predict_button = gr.Button("Classify Animal", variant="primary")

            example_images = collect_examples()
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Sample test images",
                )

        with gr.Column():
            predicted_class = gr.Textbox(label="Predicted Animal", interactive=False)
            confidence = gr.Textbox(label="Confidence", interactive=False)
            probabilities = gr.Label(label="Class Probabilities", num_top_classes=5)

    predict_button.click(
        fn=predict_animal,
        inputs=image_input,
        outputs=[predicted_class, confidence, probabilities],
    )

    gr.Markdown(
        """
        **Model Details**

        - Architecture: ResNet18 transfer learning
        - Dataset path: `data/animals_dataset/`
        - Saved model: `models/best_model.pth`
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
