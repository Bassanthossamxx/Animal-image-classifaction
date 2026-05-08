# Results Explained — Study Guide

Think of this document as your TA sitting with you, looking at every output file the project produced and explaining what it means, why it looks the way it does, and how to talk about it confidently in a viva or report.

---

## The Big Picture First

Running code is not enough to prove a machine learning project works. You need **evidence**.

Think of it like a science experiment. You do not just say "the experiment worked." You show the data, the graphs, and the measurements. Your result files are exactly that — they are the proof that:

- the data was correct before training started
- the model actually learned something useful
- validation kept training under control
- the final model performs well on images it has never seen

Every result file answers a specific question. This guide tells you what question each one answers and how to talk about it.

---

## The Full List of Outputs

| File | Produced by | What it proves |
|------|-------------|----------------|
| `results/class_distribution.png` | Notebook 1 | You checked your data before training |
| `results/sample_images.png` | Notebook 1 | You visually verified the dataset is correct |
| `results/training_curves.png` | Notebook 2 | The model learned and validation controlled it |
| `models/best_model.pth` | Notebook 2 | The trained model is saved and reusable |
| `results/classification_report.txt` | Notebook 3 | Detailed performance numbers on unseen test data |
| `results/confusion_matrix.png` | Notebook 3 | Which classes confuse the model and which do not |
| `results/sample_predictions.png` | Notebook 3 | Visual proof the model makes sensible predictions |

---

---

## 1. `class_distribution.png`

### What is it?

This is a bar chart with three panels — one for train, one for validation, one for test. Each bar inside a panel represents one animal class (cat, cow, deer, dog, lion), and the height of the bar is the number of images in that class.

It was produced by the bar chart code in Notebook 1, Cell 3.

---

### Why do we make this plot at all?

Before you train any model, you need to understand your data. One of the most important things to check is **class balance** — are there roughly the same number of images for each animal, or is one animal massively overrepresented?

Think of it like preparing for an exam. If 90% of the practice questions are about cats and only a few are about lions, you will naturally get better at cats. The model works the same way.

---

### How to read it

**If the bars are roughly the same height across all classes:**
The dataset is balanced. The model will receive a similar amount of learning signal from each class, which generally leads to more fair and equal performance across classes.

**If one bar is much taller than the others:**
That class is overrepresented. The model will tend to predict that class more often, which can hurt performance on the smaller classes.

**In your project:**
The data is not perfectly balanced. Dog has the most training images (111) and cow has the fewest (86). But the difference is not dramatic — it is acceptable. You should mention this in a discussion as a known limitation, not a failure.

> **TA might ask:** What would you do if the dataset were seriously imbalanced?
> **Answer:** Several options: collect more images for the underrepresented classes, use class weights in the loss function to penalize mistakes on rare classes more heavily, or use oversampling techniques to artificially generate more examples of small classes.

---

### What to say about it

"The class distribution plot shows that the dataset is roughly balanced across all five classes in each split. Dog has slightly more training images than the others, but the difference is small enough that it should not significantly bias the model. This was checked before training began to ensure the data was suitable."

---

---

## 2. `sample_images.png`

### What is it?

This is a grid of example images from the training set — five images per class, one row per class. So it is a 5×5 grid of animal photos.

It was produced by Notebook 1, Cell 4.

---

### Why do we make this?

Numbers like "91 cat images" tell you a count, but they do not tell you whether those 91 images actually look like cats. You could have a folder named `cat` full of dog photos and the count would still say 91.

Looking at sample images is the quickest way to check whether the data is actually correct. It also lets you understand:
- How varied are the images? (different poses, backgrounds, angles)
- How visually distinct are the classes from each other?
- Is the image quality reasonable?

This is called **qualitative data inspection** and it is a standard step in any serious machine learning workflow.

---

### How to read it

**If the images clearly show different animals in different poses and backgrounds:**
Good. The model will be exposed to variety during training, which helps it generalize to new images.

**If two classes look visually very similar in the sample:**
Expect those classes to be confused in the model's predictions. For example, if some deer images look similar to some cow images, the model might struggle to separate them.

**If an image looks wrong (e.g., a folder labeled "cat" has a photo of a dog):**
The dataset has a labeling error. This is exactly why you inspect sample images first.

> **TA might ask:** What does "generalization" mean in machine learning?
> **Answer:** Generalization means the model can correctly classify images it has never seen before, not just images it was trained on. A model that only memorizes training images but fails on new ones has not generalized — it has overfit.

---

### What to say about it

"Sample image inspection confirmed that each class folder contains correctly labeled animal images with visible diversity in pose, background, and orientation. This qualitative check validated the dataset before any training took place."

---

---

## 3. `training_curves.png`

### What is it?

This figure has two plots side by side:
- Left: training loss and validation loss over epochs
- Right: training accuracy and validation accuracy over epochs

It was produced by Notebook 2, Cell 6, using the `history` dictionary that collected metrics after every epoch.

---

### Why is this the most important training output?

The training curves are like an X-ray of what happened during training. The final accuracy number alone tells you the end result, but the curves tell you the entire story — did the model learn gradually? Did it struggle at first and then improve? Did it overfit?

You should always look at these curves to understand the quality of your training run.

---

### How to read the loss curve

**The loss curve shows how wrong the model's predictions were at each epoch.**

A lower loss is better.

**Healthy pattern — what you want to see:**
- Both training loss and validation loss decrease together over epochs
- The two lines stay reasonably close to each other
- Validation loss eventually levels off (which may trigger early stopping)

This means the model is genuinely learning useful patterns that transfer to unseen data.

**Overfitting pattern — what you want to avoid:**
- Training loss keeps falling
- Validation loss stops falling and starts rising

This means the model is memorizing the training images rather than learning general visual features. It is essentially "cheating" — doing great on what it has seen, but failing on anything new.

**What happened in your project:**
Your validation loss went from 0.7051 in epoch 1 all the way down to 0.0808 in epoch 16, then did not improve for 4 more epochs. The model did not overfit dramatically — both curves trended downward together. This is a healthy training run.

> **TA might ask:** Why did training stop before epoch 20?
> **Answer:** Because of early stopping. The patience was set to 5 — if validation loss did not improve for 5 consecutive epochs, training stops. This prevents wasting time training after the model has already reached its best generalization.

---

### How to read the accuracy curve

**The accuracy curve shows what percentage of images were classified correctly at each epoch.**

A higher accuracy is better.

**Healthy pattern:**
- Training accuracy rises over epochs
- Validation accuracy also rises and remains reasonably close to training accuracy
- Both eventually plateau at a high value

**Warning pattern:**
- Training accuracy climbs very high (like 99–100%)
- Validation accuracy is much lower and stops improving
- Large gap between the two = overfitting

**What happened in your project:**
Validation accuracy jumped to 85.54% in epoch 1 and then jumped to 97.59% in epoch 2, where it stayed for the rest of training. This is actually a great sign — the pretrained features from ResNet18 were so powerful that even after one epoch of fine-tuning, the model classified validation images almost perfectly.

---

### What to say about it

"The training curves show consistent improvement in both loss and accuracy across the training epochs. Validation loss decreased steadily alongside training loss, and validation accuracy reached 97.59% by epoch 2 and remained stable. The validation monitoring and early stopping successfully identified the best checkpoint. No significant overfitting was observed."

---

---

## 4. `models/best_model.pth`

### What is it?

This is the saved checkpoint file from Notebook 2. It is not an image or a report — it is a binary file containing the model's learned weights and other important information.

`.pth` is PyTorch's standard file format for saved models.

---

### What is stored inside it?

Your training notebook saved a dictionary with four things:

```python
{
    'model_state_dict': ...,   # the trained weights — the actual model
    'class_names': [...],      # ['cat', 'cow', 'deer', 'dog', 'lion']
    'best_val_loss': 0.0808,   # the best validation loss achieved
    'history': {...},           # all epoch metrics for plotting
}
```

**`model_state_dict`**
This is the core content — every single number (weight and bias) that the model learned during training. Loading this back gives you the exact model that produced your best validation performance.

**`class_names`**
This is critically important. The model outputs numbers (class indices like 0, 1, 2, 3, 4), not names. `class_names` is the lookup table that converts 0 → "cat", 1 → "cow", and so on. By storing this in the checkpoint, the evaluation notebook and the app always use the exact same label order as training — no risk of mismatch.

**`best_val_loss`**
Records the best validation loss at the time this checkpoint was saved. Used to decide whether a new checkpoint should overwrite the old one during training.

**`history`**
The full list of training and validation metrics per epoch. Stored here so it is available later without having to retrain.

---

### Why saving the checkpoint matters

Without `best_model.pth`:
- The evaluation notebook has no model to load
- The Gradio app cannot make predictions
- Every time you wanted to test the model, you would have to retrain from scratch (which takes several minutes each time)

This file is the bridge between training and everything that comes after it.

> **TA might ask:** Why save the best checkpoint instead of the final one?
> **Answer:** The final epoch's model is not necessarily the best one. If the model started to overfit slightly in the last few epochs, the final weights might perform worse on unseen data than an earlier checkpoint. The best checkpoint is the one that generalized best during training — as measured by the lowest validation loss.

---

### What to say about it

"The training notebook saved the model checkpoint whenever validation loss improved. The final saved checkpoint reflects the epoch with the best validation performance — epoch 16 with a validation loss of 0.0808. This file was then used by the evaluation notebook and the Gradio app."

---

---

## 5. `classification_report.txt`

### What is it?

This is the main text performance report produced by Notebook 3. It contains the final numbers that describe how well the model performed on the unseen test set.

It was produced using sklearn's `classification_report` function and saved to a file.

---

### The actual numbers from your run

```
ANIMAL CLASSIFICATION - TEST SET RESULTS
============================================================

Overall Test Accuracy: 97.56%

              precision    recall  f1-score   support

         cat     1.0000    0.8824    0.9375        17
         cow     1.0000    1.0000    1.0000        16
        deer     0.8889    1.0000    0.9412        16
         dog     1.0000    1.0000    1.0000        16
        lion     1.0000    1.0000    1.0000        17

    accuracy                         0.9756        82
   macro avg     0.9778    0.9765    0.9757        82
weighted avg     0.9780    0.9756    0.9757        82
```

---

### Understanding every number in this table

#### Overall Accuracy — `97.56%`

This means the model correctly classified 80 out of 82 test images (97.56% of 82 ≈ 80).

Two images were predicted incorrectly. Which ones? The confusion matrix (next section) will show exactly where those mistakes happened.

> **TA might ask:** Is 97.56% a good result?
> **Answer:** Yes, especially given the small training set of 464 images. The high accuracy is largely due to transfer learning — ResNet18 brought in powerful pretrained features that worked well on this task. Without transfer learning, 97% accuracy on 464 training images would be very difficult to achieve.

---

#### Precision

**Formula:** True Positives ÷ (True Positives + False Positives)

**Plain English:** Of all the images the model *called* "cat", how many were actually cats?

Think of precision as the model's reliability when it makes a claim. High precision = when the model says something, it is usually right.

**Reading your numbers:**

| Class | Precision | What it means |
|-------|-----------|---------------|
| cat | 1.0000 | Every image predicted as cat was actually a cat |
| cow | 1.0000 | Every image predicted as cow was actually a cow |
| deer | 0.8889 | About 89% of images predicted as deer were actually deer — one non-deer was incorrectly called deer |
| dog | 1.0000 | Perfect |
| lion | 1.0000 | Perfect |

The deer precision of 0.8889 tells you that one image from a different class was incorrectly predicted as deer.

---

#### Recall

**Formula:** True Positives ÷ (True Positives + False Negatives)

**Plain English:** Of all the real cats in the test set, how many did the model successfully find?

Think of recall as the model's ability to not miss anything. High recall = the model rarely misses members of a class.

**Reading your numbers:**

| Class | Recall | What it means |
|-------|--------|---------------|
| cat | 0.8824 | The model found 88% of the actual cats — 2 real cats were predicted as something else |
| cow | 1.0000 | Every real cow was correctly identified |
| deer | 1.0000 | Every real deer was correctly identified |
| dog | 1.0000 | Perfect |
| lion | 1.0000 | Perfect |

The cat recall of 0.8824 is the most notable result. Out of 17 real cats in the test set, 2 were incorrectly classified as another animal. This is why cat has a lower F1-score than the other classes.

> **TA might ask:** Can you have high precision and low recall at the same time?
> **Answer:** Yes. Example: if the model only predicts "lion" when it is extremely confident — maybe only 5 out of 17 lions — those 5 predictions would all be correct (high precision), but it missed 12 lions (low recall). Conversely, if the model predicts "lion" for everything, it will never miss a real lion (high recall) but it will be wrong most of the time (low precision).

---

#### F1-Score

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Plain English:** A single number that balances precision and recall. If either one is low, F1 drops.

F1 of 1.0 = perfect. F1 of 0.0 = worst possible.

F1 is especially useful when you cannot tolerate either type of error — you want both precision and recall to be high.

**Reading your numbers:**

| Class | F1-Score |
|-------|----------|
| cat | 0.9375 — good, but pulled down by the lower recall |
| cow | 1.0000 — perfect |
| deer | 0.9412 — good, pulled down slightly by the lower precision |
| dog | 1.0000 — perfect |
| lion | 1.0000 — perfect |

---

#### Support

Support is simply the count of real test images for each class. It is not a performance metric — it is context.

It tells you how much data each metric is based on. With only 16–17 images per class, a single wrong prediction can shift the metric by about 6 percentage points. This is why you should always mention the small test set size when discussing your results.

---

#### Macro Average

```
macro avg  |  precision: 0.9778  |  recall: 0.9765  |  F1: 0.9757
```

Macro average = calculate the metric for each class separately, then take the simple average. Each class counts equally, regardless of how many images it has.

This is the fairest way to summarize performance across all classes.

---

#### Weighted Average

```
weighted avg  |  precision: 0.9780  |  recall: 0.9756  |  F1: 0.9757
```

Weighted average = same as macro, but each class's contribution is proportional to its support count. Classes with more images matter more.

When the dataset is near-balanced (like yours), weighted and macro averages are very similar.

---

### What to say about it

"The classification report shows 97.56% overall test accuracy on 82 unseen images. Three classes — cow, dog, and lion — achieved perfect precision and recall. Minor errors occurred in cat recall (2 cats misclassified) and deer precision (one non-deer predicted as deer). Macro F1-score was 0.9757, indicating strong and consistent performance across all classes. The small test set size means that even one or two errors can visibly affect the per-class metrics."

---

---

## 6. `confusion_matrix.png`

### What is it?

This is a heatmap produced in Notebook 3. It shows, for every possible pair of (true class, predicted class), how many test images fell into that combination.

---

### How to read it — the basics

- **Rows = true class** (the actual label of the image)
- **Columns = predicted class** (what the model said)
- **Diagonal = correct predictions** — the model said the right thing
- **Off-diagonal = mistakes** — the model said the wrong thing

For example, cell at row "cat", column "deer" = the number of actual cats that were predicted as deer.

**A perfect model** would have non-zero numbers only on the diagonal and zeros everywhere else.

---

### Reading your specific matrix

Based on your results (97.56% accuracy, 2 mistakes out of 82):

The matrix is nearly perfect — almost all counts are on the diagonal. The two off-diagonal cells that contain values correspond to:

1. 2 real cat images that were predicted as another class (explaining cat recall of 0.8824 = 15/17 correct)
2. 1 non-deer image that was predicted as deer (explaining deer precision of 0.8889)

Notice that cow, dog, and lion rows are completely diagonal — the model made zero mistakes for these classes.

---

### Why the confusion matrix is more useful than accuracy alone

Accuracy gives you one number: "80 out of 82 were correct."

The confusion matrix tells you the full story: "which 2 were wrong, and what were they predicted as instead?"

That matters because the pattern of errors tells you something about the model and the data:

**If a model often confuses deer and cow:**
Those two classes might look visually similar in your dataset — similar body shapes, similar backgrounds. This is a data issue, not necessarily a model issue.

**If a model confuses everything with one class:**
That class might be overrepresented in training, or its images might have distinctive features that bleed into other classes.

**In your project:**
The errors are isolated to cat and deer. This makes sense — cats can look similar to other animals in certain poses, and some images may be ambiguous. This is a reasonable limitation given the small dataset.

> **TA might ask:** What would you do if the confusion matrix showed systematic confusion between two specific classes?
> **Answer:** First, inspect the misclassified images visually to understand why the model is confused. Then consider: collecting more diverse training images for those classes, adding more aggressive augmentation for those classes, or using a more powerful pretrained model. The confusion matrix helps you *diagnose* the problem before choosing a solution.

---

### What to say about it

"The confusion matrix confirms that the model is highly accurate. The diagonal values are dominant, meaning most predictions are correct. The two off-diagonal errors are limited to two class interactions — real cats being predicted as another class, and one non-deer image being predicted as deer. These small errors are consistent with the precision and recall values in the classification report."

---

---

## 7. `sample_predictions.png`

### What is it?

This figure shows a 3×3 grid of randomly selected test images. For each image, the plot shows:

- The image itself (de-normalized back to natural colors)
- The true class label (what the animal actually is)
- The predicted class label (what the model said)
- The confidence percentage (how sure the model was)
- A **green** title if the prediction is correct
- A **red** title if the prediction is wrong

It was produced by Notebook 3, Cell 6.

---

### Why do we make this?

Numbers in a classification report are essential, but they are abstract. When you show this plot to a TA or instructor, they can *see* the model working. That is powerful.

This visualization serves three purposes:

1. **Confirmation** — seeing correct predictions with high confidence (like 99.5%) confirms the model is not just guessing
2. **Diagnosis** — if a wrong prediction appears (red title), you can look at the image and understand why the model might have been confused
3. **Presentation** — a grid of images with labels and confidence scores is much more engaging in a report or demo than raw numbers

---

### How to interpret it

**Correct prediction with high confidence (green, 90%+ confidence):**
The model is working well. It is not only getting the answer right but is also very certain about it.

**Correct prediction with low confidence (green, 55–70% confidence):**
The model got the right answer but was uncertain. This might mean the image was visually ambiguous — unusual angle, partial occlusion, unusual lighting. The model still "passed the exam" but it was a hard question.

**Wrong prediction (red):**
Look at the image carefully. Ask yourself: could a human mistake this too? If the image is genuinely ambiguous (e.g., a low-quality cat photo that looks a bit like a deer), the model's mistake is understandable. If it is an obvious error on a clear image, that suggests the model has a weakness in that class.

---

### Understanding the confidence score

The confidence comes from `softmax` applied to the model's raw output scores:

```python
probs = torch.softmax(output, dim=1)
confidence, pred_label = torch.max(probs, dim=1)
```

Softmax converts raw scores into probabilities that sum to 1.0 across all five classes. For example:

```
cat: 0.02   cow: 0.01   deer: 0.01   dog: 0.95   lion: 0.01
```

The confidence shown in the plot title is the highest probability — `0.95` in this example, displayed as `95.0%`.

**High confidence on a correct prediction** = the model has clearly learned the distinctive features of that class.

**High confidence on a wrong prediction** = the model is confidently wrong. This is more concerning than low-confidence mistakes, because it suggests the model has learned the wrong patterns for that class.

> **TA might ask:** Why use softmax instead of just taking the highest raw score?
> **Answer:** Raw scores (logits) can be any number and are not bounded. Softmax converts them into probabilities in the range 0–1 that sum to exactly 1.0, making them interpretable as confidence values. Without softmax, a raw score of 5.3 has no direct meaning. After softmax, 0.95 means "the model assigns 95% probability to this class."

---

### What to say about it

"The sample predictions figure provides qualitative confirmation of model performance. Most images are shown with green titles and high confidence scores, indicating that the model correctly identifies animals and is appropriately confident. Any red titles can be analyzed to understand the specific image characteristics that led to an incorrect prediction."

---

---

## How to Present All Results Together

When explaining the project to a TA, instructor, or in a presentation, tell it as a story in this order:

**Step 1 — Show you checked the data first (Notebook 1 outputs)**
> "Before training, we verified the dataset structure. The class distribution plot confirmed roughly balanced classes across all three splits. Sample image inspection confirmed that each folder contains correctly labeled images with diverse poses and backgrounds."

**Step 2 — Show the model learned properly (Notebook 2 output)**
> "The training curves show consistent improvement over epochs. Validation loss decreased alongside training loss, and early stopping correctly ended training once improvement plateaued. The best model checkpoint was saved at epoch 16."

**Step 3 — Show the final performance numbers (Notebook 3 outputs)**
> "On the unseen test set of 82 images, the model achieved 97.56% accuracy. Three out of five classes achieved perfect precision and recall. The confusion matrix shows that errors are isolated to only two class interactions. The macro F1-score of 0.9757 confirms strong and consistent performance across all classes."

**Step 4 — Be honest about limitations**
> "The test set contains only 82 images, so a small number of errors can noticeably shift the per-class metrics. The training set is also slightly imbalanced, with dog having more images than other classes. These are known limitations of the available dataset."

This structure — data check, training behavior, final performance, honest limitations — is exactly what a strong academic presentation looks like.

---

## What You Should Be Able to Explain

Go through this list before your viva:

**About the data outputs:**
- What does the class distribution plot show and why was it made before training?
- What is class imbalance and how could it affect the model?
- Why do you need to visually inspect sample images, not just count them?

**About the training curves:**
- What does a healthy loss curve look like versus an overfitting one?
- What does early stopping do and why does it help?
- Why was the best checkpoint saved based on validation loss, not training accuracy?

**About the checkpoint file:**
- What is `model_state_dict` and what does it contain?
- Why is `class_names` saved inside the checkpoint?
- What would happen if you lost this file?

**About the classification report:**
- What is the difference between precision and recall?
- What does the F1-score measure?
- Why is the test set size relevant when interpreting per-class metrics?
- What is macro average and why does it treat all classes equally?

**About the confusion matrix:**
- How do you read the rows and columns?
- What does an off-diagonal cell mean?
- What does the confusion matrix tell you that accuracy alone cannot?

**About the sample predictions:**
- Why do we show images instead of just numbers?
- What does the confidence score represent?
- What is concerning about a high-confidence wrong prediction?

If you can explain all of these clearly and in your own words, you are ready for the discussion.
