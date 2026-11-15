# Project requirements

## Problem statement

Build a system that takes an uploaded image and determines whether it is a real photograph or an AI-generated (fake) image. Along with the prediction, generate a visual heatmap (Grad-CAM) showing which parts of the image influenced that decision, so the result includes an explanation of why the model decided that way.

## 1. Core functionality

- Accept an image upload (JPG, JPEG, PNG).
- Classify the image as Real or Fake.
- Return a confidence score (0–100% or 0–1 probability) along with the label.
- Generate a Grad-CAM heatmap for that specific prediction, highlighting the image regions that most influenced the decision.
- Overlay the heatmap on the original image and display both together.

## 2. Model requirements

- Use a CNN-based deep learning model (for example, Xception pretrained on ImageNet) with transfer learning to reduce training time and improve accuracy on a limited dataset.
- Train on a multi-source dataset: real images from public sources such as CelebA and LFW, and fake images from multiple generation techniques such as GAN-based and diffusion-based methods, to avoid overfitting to one generator.
- Train in two stages: freeze the pretrained base first, then fine-tune deeper layers at a lower learning rate.
- Report accuracy, precision, recall, F1-score, and AUC.

## 3. Explainability requirement

- Every prediction must come with a Grad-CAM visualization; no result should be shown without its corresponding heatmap.
- The heatmap should clearly show which regions (for example, face, edges, and texture areas) contributed most to the Real or Fake verdict.
