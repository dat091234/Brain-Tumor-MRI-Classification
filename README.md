# Brain Tumor MRI Classification With Models

A deep learning project that compares four architectures for classifying brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor

The project explores progressive fine-tuning strategies, evaluates model performance on a held-out test set, and uses visualization techniques **(Grad-CAM)** to investigate model predictions across both convolutional and Vision Transformer backbones

## Models and Experiments

The following architectures are evaluated:

- **Custom CNN** (3 convolutional blocks + classifier head)
- **ResNet18**
- **DenseNet121**
- **DINOv3 ViT-S/16**

Beyond training a baseline CNN from scratch, the experiments investigate a **3-phase progressive fine-tuning strategy** for the pretrained models: starting from linear probing (training only the classification head) and gradually unfreezing deeper layers/blocks with decayed learning rates

## Dataset

This project uses the [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains 7,200 MRI images across four classes: 
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The data is organized into:
- **Training & Validation:** 5,600 images, partitioned using an 80/20 stratified split (4,480 training and 1,120 validation images)
- **Testing:** 1,600 images (400 per class), reserved strictly for final evaluation

## Evaluation and Explainability

Model performance is evaluated using classification reports (Accuracy, Precision, Recall, F1-score) and confusion matrices **Grad-CAM** is implemented across all four architectures—including a custom patch-token reshaping pipeline for DINOv3—to compare attention regions on both correct and incorrect predictions

Saved model checkpoints are reloaded during evaluation to ensure reproducibility without retraining

## Results

The table below summarizes the final test set performance using the best validation checkpoint for each architecture:

| Model | Best Training Strategy | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| Custom CNN | Trained from scratch | 81.44% | 0.81 |
| ResNet18 | Fine-tuned `layer3`, `layer4` & head (Phase 3) | 94.31% | 0.94 |
| DenseNet121 | Fine-tuned `denseblock3`, `denseblock4` & head (Phase 3) | 94.44% | 0.94 |
| DINOv3 ViT-S/16 | Fine-tuned last 4 Transformer blocks & head (Phase 3) | 95.06% | 0.95 |

### Key Observations
- **Progressive Fine-Tuning Impact:** Unfreezing deeper blocks in Phases 2 and 3 yielded substantial validation accuracy gains (~6–8% improvement) over Phase 1 feature extraction across all pretrained backbones
- **Missed Tumor Detection (False Negatives):** All three pretrained models correctly identified 100% (400/400) of tumor-free images However, **DINOv3 ViT-S/16** achieved the lowest rate of tumor images misclassified as `notumor` (18 cases), outperforming ResNet18 (26 cases), DenseNet121 (39 cases), and the Custom CNN (133 cases)
- **Challenging Classes:** Across all models, `glioma` was the most difficult class to distinguish, most frequently confused with `meningioma`

## Notebook

For the full implementation, phase-by-phase training logs, confusion matrices, and cross-architecture Grad-CAM visualizations, please check out the [Kaggle notebook](https://www.kaggle.com/code/onphct/brain-tumor-mri-classification-with-models).

## Limitations and Future Work

This project is intended for educational and experimental purposes and should not be used for clinical diagnosis. Key technical limitations and directions for future research include:

- **Dataset Scope & Generalization:** Evaluation was limited to a single 4-class dataset without verified patient-level splits or duplicate checks. Future work should investigate potential data leakage, adopt patient-level splitting, and validate on external multi-center datasets.
- **Architecture & Hyperparameter Exploration:** Beyond the four tested models, future experiments could evaluate architectures with different efficiency trade-offs (e.g., EfficientNet, ConvNeXt, MobileNet variants) alongside systematic hyperparameter tuning and alternative unfreezing schedules.
- **Deployment Benchmarking:** The current comparison focuses on accuracy and parameter counts. Assessing inference latency, memory footprint, and energy consumption is needed to determine real-world deployment feasibility.
- **Explainability & Clinical Reliability:** Grad-CAM heatmaps did not always align with apparent tumor regions (especially for DINOv3), and ground-truth segmentation masks were unavailable to verify attributions. Future work should explore Transformer-specific explainability methods and test robustness against varying MRI quality.