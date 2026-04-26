<div align="center">

# 🧠 Early Stage Alzheimer's Disease Detection

### CNN-Based 3-Class MRI Classification with Digital Image Processing

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Google Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

| Metric | Value |
|:---|:---:|
| 🎯 Test Accuracy | **75.71%** |
| 📊 Weighted F1-Score | **0.7570** |
| 🔴 Alzheimer's F1 | **0.8000** |
| 🏋️ Training Epochs | **20** |
| 🖥️ Platform | **Google Colab** |

<br/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Digital Image Processing Techniques](#-digital-image-processing-techniques)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [How to Run](#-how-to-run)
- [Web Application](#-web-application)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)

---

## 🔍 Overview

Alzheimer's disease is the most prevalent form of dementia, affecting over **55 million people worldwide** (WHO, 2023). Early detection is critical — structural brain changes visible in MRI scans occur years before clinical symptoms appear, yet manual interpretation is expensive, time-consuming, and subjective.

This project builds a complete **deep learning pipeline** that automatically classifies brain MRI scans into three diagnostic categories:

| Class | Medical Meaning | Training Samples |
|:---:|:---|:---:|
| 🟢 **Normal** | Non-Demented — no cognitive impairment | 4,444 |
| 🟡 **Mild** | Mild Cognitive Impairment (MCI) | 4,412 |
| 🔴 **Alzheimer's** | Moderately Demented | 88 |

The system combines **Digital Image Processing techniques** for preprocessing and feature visualisation with a **ResNet50 transfer learning model**, and includes **Grad-CAM explainability** to confirm the model attends to clinically relevant brain regions.

---

## 🎯 Problem Statement

> Given a brain MRI scan image, automatically classify it into **Normal**, **Mild**, or **Alzheimer's Disease** with sufficient accuracy and interpretability to support clinical decision-making.

**Key challenges addressed:**
- **Severe class imbalance** — Alzheimer's class is 50× rarer than Normal (88 vs 4,444 training samples)
- **RAM overflow** — naive loading of all images crashes 12 GB Colab sessions; solved with `tf.data` lazy pipeline
- **Medical interpretability** — not just accuracy, but *why* the model predicts what it does (Grad-CAM)

---

## 🗃️ Dataset

The dataset is derived from the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** repository, available via [Kaggle]([https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset]).


### CSV Label Encoding

Labels are stored as one-hot encoded columns in CSV files:

| CSV Column | Meaning | Integer Label |
|:---:|:---|:---:|
| `ND = 1` | Non-Demented | `0` → Normal |
| `VMD = 1` or `MD = 1` | Very Mild / Mildly Demented | `1` → Mild |
| `MoD = 1` | Moderately Demented | `2` → Alzheimer's |

### Dataset Folder Structure

```
Alzheimers_Detection_dataset/
├── train/               ← 8,960 MRI images
├── valid/               ← 1,280 MRI images
├── test/                ← 640 MRI images
└── CSV_datafiles/
    ├── _train_classes.csv
    ├── _valid_classes.csv
    └── _test_classes.csv
```

---

## 🔄 Project Pipeline

```
Raw MRI Images + CSV Labels
         │
         ▼
┌─────────────────────────────┐
│   DIP Techniques            │  ← Visualisation on 3 sample images
│  (Preprocessing + Analysis) │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  tf.data Pipeline           │  ← Lazy batch loading (32 images at a time)
│  + Data Augmentation        │    
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  ResNet50 CNN               │  ← ImageNet weights, fine-tune from layer 140
│  + Custom Classification    │    24.1M total params / 15.5M trainable
│    Head                     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Training                   │  ← 20 epochs, class weights, 3 callbacks
│  Class Weights + Callbacks  │    Normal:0.671 | Mild:0.676 | Alz:33.879
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Evaluation                 │  ← Accuracy, F1, Confusion Matrix
│  + Grad-CAM Explainability  │    Grad-CAM attention heatmaps
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Flask Web App              │  ← Real-time scan upload + inference
│  + Training Dashboard       │    Grad-CAM overlay + probability bars
└─────────────────────────────┘
```

---

## 🖼️ Digital Image Processing Techniques

All DIP techniques are demonstrated on a representative subset of **3 images (1 per class)** with before/after visualisations. The **preprocessing chain** (Techniques 1–3) is applied to all images during training.

| # | Technique | OpenCV / NumPy Function | Purpose |
|:---:|:---|:---|:---|
| 1 | **Grayscale Conversion** | `cv2.cvtColor(BGR2GRAY)` | Remove colour noise; MRI is inherently grayscale |
| 2 | **Gaussian Blur** | `cv2.GaussianBlur((5,5))` | Suppress acquisition noise before edge detection |
| 3 | **Histogram Equalisation** | `cv2.equalizeHist()` | Redistribute pixel intensities for contrast enhancement |
| 4 | **CLAHE** | `cv2.createCLAHE(clipLimit=2.0)` | Adaptive local contrast — superior to global HE for MRI |
| 5 | **Canny Edge Detection** | `cv2.Canny(50, 150)` | Detect tissue boundary changes from atrophy |
| 6 | **Morphological Ops** | `cv2.erode() / cv2.dilate()` | Remove noise speckles, refine brain tissue masks |
| 7 | **Fourier Transform** | `np.fft.fft2() + fftshift()` | Frequency domain analysis of tissue density changes |
| 8 | **Otsu Thresholding** | `cv2.THRESH_BINARY + THRESH_OTSU` | Automatic brain segmentation without manual threshold |
| 9 | **Laplacian Sharpening** | `cv2.Laplacian(CV_64F)` | Enhance cortical folds and sulcal detail |

### Preprocessing Chain Applied to All Training Images

```
BGR Image
    → Grayscale         (cv2.COLOR_BGR2GRAY)
    → Gaussian Blur     (5×5 kernel)
    → Histogram Equali  (cv2.equalizeHist)
    → RGB               (cv2.COLOR_GRAY2RGB)
    → Normalise         (/ 255.0 → float32 [0,1])
    → Resize            (224×224 for ResNet50)
```

---

## 🏗️ Model Architecture

```
Input (224 × 224 × 3)
        │
        ▼
┌───────────────────────────────────┐
│  ResNet50 Backbone                │
│  Pre-trained on ImageNet          │
│  Layers 0–139: FROZEN             │  ← Universal features (edges, textures)
│  Layers 140–175: FINE-TUNED       │  ← Domain-adapted to MRI features
└───────────────┬───────────────────┘
                │  7×7×2048 feature maps
                ▼
        GlobalAveragePooling2D         → 2048-d vector
                │
        BatchNormalization
                │
        Dense(256, relu) + L2(1e-4)
                │
        Dropout(0.4)
                │
        Dense(128, relu) + L2(1e-4)
                │
        Dropout(0.3)
                │
        Dense(3, softmax)
                │
                ▼
    [P(Normal), P(Mild), P(Alzheimer's)]
```

### Parameter Summary

| Component | Parameters |
|:---|:---:|
| Total parameters | 24,153,731 |
| Trainable (fine-tuned) | 15,539,971 |
| Frozen (preserved) | 8,613,760 |
| Model size | ~92 MB |


## 🏋️ Training Strategy

### Hyperparameters

| Parameter | Value | Reason |
|:---|:---:|:---|
| Optimiser | Adam | Adaptive learning rate, fast convergence |
| Learning Rate | `1e-4` | Conservative start for fine-tuning |
| Loss Function | Categorical Cross-Entropy | Standard for multi-class softmax |
| Batch Size | 32 | Balance between GPU utilisation and gradient stability |
| Epochs | 20 | With EarlyStopping active |

### Class Weights (Handles 1:50 Imbalance)

```python
Normal      → 0.671
Mild        → 0.676
Alzheimer's → 33.879   # 50× higher weight forces focus on rare class
```

### Callbacks

| Callback | Configuration | Effect |
|:---|:---|:---|
| `EarlyStopping` | `monitor=val_accuracy, patience=5` | Stops and restores best weights |
| `ReduceLROnPlateau` | `factor=0.5, patience=3, min_lr=1e-7` | Halves LR when val_loss plateaus |
| `ModelCheckpoint` | `monitor=val_accuracy, save_best_only=True` | Saves best model to `.h5` |

### Data Augmentation (Training Only)

```python
tf.image.random_flip_left_right()
tf.image.random_flip_up_down()
tf.image.random_brightness(max_delta=0.1)
tf.image.random_contrast(lower=0.9, upper=1.1)
```

## 📊 Results

### Final Test Set Performance

| Metric | Value |
|:---|:---:|
| **Test Accuracy** | **75.71%** |
| **Test Loss** | 0.5570 |
| **Weighted Precision** | 0.7571 |
| **Weighted Recall** | 0.7571 |
| **Weighted F1-Score** | **0.7570** |
| **Macro F1-Score** | 0.7711 |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| 🟢 Normal | 0.7570 | 0.7618 | 0.7594 | 319 |
| 🟡 Mild | 0.7564 | 0.7516 | 0.7540 | 314 |
| 🔴 Alzheimer's | **0.8000** | **0.8000** | **0.8000** | 5 |
| **Weighted Avg** | 0.7571 | 0.7571 | 0.7570 | 638 |

### Per-Class Accuracy

```
Normal      ████████████████████░░░░  76.18%  (243/319)
Mild        ███████████████████░░░░░  75.16%  (236/314)
Alzheimer's ████████████████████████  80.00%   (4/5)
Overall     ███████████████████░░░░░  75.71%  (483/638)
```

### Training Curve Summary

| Epoch | Train Acc | Val Acc | Event |
|:---:|:---:|:---:|:---|
| 1 | 47.95% | 50.82% | Training begins |
| 7 | 62.78% | 70.71% | Fine-tuned layers adapting |
| 10 | 65.87% | 71.18% | ReduceLR → `5e-5` |
| 14 | 69.03% | **77.29%** | Best validation epoch |
| 20 | 70.93% | 71.42% | Final epoch |

### Memory Efficiency

| Approach | RAM Used | Result |
|:---|:---:|:---:|
| NumPy array (old) | ~7.5 GB | Session crash |
| `tf.data` pipeline (new) | ~50 MB peak | Stable |


## 📁 Project Structure

```
alzheimers-detection/
│
├── 📓 Alzheimers_disease_detection.ipynb   ← Main notebook (Google Colab)
│
├── 🌐 app/
│   ├── app.py                              ← Flask backend (inference API)
│   └── templates/
│       └── index.html                      ← Frontend web UI
│
├── 📊 outputs/
│   ├── best_alzheimer_model.h5             ← Saved model (best val_accuracy)
│   └── alzheimer_cnn_model.h5             ← Final saved model
│
├── 📋 requirements.txt
└── 📖 README.md
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) GPU with CUDA for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/alzheimers-detection.git
cd alzheimers-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
tensorflow>=2.19.0
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
flask
Pillow
```

### 3. Prepare the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset) and structure it as follows:

```
/content/drive/MyDrive/Alzheimers_Detection_dataset/
├── train/
├── valid/
├── test/
└── CSV_datafiles/
    ├── _train_classes.csv
    ├── _valid_classes.csv
    └── _test_classes.csv
```

## 🚀 How to Run

### Option A — Google Colab (Recommended)

1. Upload `Alzheimers_disease_detection.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set Runtime → **T4 GPU**
3. Upload the dataset to Google Drive at the path above
4. Run all cells in order — the full pipeline takes approximately **50 minutes** on T4

### Option B — Local Jupyter

```bash
jupyter notebook Alzheimers_disease_detection.ipynb
```

> Requires GPU with at least 8 GB VRAM for comfortable training. On CPU, training will be significantly slower.

### Notebook Cell Execution Order

```
Cell 1   → Mount Google Drive
Cell 3   → Install dependencies + imports
Cell 5   → Load CSV files
Cell 6–7 → Class distribution analysis
Cell 9   → Load 3 sample images for DIP demo
Cell 11–20 → Run all 10 DIP techniques
Cell 23  → Build tf.data pipeline
Cell 25  → Build ResNet50 model
Cell 27  → Train model (20 epochs)
Cell 29  → Plot training curves
Cell 31  → Evaluate on test set
Cell 32  → Classification report
Cell 33–34 → Confusion matrices
Cell 36  → Visualise predictions
Cell 38  → Grad-CAM explainability
Cell 40  → Save model
Cell 42  → Final summary
```

## 🌐 Web Application

The trained model can be deployed as a Flask web application for real-time inference.

### Run the Flask App

```bash
cd app
python app.py
```

Open your browser at `http://localhost:5000`

### Features

- **Drag-and-drop** MRI scan upload
- **Real-time CNN inference** with actual model weights
- **Probability bars** for all 3 classes
- **Grad-CAM overlay** showing which brain regions influenced the prediction
- **Training dashboard** — accuracy/loss curves, confusion matrix, per-class metrics

### API Endpoint

```
POST /predict
Content-Type: multipart/form-data
Body: file=<MRI image>

Response:
{
  "prediction":    "Normal" | "Mild" | "Alzheimer's",
  "confidence":    87.3,
  "probabilities": [{"class": "Normal", "probability": 87.3}, ...],
  "gradcam":       "data:image/png;base64,...",
  "risk":          "Low" | "Medium" | "High",
  "description":   "Clinical interpretation text"
}
```

## 🛠️ Technologies Used

| Technology | Version | Usage |
|:---|:---:|:---|
| **TensorFlow / Keras** | 2.19.0 | Model building, training, tf.data pipeline |
| **ResNet50** | ImageNet weights | Transfer learning backbone |
| **OpenCV (cv2)** | 4.x | All 10 DIP techniques |
| **NumPy** | 1.24+ | Array operations, predictions |
| **Pandas** | 2.0+ | CSV loading and label extraction |
| **Matplotlib** | 3.7+ | All visualisations and DIP output plots |
| **Seaborn** | 0.12+ | Confusion matrix heatmaps |
| **Scikit-learn** | 1.3+ | Metrics, class weight computation |
| **Flask** | 2.x | Web application backend |
| **Google Colab** | — | Training environment (NVIDIA T4 GPU) |

## 🔮 Future Work

- [ ] **GAN-based augmentation** — Synthetically generate Alzheimer's MRI samples to address the 1:50 class imbalance
- [ ] **Multi-modal fusion** — Incorporate PET scans, diffusion tensor imaging (DTI), and clinical biomarkers
- [ ] **3D CNN** — Exploit volumetric MRI information across axial, coronal, and sagittal planes simultaneously
- [ ] **4-class extension** — Separate Very Mild Demented (VMD) and Mild Demented (MD) for finer clinical granularity
- [ ] **RadImageNet pretraining** — Use medical imaging-specific pre-trained weights for better transfer
- [ ] **Clinical validation** — Evaluate on an independent hospital dataset before any clinical deployment
- [ ] **Mobile deployment** — Convert model to TensorFlow Lite for on-device inference

## ⚠️ Disclaimer

This project is developed for **academic and research purposes** as part of a Digital Image Processing course. The model has **not been clinically validated** and must **not** be used for medical diagnosis. Always consult a qualified healthcare professional for medical decisions.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **ADNI (Alzheimer's Disease Neuroimaging Initiative)** for the original MRI dataset
- **Kaggle** for making the dataset publicly accessible
- **He et al. (2016)** — Deep Residual Learning for Image Recognition (ResNet50)
- **Selvaraju et al. (2017)** — Grad-CAM: Visual Explanations from Deep Networks
- **TensorFlow Team** — tf.data pipeline documentation
- **Google Colab** — Free GPU compute environment

<div align="center">

Made with ❤️ for the Digital Image Processing course project

**Test Accuracy: 75.71% · F1-Score: 0.7570 · Alzheimer's F1: 0.8000**

</div>
