# 🔩 Steel Surface Defect Detection

A computer vision pipeline that detects manufacturing defects in steel surfaces using CNNs and transfer learning, with a Streamlit web app for live inference.

Built on the [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) Kaggle dataset.

---

## 🖥️ App Preview

The Streamlit app includes 5 pages:

| Page | Description |
|---|---|
| 🏠 Overview | Project summary and setup guide |
| 📊 EDA | Class distribution, sample images, ground-truth defect masks |
| 🧠 Train & Evaluate | Training curves, confusion matrices, ROC comparison |
| 🔬 Grad-CAM | Heatmap visualisation of model attention |
| 🖼️ Inference | Upload any steel image for live prediction |

---

## 🧠 Models

Three models are trained and compared:

| Model | Key Feature |
|---|---|
| Baseline CNN | 3 conv blocks, no regularisation |
| Improved CNN | BatchNormalization + `padding="same"` |
| MobileNetV2 (Transfer) | Pretrained ImageNet weights, frozen base |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/steel-defect-detection.git
cd steel-defect-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Kaggle API required** for auto-download. Place your `kaggle.json` in `~/.kaggle/` before running.

### 3. Train all models

```bash
python train.py
```

This downloads the dataset, trains all 3 models, and saves weights + metrics to `results/`.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
steel-defect-detection/
├── app.py               # Streamlit web app (5 pages)
├── train.py             # Standalone training script
├── requirements.txt
├── src/
│   ├── dataset.py       # Data loading, generators, RLE decode
│   ├── model.py         # CNN architectures + MobileNetV2
│   ├── evaluate.py      # Metrics, confusion matrix, ROC curves
│   └── gradcam.py       # Grad-CAM heatmap generation
├── results/             # Saved models & training histories (gitignored)
└── .streamlit/
    └── config.toml      # Dark theme config
```

---

## 🔬 Grad-CAM Explainability

Grad-CAM highlights the regions of the image that contributed most to the model's prediction, making the model interpretable and trustworthy for production use.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model training
- **OpenCV** — image processing
- **scikit-learn** — evaluation metrics
- **Streamlit** — web interface
- **Matplotlib / Seaborn** — visualisations
- **kagglehub** — dataset download

---

## 📄 License

MIT
