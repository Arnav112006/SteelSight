"""
app.py
------
Streamlit web UI for Steel Surface Defect Detection.
Run with:  streamlit run app.py
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import rle_decode, load_image, load_dataframe, make_eval_generator
from src.evaluate import (
    predict, get_metrics, plot_training_history,
    plot_confusion_matrix, plot_roc_comparison, PALETTE
)
from src.gradcam import (
    make_gradcam_heatmap, gradcam_figure, prepare_image_for_model,
    get_last_conv_layer
)
from src.model import MODEL_REGISTRY

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    background-color: #1D1D2B;
    color: #EAEAEA;
    font-family: 'Syne', sans-serif;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.stSidebar { background-color: #141420 !important; border-right: 1px solid #2A2A3D; }
.stSidebar .stSelectbox label, .stSidebar .stRadio label { color: #EAEAEA !important; }

/* metric cards */
div[data-testid="metric-container"] {
    background-color: #2A2A3D;
    border: 1px solid #3a3a55;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #aaa !important; font-family: 'Space Mono', monospace; font-size: 11px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #E63946 !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}

/* section divider */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #E63946;
    letter-spacing: 0.05em;
    border-left: 4px solid #E63946;
    padding-left: 10px;
    margin: 24px 0 12px 0;
}

/* hero banner */
.hero {
    background: linear-gradient(135deg, #1D1D2B 0%, #2A1A2B 50%, #1D1D2B 100%);
    border: 1px solid #3a1a3a;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(230,57,70,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #E63946, #F4A261);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #aaa; font-size: 1rem; margin: 0; }

/* tag pills */
.tag {
    display: inline-block;
    background: #2A2A3D;
    border: 1px solid #457B9D;
    color: #457B9D;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 4px 3px;
}

/* inference result box */
.result-defect {
    background: rgba(230,57,70,0.12);
    border: 2px solid #E63946;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-clean {
    background: rgba(69,123,157,0.12);
    border: 2px solid #457B9D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}

/* scrollable table */
.dataframe { background-color: #2A2A3D !important; color: #EAEAEA !important; }

/* file uploader */
section[data-testid="stFileUploadDropzone"] {
    background-color: #2A2A3D;
    border: 2px dashed #3a3a55;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_histories():
    path = os.path.join(RESULTS_DIR, "histories.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_metric_results():
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model(name):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(RESULTS_DIR, f"{safe}.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

def get_base_path():
    p = os.path.join(RESULTS_DIR, "base_path.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return None

def get_val_df():
    p = os.path.join(RESULTS_DIR, "val_df.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

def models_trained():
    return os.path.exists(os.path.join(RESULTS_DIR, "histories.json"))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔩 Steel Defect Detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA", "🧠 Train & Evaluate", "🔬 Grad-CAM", "🖼️ Inference"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <span class='tag'>TensorFlow</span>
    <span class='tag'>OpenCV</span>
    <span class='tag'>Streamlit</span>
    <span class='tag'>Grad-CAM</span>
    <span class='tag'>MobileNetV2</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#666'>Severstal Steel Defect Detection</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
        <h1>🔩 Steel Defect Detection</h1>
        <p>Computer vision pipeline for detecting surface defects in steel manufacturing<br>
        using CNNs, transfer learning, and Grad-CAM explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **📦 Dataset**
    - Severstal Kaggle Competition
    - ~12,600 steel surface images
    - 256 × 1600 px per image
    - Binary: Defect / No Defect
    """)
    c2.markdown("""
    **🧠 Models**
    - Baseline CNN (3 conv blocks)
    - Improved CNN + BatchNorm
    - MobileNetV2 Transfer Learning
    """)
    c3.markdown("""
    **🔬 Features**
    - Data augmentation
    - Grad-CAM heatmaps
    - ROC / AUC comparison
    - Live inference on new images
    """)

    st.markdown("---")
    st.markdown("<div class='section-title'>How to Use</div>", unsafe_allow_html=True)
    st.markdown("""
    1. **First time?** Run `python train.py` in your terminal to train all models.
    2. Come back here and explore **EDA** to understand the dataset.
    3. Go to **Train & Evaluate** to compare model performance.
    4. Use **Grad-CAM** to visualise what the model "sees".
    5. Upload your own steel image in **Inference**.
    """)

    if not models_trained():
        st.warning("⚠️  Models not trained yet. Run `python train.py` first.")
    else:
        st.success("✅  Trained models found in `results/`. All pages are active.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>",
                unsafe_allow_html=True)

    base_path = get_base_path()
    if base_path is None:
        st.warning("Run `python train.py` first to download the dataset.")
        st.stop()

    final_df, raw_df, paths = load_dataframe(base_path)
    train_img_path = paths["train_img"]

    # ── Class Distribution ────────────────────────────────────────────────────
    st.markdown("#### Class Distribution")
    counts = final_df["Label"].value_counts()
    no_def, defect = counts.get(0, 0), counts.get(1, 0)
    total = no_def + defect

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Images", f"{total:,}")
    c2.metric("Defect", f"{defect:,}", f"{defect/total:.1%}")
    c3.metric("No Defect", f"{no_def:,}", f"{no_def/total:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=PALETTE["bg"])
    ax1, ax2 = axes

    ax1.set_facecolor(PALETTE["surface"])
    bars = ax1.bar(["No Defect", "Defect"], [no_def, defect],
                   color=[PALETTE["secondary"], PALETTE["primary"]], width=0.5)
    for bar, val in zip(bars, [no_def, defect]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 str(val), ha="center", color=PALETTE["text"], fontsize=11)
    ax1.set_title("Class Distribution", color=PALETTE["text"], fontweight="bold")
    ax1.tick_params(colors=PALETTE["text"])
    ax1.set_facecolor(PALETTE["surface"])
    for spine in ax1.spines.values(): spine.set_edgecolor(PALETTE["surface"])

    ax2.set_facecolor(PALETTE["bg"])
    wedges, texts, autotexts = ax2.pie(
        [no_def, defect],
        labels=["No Defect", "Defect"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[PALETTE["secondary"], PALETTE["primary"]],
        wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2)
    )
    for t in texts: t.set_color(PALETTE["text"])
    for t in autotexts: t.set_color("#fff")
    ax2.set_title("Defect Ratio", color=PALETTE["text"], fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Sample Images ─────────────────────────────────────────────────────────
    st.markdown("#### Sample Steel Images")
    n = 8
    samples = final_df.sample(n).reset_index(drop=True)
    cols = st.columns(n)
    for i, col in enumerate(cols):
        img_name = samples.loc[i, "ImageId"]
        label = samples.loc[i, "Label"]
        img = load_image(os.path.join(train_img_path, img_name),
                         target_size=(256, 256))
        col.image(img, caption="Defect" if label == 1 else "No Defect", use_container_width=True)

    # ── Ground Truth Masks ────────────────────────────────────────────────────
    st.markdown("#### Ground Truth Defect Masks")
    mask_df = raw_df[raw_df["EncodedPixels"].notnull()]
    samples_mask = mask_df.sample(6).reset_index(drop=True)

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 7), facecolor=PALETTE["bg"])
    for i, ax in enumerate(axes2.flat):
        img_name = samples_mask.loc[i, "ImageId"]
        rle = samples_mask.loc[i, "EncodedPixels"]
        img = load_image(os.path.join(train_img_path, img_name))
        mask = rle_decode(rle)
        mask = cv2.resize(mask.astype("uint8"),
                          (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        overlay = img.copy()
        overlay[mask == 1] = [230, 57, 70]
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        ax.imshow(blended)
        ax.set_title(img_name, color=PALETTE["text"], fontsize=8)
        ax.axis("off")
        ax.set_facecolor(PALETTE["surface"])

    fig2.suptitle("Ground Truth Defect Regions (Red Overlay)",
                  color=PALETTE["text"], fontsize=14, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Train & Evaluate":
    st.markdown("<div class='section-title'>Model Training & Evaluation</div>",
                unsafe_allow_html=True)

    if not models_trained():
        st.warning("⚠️  No trained models found. Run `python train.py` in your terminal.")
        st.code("python train.py", language="bash")
        st.stop()

    histories = load_histories()
    metric_results = load_metric_results()
    model_names = list(histories.keys())

    # ── Metrics Summary Table ─────────────────────────────────────────────────
    st.markdown("#### Performance Summary")
    rows = []
    for name, m in metric_results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "AUC": f"{m['auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ── Select model ──────────────────────────────────────────────────────────
    selected = st.selectbox("Select model to inspect", model_names)
    hist = histories[selected]

    # ── Training curves ───────────────────────────────────────────────────────
    st.markdown("#### Training History")
    fig = plot_training_history(hist, model_name=selected)
    st.pyplot(fig); plt.close(fig)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")

    base_path = get_base_path()
    val_df = get_val_df()

    if base_path and val_df is not None:
        _, raw_df, paths = load_dataframe(base_path)
        train_img_path = paths["train_img"]
        model = load_model(selected)

        if model:
            with st.spinner("Running predictions…"):
                eval_gen = make_eval_generator(val_df, train_img_path)
                probs, preds, true_labels = predict(model, eval_gen)

            c1, c2 = st.columns(2)
            with c1:
                fig = plot_confusion_matrix(true_labels, preds, selected)
                st.pyplot(fig); plt.close(fig)
            with c2:
                m = get_metrics(true_labels, preds, probs)
                st.metric("Accuracy",  f"{m['accuracy']:.2%}")
                st.metric("F1-Score",  f"{m['f1']:.4f}")
                st.metric("ROC-AUC",   f"{m['auc']:.4f}")
                st.metric("Precision", f"{m['precision']:.4f}")
                st.metric("Recall",    f"{m['recall']:.4f}")

    # ── ROC Comparison ────────────────────────────────────────────────────────
    st.markdown("#### ROC Curve Comparison (All Models)")
    if base_path and val_df is not None:
        roc_data = {}
        with st.spinner("Computing ROC curves for all models…"):
            for name in model_names:
                m = load_model(name)
                if m:
                    eg = make_eval_generator(val_df, train_img_path)
                    pr, _, tl = predict(m, eg)
                    roc_data[name] = (tl, pr)

        if roc_data:
            fig = plot_roc_comparison(roc_data)
            st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Grad-CAM":
    st.markdown("<div class='section-title'>Grad-CAM Explainability</div>",
                unsafe_allow_html=True)
    st.markdown("Visualise *which regions* the model focuses on when making predictions.")

    if not models_trained():
        st.warning("Run `python train.py` first.")
        st.stop()

    base_path = get_base_path()
    val_df = get_val_df()
    if base_path is None or val_df is None:
        st.warning("Training data not found. Run `python train.py`.")
        st.stop()

    _, raw_df, paths = load_dataframe(base_path)
    train_img_path = paths["train_img"]

    # Only CNN models support Grad-CAM (not MobileNetV2 directly)
    cam_models = ["Baseline CNN", "Improved CNN"]
    model_choice = st.selectbox("Model", cam_models)
    n_samples = st.slider("Number of samples", 1, 8, 4)

    if st.button("Generate Grad-CAM Visualisations", type="primary"):
        model = load_model(model_choice)
        if model is None:
            st.error(f"Model '{model_choice}' not found. Run training first.")
            st.stop()

        last_conv = get_last_conv_layer(model)
        samples = val_df.sample(n_samples).reset_index(drop=True)

        for i in range(len(samples)):
            img_name = samples.loc[i, "ImageId"]
            true_label = int(samples.loc[i, "Label"])
            img_path = os.path.join(train_img_path, img_name)

            img_rgb = load_image(img_path)
            img_arr = prepare_image_for_model(img_rgb)

            heatmap, pred_prob = make_gradcam_heatmap(img_arr, model, last_conv)
            fig = gradcam_figure(img_rgb, heatmap, pred_prob, true_label)
            st.pyplot(fig); plt.close(fig)
            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🖼️ Inference":
    st.markdown("<div class='section-title'>Live Inference</div>",
                unsafe_allow_html=True)
    st.markdown("Upload any steel surface image to get a prediction + Grad-CAM heatmap.")

    if not models_trained():
        st.warning("Run `python train.py` first.")
        st.stop()

    model_choice = st.selectbox("Choose model", list(MODEL_REGISTRY.keys()))
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded = st.file_uploader("Upload a steel surface image",
                                type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        model = load_model(model_choice)
        if model is None:
            st.error(f"Model '{model_choice}' not found. Run `python train.py`.")
            st.stop()

        img_arr = prepare_image_for_model(img_rgb)
        pred_prob = float(model.predict(img_arr, verbose=0)[0][0])
        pred_label = 1 if pred_prob > threshold else 0
        confidence = pred_prob if pred_label == 1 else 1 - pred_prob

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        with c2:
            if pred_label == 1:
                st.markdown(f"""
                <div class='result-defect'>
                    <div class='result-label' style='color:#E63946'>⚠️ DEFECT DETECTED</div>
                    <br>
                    <b>Confidence:</b> {confidence:.1%}<br>
                    <b>Raw probability:</b> {pred_prob:.4f}<br>
                    <b>Model:</b> {model_choice}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-clean'>
                    <div class='result-label' style='color:#457B9D'>✅ NO DEFECT</div>
                    <br>
                    <b>Confidence:</b> {confidence:.1%}<br>
                    <b>Raw probability:</b> {pred_prob:.4f}<br>
                    <b>Model:</b> {model_choice}
                </div>""", unsafe_allow_html=True)

        # Grad-CAM for CNN models
        if "MobileNet" not in model_choice:
            st.markdown("#### Grad-CAM Analysis")
            try:
                last_conv = get_last_conv_layer(model)
                heatmap, _ = make_gradcam_heatmap(img_arr, model, last_conv)
                fig = gradcam_figure(img_rgb, heatmap, pred_prob)
                st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")
        else:
            st.info("Grad-CAM is available for Baseline CNN and Improved CNN models.")
