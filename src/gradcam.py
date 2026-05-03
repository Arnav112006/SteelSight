"""
gradcam.py
----------
Grad-CAM heatmap generation and overlay visualization.
Works with any Keras model that has Conv2D layers.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

PALETTE_BG      = "#1D1D2B"
PALETTE_SURFACE = "#2A2A3D"
PALETTE_TEXT    = "#EAEAEA"


def get_last_conv_layer(model):
    """Find the name of the last Conv2D layer automatically."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array: np.ndarray, model,
                         last_conv_layer_name: str = None):
    """
    Compute Grad-CAM heatmap for a single image.

    Parameters
    ----------
    img_array : np.ndarray, shape (1, H, W, 3), values in [0, 1]
    model     : compiled Keras model
    last_conv_layer_name : str or None (auto-detected if None)

    Returns
    -------
    heatmap : np.ndarray, shape (H', W'), values in [0, 1]
    pred_prob : float
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)

    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap, float(predictions[0, 0])


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4):
    """
    Overlay colour-mapped heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray, shape (H, W, 3), uint8, RGB
    heatmap      : np.ndarray, shape (H', W'), values in [0, 1]
    alpha        : heatmap blend weight

    Returns
    -------
    overlay : np.ndarray, shape (H, W, 3), uint8, RGB
    """
    heatmap_resized = cv2.resize(heatmap,
                                 (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def gradcam_figure(original_img: np.ndarray, heatmap: np.ndarray,
                   pred_prob: float, true_label: int = None):
    """
    Return a matplotlib Figure showing:
    Original | Heatmap | Grad-CAM Overlay
    """
    overlay = overlay_heatmap(original_img, heatmap)
    pred_label = "Defect" if pred_prob > 0.5 else "No Defect"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             facecolor=PALETTE_BG)
    fig.suptitle(
        f"Grad-CAM  |  Prediction: {pred_label}  ({confidence:.1%} confidence)"
        + (f"  |  True: {'Defect' if true_label == 1 else 'No Defect'}"
           if true_label is not None else ""),
        color=PALETTE_TEXT, fontsize=13, fontweight="bold"
    )

    titles = ["Original", "Heatmap", "Grad-CAM Overlay"]
    images = [original_img, plt.cm.jet(heatmap)[:, :, :3], overlay]

    for ax, title, img in zip(axes, titles, images):
        ax.set_facecolor(PALETTE_SURFACE)
        if title == "Heatmap":
            ax.imshow(img, cmap="jet")
        else:
            ax.imshow(img)
        ax.set_title(title, color=PALETTE_TEXT, fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    return fig


def prepare_image_for_model(img_rgb: np.ndarray, target_size=(128, 128)):
    """Resize + normalise a uint8 RGB image into model input format."""
    resized = cv2.resize(img_rgb, target_size)
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1, H, W, 3)
