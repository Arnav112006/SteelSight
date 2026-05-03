"""
train.py
--------
Standalone training script. Run from the project root:

    python train.py

Trains all three models, saves weights to results/,
and saves training histories for the Streamlit app.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf

# ── silence TF info logs ──────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import download_dataset, load_dataframe, make_generators, make_eval_generator
from src.model import MODEL_REGISTRY, get_callbacks
from src.evaluate import predict, get_metrics, print_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE = (128, 128)


def train_all():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    base_path = download_dataset()
    final_df, raw_df, paths = load_dataframe(base_path)

    # save base_path so Streamlit can find images
    with open(os.path.join(RESULTS_DIR, "base_path.txt"), "w") as f:
        f.write(base_path)

    train_data, val_data, val_df, val_gen = make_generators(
        final_df, paths["train_img"],
        img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    # save val_df for the app
    val_df.to_csv(os.path.join(RESULTS_DIR, "val_df.csv"), index=False)

    all_histories = {}
    all_metrics = {}

    # ── 2. Train each model ───────────────────────────────────────────────────
    for name, build_fn in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        model = build_fn()
        model.summary()

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=get_callbacks(patience=3),
            verbose=1
        )

        # ── Save weights ──────────────────────────────────────────────────────
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(RESULTS_DIR, f"{safe_name}.keras")
        model.save(model_path)
        print(f"Model saved → {model_path}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        eval_gen = make_eval_generator(val_df, paths["train_img"],
                                       img_size=IMG_SIZE, batch_size=BATCH_SIZE)
        probs, preds, true_labels = predict(model, eval_gen)
        metrics = get_metrics(true_labels, preds, probs)
        print_report(true_labels, preds, model_name=name)

        all_histories[name] = history.history
        all_metrics[name] = metrics

    # ── 3. Persist histories & metrics ───────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "histories.json"), "w") as f:
        json.dump(all_histories, f)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    print("\n✅  Training complete. All models saved to results/")
    print("    Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    train_all()
