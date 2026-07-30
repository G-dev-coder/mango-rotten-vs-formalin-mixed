"""
train_model.py
----------------
Trains, evaluates, and saves the mango classifier.

Usage
-----
    # Train the from-scratch custom CNN
    python train_model.py --arch custom

    # Train MobileNetV2 transfer-learning model (recommended: fewer images
    # needed, higher accuracy for a mini-project timeline)
    python train_model.py --arch transfer

    # Train transfer model, then unfreeze top layers of MobileNetV2 and
    # fine-tune for a few more epochs
    python train_model.py --arch transfer --fine-tune

Outputs
-------
    saved_models/custom_cnn_mango.keras          (if --arch custom)
    saved_models/mobilenetv2_mango.keras          (if --arch transfer)
    assets/training_history.png                   accuracy/loss curves
    assets/confusion_matrix.png                    confusion matrix heatmap
    assets/classification_report.txt               precision/recall/F1
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless plotting (safe for servers / CI)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_preprocessing import get_datasets
from model import build_custom_cnn, build_transfer_model


def plot_history(history, extra_history=None, out_path=config.TRAINING_HISTORY_PLOT):
    """Plot accuracy & loss curves. If extra_history is given (fine-tune
    phase), the curves are concatenated so the plot shows both phases."""
    def concat(key):
        vals = list(history.history.get(key, []))
        if extra_history is not None:
            vals += list(extra_history.history.get(key, []))
        return vals

    acc, val_acc = concat("accuracy"), concat("val_accuracy")
    loss, val_loss = concat("loss"), concat("val_loss")
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs_range, acc, label="Train")
    axes[0].plot(epochs_range, val_acc, label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label="Train")
    axes[1].plot(epochs_range, val_loss, label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves -> {out_path}")


def evaluate_and_report(model, test_ds, class_names):
    """Run inference on the held-out test set and save a confusion matrix
    heatmap + a text classification report (precision/recall/F1)."""
    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + report)

    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    with open(config.CLASSIFICATION_REPORT_TXT, "w") as f:
        f.write(report)
    print(f"Saved classification report -> {config.CLASSIFICATION_REPORT_TXT}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {config.CONFUSION_MATRIX_PLOT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["custom", "transfer"], default="transfer",
                         help="Which architecture to train (default: transfer)")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override default epoch count")
    parser.add_argument("--fine-tune", action="store_true",
                         help="(transfer only) Unfreeze top MobileNetV2 layers "
                              "for a short fine-tuning phase after head training")
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    args = parser.parse_args()

    print("Loading datasets from:", config.DATA_DIR)
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print("Detected classes (alphabetical order == label index):", class_names)
    assert class_names == config.CLASS_NAMES, (
        f"Class order mismatch: found {class_names}, expected {config.CLASS_NAMES}. "
        "Check your data/train, data/val, data/test folder names."
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    extra_history = None

    if args.arch == "custom":
        model = build_custom_cnn()
        epochs = args.epochs or config.EPOCHS_CUSTOM_CNN
        save_path = config.MODEL_PATH_CUSTOM_CNN
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                             callbacks=callbacks)

    else:  # transfer
        model, base_model = build_transfer_model()
        epochs = args.epochs or config.EPOCHS_TRANSFER
        save_path = config.MODEL_PATH_TRANSFER
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                             callbacks=callbacks)

        if args.fine_tune:
            print("\nUnfreezing top layers of MobileNetV2 for fine-tuning...")
            head_weights_path = os.path.join(config.SAVED_MODELS_DIR, "_head_weights.weights.h5")
            model.save_weights(head_weights_path)

            fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers
            model, _ = build_transfer_model(fine_tune_at=fine_tune_at)
            model.load_weights(head_weights_path)  # resume from head-trained weights

            extra_history = model.fit(
                train_ds, validation_data=val_ds,
                epochs=args.fine_tune_epochs, callbacks=callbacks,
            )
            os.remove(head_weights_path)

    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved -> {save_path}")

    plot_history(history, extra_history)
    evaluate_and_report(model, test_ds, class_names)

    print("\nDone. Update config.ACTIVE_MODEL_PATH if you want app.py to "
          "load this model file.")


if __name__ == "__main__":
    main()
