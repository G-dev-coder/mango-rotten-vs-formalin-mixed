"""
data_preprocessing.py
----------------------
Builds tf.data pipelines for the Rotten Mango vs Formalin-Treated Mango
binary classification task.

Expected folder layout (create this under data/ before running):

    data/
        train/
            formalin/   *.jpg, *.png ...
            rotten/     *.jpg, *.png ...
        val/
            formalin/
            rotten/
        test/
            formalin/
            rotten/

If you only have ONE big folder per class (no train/val/test split yet),
run `python data_preprocessing.py --split` first to auto-split your raw
images into an 70/15/15 train/val/test layout. See `--help` for options.
"""

import argparse
import os
import random
import shutil

import tensorflow as tf

import config


# ---------------------------------------------------------------------------
# tf.data pipelines used by train_model.py
# ---------------------------------------------------------------------------
def _build_augmentation_layer():
    """Light, mango-appropriate augmentation.

    We avoid heavy color jitter because color/skin texture is one of the
    main visual cues that distinguishes a naturally rotten mango (dark,
    irregular, sunken patches with mold) from a formalin-treated mango
    (unnaturally firm, unusually uniform/glossy skin with no insect or
    decay activity despite the fruit being past ripeness). Distorting hue
    too aggressively could erase that signal.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def get_datasets(train_dir=config.TRAIN_DIR, val_dir=config.VAL_DIR,
                  test_dir=config.TEST_DIR, img_size=config.IMG_SIZE,
                  batch_size=config.BATCH_SIZE, augment=True):
    """Return (train_ds, val_ds, test_ds, class_names) as tf.data.Dataset.

    Images are NOT rescaled here -- rescaling / preprocess_input is applied
    inside the model itself (see model.py) so the exact same saved .keras
    model can be reused unmodified inside the Streamlit app.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    if augment:
        aug = _build_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# Optional helper: auto-split a flat raw dataset into train/val/test
# ---------------------------------------------------------------------------
def split_raw_dataset(raw_dir, out_dir=config.DATA_DIR,
                       split=(0.70, 0.15, 0.15), seed=config.SEED):
    """Split data/raw/<class>/*.jpg into data/{train,val,test}/<class>/*.jpg

    raw_dir must look like:
        raw_dir/
            formalin/  *.jpg
            rotten/    *.jpg
    """
    assert abs(sum(split) - 1.0) < 1e-6, "split ratios must sum to 1.0"
    random.seed(seed)

    classes = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]
    if not classes:
        raise FileNotFoundError(f"No class subfolders found inside {raw_dir}")

    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        buckets = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for bucket, bucket_files in buckets.items():
            dest_dir = os.path.join(out_dir, bucket, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in bucket_files:
                shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_dir, f))

        print(f"[{cls}] total={n}  train={len(buckets['train'])}  "
              f"val={len(buckets['val'])}  test={len(buckets['test'])}")

    print(f"\nDone. Split dataset written under: {out_dir}")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", action="store_true",
        help="Split a flat raw dataset into train/val/test folders.",
    )
    parser.add_argument(
        "--raw-dir", default=os.path.join(config.DATA_DIR, "raw"),
        help="Path to raw_dir/{formalin,rotten}/*.jpg (default: data/raw)",
    )
    args = parser.parse_args()

    if args.split:
        split_raw_dataset(args.raw_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
