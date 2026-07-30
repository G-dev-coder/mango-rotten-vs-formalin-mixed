"""
utils/make_dummy_dataset.py
----------------------------
Generates a small SYNTHETIC placeholder dataset so you can sanity-check
that data_preprocessing.py / model.py / train_model.py / app.py all run
end-to-end BEFORE you plug in your real mango photos.

This does NOT create a scientifically meaningful dataset -- it just draws
mango-shaped blobs with different color/texture statistics for each class
so the training loop has something to chew on. Replace data/train,
data/val, data/test with your real, labeled mango photos before you train
the model you actually submit.

Usage:
    python utils/make_dummy_dataset.py --per-class 40
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFilter

import config


def _make_mango_image(hue_bias, blotchy, size=(300, 300), seed=0):
    rng = random.Random(seed)
    img = Image.new("RGB", size, (245, 235, 210))
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2
    rx, ry = size[0] // 3, size[1] // 2.6

    base_color = (
        max(0, min(255, hue_bias[0] + rng.randint(-15, 15))),
        max(0, min(255, hue_bias[1] + rng.randint(-15, 15))),
        max(0, min(255, hue_bias[2] + rng.randint(-15, 15))),
    )
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=base_color)

    # Add blotches for the "rotten" class; smooth gloss highlight for
    # "formalin" class.
    if blotchy:
        for _ in range(rng.randint(8, 16)):
            bx = cx + rng.randint(-int(rx * 0.8), int(rx * 0.8))
            by = cy + rng.randint(-int(ry * 0.8), int(ry * 0.8))
            r = rng.randint(5, 18)
            dark = (max(0, base_color[0] - 90), max(0, base_color[1] - 90),
                    max(0, base_color[2] - 70))
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=dark)
        img = img.filter(ImageFilter.GaussianBlur(1.2))
    else:
        # glossy highlight
        hx, hy = cx - rx // 2, cy - ry // 2
        draw.ellipse([hx - 20, hy - 30, hx + 20, hy + 10],
                      fill=(255, 255, 255))
        img = img.filter(ImageFilter.GaussianBlur(0.4))

    img = img.filter(ImageFilter.GaussianBlur(0.3))
    return img


def build(per_class=40, seed=config.SEED):
    rng = random.Random(seed)
    splits = {"train": int(per_class * 0.7), "val": int(per_class * 0.15),
              "test": per_class - int(per_class * 0.7) - int(per_class * 0.15)}

    class_settings = {
        "formalin": {"hue_bias": (235, 205, 90), "blotchy": False},
        "rotten": {"hue_bias": (150, 110, 60), "blotchy": True},
    }

    counter = 0
    for split, n in splits.items():
        for cls, settings in class_settings.items():
            out_dir = os.path.join(config.DATA_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                counter += 1
                img = _make_mango_image(
                    settings["hue_bias"], settings["blotchy"],
                    seed=rng.randint(0, 999999),
                )
                img.save(os.path.join(out_dir, f"dummy_{cls}_{split}_{i}.jpg"),
                          quality=90)

    print(f"Generated {counter} synthetic placeholder images under {config.DATA_DIR}")
    print("Reminder: this is ONLY for testing that the pipeline runs. "
          "Replace with real labeled mango photos before final training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=40)
    args = parser.parse_args()
    build(per_class=args.per_class)
