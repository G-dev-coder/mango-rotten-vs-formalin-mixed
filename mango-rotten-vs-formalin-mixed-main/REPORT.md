# Project Report — Mango Safety Classifier

**Dataset source:** Images were sourced from the FruitVision dataset
(Bijoy, Tasnim, Awsaf & Hasan, 2025, Mendeley Data, DOI:
10.17632/xkbjx8959c.2), keeping only the "Rotten Mango" and
"Formalin-Mixed Mango" classes and discarding the "Fresh Mango" class,
which this task does not use. Images were split 70/15/15 into
train/validation/test sets by class.

**How to use:** Open the deployed Streamlit URL, upload or capture a
mango photo, and the model returns a predicted class — "formalin-treated"
or "naturally rotten" — with a confidence score for each.

**Challenges & solutions:** Data scarcity and lighting inconsistency were
addressed with augmentation (rotation, zoom, brightness jitter) and
MobileNetV2 transfer learning to limit overfitting. Streamlit Cloud's
default Python version had no compatible TensorFlow wheels, and pinned
dependencies conflicted; both were fixed by loosening version
constraints and selecting Python 3.11 in deployment settings.

**Possible improvements:** Larger dataset, Grad-CAM explainability, and
validation against real formalin test-kit results.

---
*Dataset citation: Bijoy, Md Hasan Imam; Tasnim, Syeda Zarin; Awsaf, Syed
Ali; Hasan, Md Zahid (2025), "FruitVision: A Benchmark Dataset for Fresh,
Rotten, and Formalin-mixed Fruit Detection", Mendeley Data, V2, doi:
10.17632/xkbjx8959c.2. License: CC BY-NC-ND 4.0.*
