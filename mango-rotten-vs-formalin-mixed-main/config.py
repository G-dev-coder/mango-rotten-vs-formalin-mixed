"""
config.py
---------
Central configuration for the Mango Safety Classifier project
(Rotten Mango vs Formalin-Treated Mango).

Keeping every tunable parameter in one place makes the project easy to
grade, reproduce, and hand off between group members.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH_CUSTOM_CNN = os.path.join(SAVED_MODELS_DIR, "custom_cnn_mango_rotten_vs_formalin.keras")
MODEL_PATH_TRANSFER = os.path.join(SAVED_MODELS_DIR, "mobilenetv2_mango_rotten_vs_formalin.keras")

# The app always loads whichever file this points to. Point it at whichever
# model performed best on the test set after training (see train_model.py).
ACTIVE_MODEL_PATH = MODEL_PATH_TRANSFER

TRAINING_HISTORY_PLOT = os.path.join(ASSETS_DIR, "training_history.png")
CONFUSION_MATRIX_PLOT = os.path.join(ASSETS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_TXT = os.path.join(ASSETS_DIR, "classification_report.txt")

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
# Binary classification: index 0 -> "formalin", index 1 -> "rotten"
# (folder names sort alphabetically: "formalin" < "rotten")
CLASS_NAMES = ["formalin", "rotten"]

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 16
EPOCHS_CUSTOM_CNN = 30
EPOCHS_TRANSFER = 15
LEARNING_RATE = 1e-4
SEED = 42

# Streamlit page config
APP_TITLE = "Mango Safety Classifier"
APP_ICON = "🥭"
