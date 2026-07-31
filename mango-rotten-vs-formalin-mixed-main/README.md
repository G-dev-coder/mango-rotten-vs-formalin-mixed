# 🥭 Mango Safety Classifier — Rotten vs. Formalin-Treated Mango

**GET 324 — Cloud Computing and AI Model Deployment for Engineering Applications**
Laboratory Exercise 10 (Mini-Project)

A CNN-based binary image classifier that distinguishes **naturally rotten
mangoes** from **mangoes suspected of formalin treatment**, deployed as a
Streamlit web application.

---

## 1. Problem Statement

Formalin (formaldehyde solution) is sometimes illegally applied to fruit to
mask spoilage and extend shelf life, which poses a food-safety risk. This
project trains a Convolutional Neural Network to flag suspicious mangoes
from a photo, as a low-cost visual screening aid.

| | |
|---|---|
| **Task** | Binary image classification |
| **Classes** | `formalin` (formalin-suspected), `rotten` (naturally rotten) |
| **Architectures provided** | (a) Custom CNN from scratch, (b) MobileNetV2 transfer learning |
| **Deployment** | Streamlit Community Cloud |

---

## 2. Project Structure

```
mango-formalin-classifier/
├── app.py                      # Streamlit application (deployment entry point)
├── config.py                   # All paths & hyperparameters in one place
├── data_preprocessing.py       # tf.data pipelines + raw-dataset auto-splitter
├── model.py                    # Custom CNN + MobileNetV2 transfer-learning models
├── train_model.py              # Training / evaluation / plotting script
├── requirements.txt
├── .streamlit/config.toml      # App theme
├── utils/
│   └── make_dummy_dataset.py   # Generates synthetic images to smoke-test the pipeline
├── data/
│   ├── train/{formalin,rotten}/
│   ├── val/{formalin,rotten}/
│   └── test/{formalin,rotten}/
├── saved_models/                # Trained .keras files land here
└── assets/                      # training curves, confusion matrix, report.txt
```

---

## 3. Dataset

Place your labeled images in this structure (create the folders if missing):

```
data/train/formalin/*.jpg     data/train/rotten/*.jpg
data/val/formalin/*.jpg       data/val/rotten/*.jpg
data/test/formalin/*.jpg      data/test/rotten/*.jpg
```

Recommended split: 70% train / 15% val / 15% test, roughly balanced
between the two classes (aim for 100+ images per class minimum; more is
better for a CNN trained from scratch).

**Sourcing images for this specific task:**
- **Recommended: the FruitVision dataset** (Bijoy, Tasnim, Awsaf & Hasan,
  2025), hosted on Mendeley Data, already contains labeled Fresh /
  Rotten / Formalin-Mixed images for mango (and 4 other fruits). Download
  it from https://data.mendeley.com/datasets/xkbjx8959c/2 ("Download
  All"), then keep only the `Rotten Mango` and `Formalin-Mixed Mango`
  folders and discard `Fresh Mango`. License: **CC BY-NC-ND 4.0**
  (free for academic/research use — cite it in your report; full
  citation text is in `REPORT.md`).
- Alternatively, self-collect: photograph a batch of mangoes and let them
  rot naturally; treat a separate subset with a dilute
  formalin/formaldehyde solution under proper lab safety supervision
  (gloves, ventilation, disposal protocol), then photograph both sets
  under consistent lighting/background.
- If you only have one flat folder per class (no train/val/test split
  yet), run:
  ```bash
  python data_preprocessing.py --split --raw-dir data/raw
  ```
  which expects `data/raw/formalin/*.jpg` and `data/raw/rotten/*.jpg` and
  writes a 70/15/15 split into `data/train`, `data/val`, `data/test`.

**Sanity-check the pipeline without real photos:**
```bash
python utils/make_dummy_dataset.py --per-class 60
```
This fills `data/` with synthetic placeholder images so you can confirm
every script below runs end-to-end before spending time on real data
collection. Delete `data/train`, `data/val`, `data/test` contents and
replace with real photos before your final training run.

---

## 4. Setup

```bash
git clone <your-repo-url>
cd mango-formalin-classifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Train the Model

Train the transfer-learning model (recommended — needs less data, gets
to a good accuracy faster than a from-scratch CNN, which matters given
the mini-project timeline):

```bash
python train_model.py --arch transfer
```

Optionally fine-tune the top MobileNetV2 layers for extra accuracy:

```bash
python train_model.py --arch transfer --fine-tune
```

Or train the required-from-scratch custom CNN option instead:

```bash
python train_model.py --arch custom
```

This produces:
- `saved_models/mobilenetv2_mango.keras` or `saved_models/custom_cnn_mango.keras`
- `assets/training_history.png` — accuracy/loss curves
- `assets/confusion_matrix.png`
- `assets/classification_report.txt` — precision / recall / F1 per class

Set `config.ACTIVE_MODEL_PATH` to whichever saved model performed best on
the test set — `app.py` always loads that file.

---

## 6. Run the App Locally

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a mango photo (or use your webcam), and view the prediction with
confidence scores for both classes.

---

## 7. Deploy to the Cloud (Streamlit Community Cloud — free)

1. Push this repository to GitHub, including `saved_models/*.keras`
   (use [Git LFS](https://git-lfs.github.com/) if the file is large).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub.
3. Click **"New app"** (or **"Create app"**) → **"Deploy a public app
   from GitHub"** → select this repo/branch → set the main file path to
   `app.py`.
4. **Before clicking Deploy**, click **"Advanced settings..."** on that
   same form and set the **Python version** dropdown to **3.11**. This
   step matters: TensorFlow doesn't ship wheels for Streamlit Cloud's
   newest default Python versions, and this dropdown is the *only*
   reliable way to pin the version — it can't be changed later by
   editing files in the repo, only by deleting and redeploying the app
   with this dropdown set correctly at creation time.
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL — that URL is Deliverable #3.

### Deployment troubleshooting

| Symptom in the build log | Cause | Fix |
|---|---|---|
| `tensorflow-cpu==X has no wheels with a matching Python ABI tag` | Streamlit Cloud is running a Python version too new for the pinned TensorFlow release | Delete the app and redeploy, setting **Python 3.11** in Advanced Settings *before* clicking Deploy (see step 4 above) |
| `tensorflow-cpu X depends on protobuf>=6... and streamlit Y depends on protobuf<6...` | Exact-pinned versions of `tensorflow-cpu` and `streamlit` in `requirements.txt` demand incompatible `protobuf` ranges | Use loose version bounds instead of exact pins — see the `requirements.txt` contents in this repo, which are already tested to resolve cleanly |
| `Invalid requirement: '@"'` (or similar) | Stray shell syntax accidentally pasted into `requirements.txt` | Open the file, click **Raw** on GitHub to see the literal content, delete anything that isn't a plain package line |
| Still shows an old/wrong Python version after editing files or rebooting | Python version is fixed at app-creation time, not read from any file in the repo | Delete the app entirely and recreate it, setting the version in Advanced Settings before the first deploy |
| App builds but shows "No trained model found" | `saved_models/*.keras` wasn't actually pushed to GitHub | Check `git status` locally and confirm the file is committed and pushed; check it's visible on the GitHub repo page in your browser |

---

## 8. Team

| Name | Registration Number | GitHub Username | Contribution |
|---|---|---|---|
| Sunday, Godbless Ekerette | 23/EG/AE/025 | https://github.com/G-dev-coder | Dataset collection & preprocessing |
| YOUR NAME |  XX/EG/AE/XXX |  https://github.com/ | Model development & training |
| YOUR NAME |  XX/EG/AE/XXX |  https://github.com/ | Streamlit app & cloud deployment |
| YOUR NAME |  XX/EG/AE/XXX |  https://github.com/ | Documentation & report |

---

## 9. Course Learning Outcomes Addressed

- **CLO5** — Designed, trained, and evaluated a CNN (custom architecture)
  and a transfer-learning architecture (MobileNetV2) using TensorFlow/Keras
  for image data.
- **CLO7** — Deployed the trained model as a cloud-based Streamlit web
  application, with the project managed via Git/GitHub.
- **CLO8** — Documented the dataset, methodology, results, and challenges
  in this README and the accompanying project report.

---

## 10. Disclaimer

This tool is an educational proof-of-concept for a coursework mini-project.
It is **not** a certified food-safety instrument. Suspected formalin
contamination should be confirmed with an accredited laboratory test
before making any real consumption or commercial decisions.
