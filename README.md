# LeafCare — Tomato Leaf Disease Classification

A learning project built with Django and PyTorch.
Users upload a tomato leaf image and receive a model prediction.

## Supported classes

- Healthy
- Early blight
- Late blight
- Leaf mold

## Features

- Registration, login and logout
- Image validation and preview
- Prediction result pages
- Personal history with filtering and pagination
- CSV history export
- Personal dashboard
- Model evaluation report and confusion matrix
- Read-only prediction records in Django Admin

## Technology

Python 3.12, Django, PyTorch, TorchVision, Pillow,
scikit-learn, SQLite, HTML, CSS and JavaScript.

## Local setup — Windows PowerShell

Run these commands from the project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
python -c "import secrets; from dotenv import set_key; set_key('.env', 'DJANGO_SECRET_KEY', secrets.token_urlsafe(64)); print('Secret key saved.')"
```

Initialize the database and start the application:

```powershell
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/ and create an account.

Optional administrator account:

```powershell
python manage.py createsuperuser
```

## Model and reports

The trained model is not included in Git.
Prediction requires:

- artifacts/leafcare_best.pt

The performance page uses:

- artifacts/test_metrics.json
- artifacts/confusion_matrix.png

Training also creates artifacts/history.json.

To reproduce training, first obtain the PlantVillage dataset
and prepare the four supported classes under data/raw/.
Run the dataset-checking and splitting scripts before training:

```powershell
python check_dataset.py
python split_dataset.py
python train_model.py
python evaluate_model.py
```

The split script stops if data/split already exists.
Do not regenerate the split during model comparison.
The training script stops if its model checkpoint already exists.

## Recorded baseline results

- Model: MobileNetV3 Small with a trained final classification layer
- Unique images after exact-pixel deduplication: 5,438
- Training images: 3,805
- Validation images: 814
- Test images: 819
- Selected epoch: 8
- Test accuracy: 95.97%
- Early blight recall: 83.33%

These are recorded results from the initial experiment.
A new training run may produce different results.

## Checks

```powershell
python manage.py check
python manage.py test detection.test_access
python -m pip check
```

The five access tests cover login requirements, ownership of
results, history and CSV isolation, and result-page refresh.

## Limitations

- Supports only four tomato leaf classes.
- Cannot reliably reject unrelated images or unknown diseases.
- Model scores are not calibrated guarantees of correctness.
- Test images come from the same dataset as training images.
- Exact-pixel duplicates were excluded, but similar images or
  different images of the same leaf may remain across splits.
- Performance on real-world phone photos is not established.
- Uploaded images and all-class scores are not stored;
  history stores the top label, score, user and timestamp.
- Current settings are for local development.

## Dataset source

PlantVillage:
https://github.com/spMohanty/PlantVillage-Dataset

Refer to the dataset's own documentation for attribution
and applicable usage terms.

## Local files excluded from Git

.env, .venv/, db.sqlite3, data/, dataset-source/ and model weights.
## Experimental tomato leaf checker

Before disease classification, a separate MobileNetV3 Small
classifier checks whether the image is accepted as a tomato leaf.

- Training: 2,000 images
- Validation: 400 images
- Test: 400 images
- Test accuracy: 100% on the selected test set
- Data: PlantVillage tomato leaves versus selected Imagenette images
- Decision rule: argmax
- Rejected uploads are not saved as disease predictions.

This result does not establish reliable rejection of all unknown
images. Other plant leaves and real-world phone photos still need
evaluation.

Prediction now requires both model files:

- artifacts/leafcare_best.pt
- artifacts/leaf_check_best.pt

Checker preparation and evaluation:

    python prepare_leaf_check.py
    python train_leaf_check.py
    python evaluate_leaf_check.py

Imagenette source: https://github.com/fastai/imagenette