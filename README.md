# Heart Disease Risk Prediction API

A machine learning web service that predicts a patient's risk of heart disease from clinical parameters, built with scikit-learn and served via a Flask REST API. Deployed on Render.

🔗 **Dataset:** [Heart Disease Dataset — Kaggle (johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
🚀 **Live Demo:** `https://<your-app>.onrender.com` *(update after deploying)*

## Overview

This project trains a Logistic Regression classifier on patient clinical data and exposes it through a simple REST API. Send a patient's clinical parameters as JSON and get back a risk prediction.

## Features

- Data preprocessing and exploratory checks with Pandas
- Logistic Regression classifier trained with scikit-learn
- Model persisted with Pickle
- Flask REST API with JSON request/response
- Optional web form for manual testing
- Ready to deploy on Render with `gunicorn`

## Tech Stack

- Python 3
- Pandas / scikit-learn
- Flask + Gunicorn
- Render (hosting)

## Project Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl               # Trained model (Pickle)
├── model_meta.pkl          # Feature order + algorithm name
├── requirements.txt
├── Procfile                # Render/gunicorn start command
├── README.md
├── train_model.py          # Data preprocessing + model training
├── heart.csv                # Dataset
├── templates/
│   └── index.html          # Optional manual-test web form
└── static/
```

## Dataset

14 columns, 1025 rows:

| Column | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia (0–3) |
| target | 1 = heart disease, 0 = no heart disease |

**Note on data quality:** this dataset repeats the original 303-patient UCI records roughly 3x to pad it out to 1025 rows, resulting in 723 exact duplicate rows (only 302 unique patients). `train_model.py` removes duplicates before splitting into train/test sets, so the reported accuracy reflects genuine generalization rather than leakage from duplicate patients appearing in both sets.

## Model

Logistic Regression, trained on the de-duplicated data (302 unique records) with an 80/20 train/test split.

**Test accuracy: 80.3%**

The trained model is saved to `model.pkl`; feature order and algorithm name are saved to `model_meta.pkl` so the API loads inputs consistently.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/HeartDiseaseDeployment.git
cd HeartDiseaseDeployment
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

This loads `heart.csv`, preprocesses it, trains the Logistic Regression model, and saves `model.pkl` and `model_meta.pkl`.

### Run the API locally

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

## API Reference

| Route | Method | Description |
|---|---|---|
| `/` | GET | Web form for manual testing |
| `/health` | GET | Health check |
| `/predict` | POST | Accepts patient JSON, returns a prediction |

### Example request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 62, "sex": 1, "cp": 2, "trestbps": 113, "chol": 249,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 1,
        "oldpeak": 0.9, "slope": 2, "ca": 0, "thal": 2
      }'
```

### Example response

```json
{
  "prediction": "Heart Disease Detected",
  "confidence": 0.53
}
```

## Deployment

This project is configured to deploy on [Render](https://render.com):

1. Push the repository to GitHub.
2. On Render, create a **New Web Service** and connect the repository.
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set in `Procfile`)
   - **Environment:** Python 3
4. Deploy. Render assigns a public URL, e.g. `https://heart-disease-deployment.onrender.com`.

## Conclusion

Logistic Regression was chosen for this classification task and achieved 80% accuracy on unseen patient data after correcting a data quality issue: the raw Kaggle CSV contained 723 duplicate rows, which would have leaked test patients into the training set and produced a misleadingly perfect score. After de-duplicating to 302 genuine patient records and evaluating on a proper held-out split, Logistic Regression proved a solid, interpretable choice for a clinical setting — its coefficients can be inspected to see which factors (e.g. chest pain type, ST depression) drive risk, which matters for trust in healthcare applications. The main challenges during development involved catching that data leakage before it silently inflated confidence in the model, keeping the API's input feature order perfectly aligned with training, handling malformed JSON gracefully, and configuring a production-ready start command (`gunicorn`) since Flask's built-in server isn't meant for live traffic. This project is a good example of why MLOps matters: a model is only trustworthy once its data pipeline, evaluation, and serving layer are all treated with the same rigor as the algorithm itself. Practices like validating for leakage, saving model metadata alongside the model, and using a reproducible `requirements.txt` are what separate a notebook experiment from a dependable, maintainable application.

## License

This project is available for educational use.
