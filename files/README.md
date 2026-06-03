# Diabetes Risk Prediction System

A clinical decision-support dashboard built with Streamlit.  
Trained on 96,128 patient records using Random Forest + SMOTE.

## Project Structure

```
diabetes_app/
├── app.py                          # Main Streamlit application
├── cleaned_diabetes_dataset.csv    # Dataset (place in root directory)
├── requirements.txt                # Python dependencies
└── README.md
```

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repository**
   - The repo must contain `app.py`, `requirements.txt`, and `cleaned_diabetes_dataset.csv`

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **New app** → select your repo → set:
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Click **Deploy** — Streamlit Cloud installs dependencies automatically

## Features

| Section | Content |
|---------|---------|
| 01 — Input Panel | Age, BMI, HbA1c, Blood Glucose, Hypertension, Heart Disease, Smoking History |
| 02 — Predict | Single-click risk prediction |
| 03 — Output | Risk level badge, probability %, visual gauge (green/yellow/red) |
| 04 — Feature Impact | Personalised bar chart explaining the prediction |
| 05 — Recommendations | Clinical health advice based on patient values |
| Model Summary | Accuracy, F1, AUC, Precision, Recall chips |

## Model

- **Algorithm:** Random Forest (200 trees, max_depth=10)
- **Imbalance handling:** SMOTE on training set only
- **Train/test split:** 80/20 stratified
- **Target:** `diabetes` (binary: 0 = negative, 1 = positive)
