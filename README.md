# Water_Quality_Classification


### Water Quality Classification

Predicting drinking water potability using Machine Learning — built as a Data Mining course project .


### Project Overview

This project tackles a binary classification problem: given physicochemical measurements of water samples, can we predict whether the water is safe to drink (potable) or not?
The full ML pipeline covers everything from raw data exploration to hyperparameter-tuned models, with a feature engineering step grounded in WHO water quality standards.

### Dataset
Source: Kaggle – Water Potability Dataset
File: water_potability.csv
Samples: 3,276 water samples
Target: Potability (0 = Not Potable, 1 = Potable)
Missing values: ph (15%), Sulfate (24%), Trihalomethanes (5%)

### Original Features

ph — pH value (WHO safe range: 6.5–8.5)
Hardness — Capacity of water to precipitate soap (mg/L)
Solids — Total dissolved solids (ppm)
Chloramines — Amount of chloramines (ppm)
Sulfate — Amount of sulfates dissolved (mg/L)
Conductivity — Electrical conductivity (μS/cm)
Organic_carbon — Amount of organic carbon (ppm)
Trihalomethanes — Amount of trihalomethanes (μg/L)
Turbidity — Measure of light emitting property (NTU)


### Pipeline
Raw Data → EDA → Data Cleaning → Feature Engineering → Feature Selection → Train/Test Split → Scaling → Model Training (GridSearchCV) → Evaluation
##### 1.  EDA

Class distribution analysis (imbalanced: ~61% Not Potable / ~39% Potable)
Correlation heatmap of all features

##### 2.  Data Cleaning
Missing values filled using median per Potability class — smarter than global median since potable and non-potable water have different distributions.
##### 3.  Feature Engineering
4 new features derived from domain knowledge:

ph_deviation = abs(ph - 7.0) — pH far from neutral 7.0 means higher contamination risk (WHO range: 6.5–8.5)
mineral_load = (Hardness + Solids) / 2 — combined mineral hardness index
chem_risk = Chloramines × Trihalomethanes / 100 — chemical toxicity proxy from disinfection byproducts interaction
turbidity_organic_ratio = Turbidity / (Organic_carbon + 1) — high turbidity with low organic carbon signals contamination

##### 4.  Models Trained
All models tuned with 5-Fold Stratified GridSearchCV:

Logistic Regression — tuned C and solver
Decision Tree — tuned max_depth, min_samples_split, criterion
Random Forest — tuned n_estimators, max_depth, min_samples_split
SVM — tuned C, kernel, gamma


###  Results

Random Forest achieved the best overall performance.

Random Forest — Accuracy ~0.69 | Precision ~0.62 | Recall ~0.56 | F1 ~0.59
SVM — Accuracy ~0.67 | Precision ~0.60 | Recall ~0.52 | F1 ~0.56
Logistic Regression — Accuracy ~0.62 | Precision ~0.53 | Recall ~0.43 | F1 ~0.47
Decision Tree — Accuracy ~0.61 | Precision ~0.52 | Recall ~0.49 | F1 ~0.50


### Project Structure

├── Water_Qulaity_Classification.ipynb   # Full ML pipeline notebook
├── app.py                               # Streamlit web application
├── water_potability.csv                 # Dataset
├── models/
   ├── scaler.pkl                       # Saved StandardScaler
   ├── feature_cols.pkl                 # Saved feature list
   ├── Logistic_Regression.pkl
   ├── Decision_Tree.pkl
   ├── Random_Forest.pkl
   └── SVM.pkl
