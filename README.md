# Water Potability Prediction

A machine learning pipeline that predicts whether a water sample is safe to drink (potable) from nine water-quality measurements. It covers cleaning, class-imbalance handling with SMOTE, comparison of five classifiers, hyperparameter tuning, and a Streamlit app for predictions.

## Key result

Final model: **Random Forest (tuned)**, evaluated on a stratified held-out test set of 656 samples.

| Metric | Value |
|---|---:|
| Accuracy | 0.637 |
| Precision (potable) | 0.539 |
| Recall (potable) | 0.484 |
| F1 (potable) | 0.510 |

Tuned models on the same test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.637** | 0.539 | 0.484 | 0.510 |
| XGBoost | 0.627 | 0.523 | 0.488 | 0.505 |
| SVM (RBF) | 0.578 | 0.464 | 0.527 | 0.494 |

> **Context:** this dataset has a weak signal. No single feature has a correlation above 0.1 with potability, so every model lands in the low-to-mid 60s for accuracy. The goal here is a sound, leak-aware workflow rather than a high headline number.

## Dataset

- 3,276 samples, 9 features plus the target `Potability` (1 = potable, 0 = not).
- Features: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`.
- Class balance: 61% non-potable, 39% potable.
- Missing values: `ph` (491), `Sulfate` (781), `Trihalomethanes` (162). No duplicates.

## What the project does

1. **EDA:** feature distributions, boxplots, class-wise comparisons and a correlation heatmap. Outliers were kept because they plausibly reflect real extreme measurements.
2. **Preprocessing:** mean imputation for the three columns with missing values (distributions were checked before and after), stratified train/test split, `StandardScaler` fit on the training data only.
3. **Imbalance handling:** SMOTE applied to the training set only, after the split, so no synthetic samples reach the test set.
4. **Modelling:** KNN, Decision Tree, SVM, Random Forest and XGBoost; the top three were tuned with `GridSearchCV` (Random Forest, SVM) and `RandomizedSearchCV` (XGBoost).
5. **Evaluation:** accuracy, precision, recall, F1, confusion matrices and ROC curves on the held-out test set.
6. **Feature importance:** `Solids`, `Sulfate` and `pH` rank highest in the Random Forest; `Conductivity` and `Turbidity` rank lowest.
7. **Deployment:** the trained model and scaler are saved with `joblib` and used by a Streamlit app.

## SMOTE vs. no SMOTE

| Tuned model | Setup | Accuracy | Recall (potable) | F1 |
|---|---|---:|---:|---:|
| Random Forest | with SMOTE | 0.637 | 0.484 | 0.510 |
| Random Forest | without SMOTE | 0.654 | 0.328 | 0.425 |
| XGBoost | with SMOTE | 0.627 | 0.488 | 0.505 |
| XGBoost | without SMOTE | 0.659 | 0.324 | 0.426 |

SMOTE costs a little accuracy and precision but raises recall on potable samples from about 0.33 to about 0.49 and lifts F1 by roughly 0.08. For a water-safety use case, recall and the type of error matter more than raw accuracy.

## Limitations

- **Weak features.** The measurements only loosely separate potable from non-potable samples, which caps what any model can achieve here.
- **Tuning did not improve test accuracy.** The default Random Forest scored 0.652 accuracy on the test set, slightly above the tuned model's 0.637. The tuned model was kept because it was selected on cross-validation and has stronger precision/recall balance.
- **Cross-validation was run on SMOTE-resampled training data**, which lets synthetic samples leak across folds and makes CV scores (about 0.70) optimistic. The held-out test figures above are the reliable ones. A cleaner setup would apply SMOTE inside each fold with an `imblearn` Pipeline.
- A single train/test split on 3,276 samples gives noisy estimates; differences of a point or two between models are not meaningful.

## Future work

- Move SMOTE inside a cross-validation Pipeline and report cross-validated metrics from it.
- Tune the decision threshold for higher recall on potable samples.
- Try gradient boosting variants and report confidence intervals via repeated cross-validation.

## How to run

```bash
git clone https://github.com/UmarIntakhab/water-potability-prediction.git
cd water-potability-prediction
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells top to bottom. It reads `water_potability.csv` from the same folder and saves the model (`water_potability_model.pkl`) and scaler (`scaler.pkl`).

To launch the Streamlit app (if `app.py` is in the repo):

```bash
streamlit run app.py
```

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit · Matplotlib · Seaborn · Joblib

## Author

**Umar Intakhab**: BCA graduate, Jamia Hamdard University.
Contact: umarintakhab1976@gmail.com
