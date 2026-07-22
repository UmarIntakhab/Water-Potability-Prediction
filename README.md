# 💧 Water Potability Prediction using Machine Learning

## 📌 Project Overview

This project aims to predict whether water is **safe for human consumption (Potable)** or **unsafe (Non-Potable)** using Machine Learning. Various water quality parameters such as pH, Hardness, Sulfate, Conductivity, Turbidity, and others are used to train multiple classification models.

The project includes:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Handling Missing Values
- Handling Class Imbalance using SMOTE
- Model Training & Evaluation
- Hyperparameter Tuning
- Feature Importance Analysis
- Streamlit Web Application Deployment

---

# 📂 Dataset Information

The dataset contains **3276 water samples** with **9 water quality features** and **1 target variable**.

### Features

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

### Target Variable

- **0 → Non-Potable**
- **1 → Potable**

---

# 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset Overview
- Missing Value Analysis
- Statistical Summary
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Target Variable Distribution

### Key Findings

- Missing values were found in **pH**, **Sulfate**, and **Trihalomethanes**.
- Most features are approximately normally distributed.
- The dataset is moderately imbalanced.
- No feature showed strong correlation with the target variable.
- Outliers appear to represent genuine water quality measurements and were retained.

---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Missing values filled using Mean Imputation
- Train-Test Split
- Class balancing using **SMOTE**
- Feature Scaling using **StandardScaler**

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

# 🔄 Cross Validation

Cross Validation was performed to evaluate model stability before hyperparameter tuning.

---

# 🎛 Hyperparameter Tuning

The best-performing models were optimized using:

- GridSearchCV
- RandomizedSearchCV

Optimized Models:

- Random Forest
- Support Vector Machine
- XGBoost

---

# 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Confusion Matrix
- Classification Report

---

# ⭐ Best Model

The **Random Forest Classifier** achieved the best overall performance after hyperparameter tuning.

Performance was evaluated based on:

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation Score

---

# 🌟 Feature Importance

Feature importance was analyzed using the Random Forest model to identify the most influential water quality parameters affecting potability prediction.

---

# 💻 Streamlit Web Application

A Streamlit application was developed to allow users to predict water potability by entering water quality parameters.

### Features

- User-friendly interface
- Manual feature input
- Sample Potable Data
- Sample Non-Potable Data
- Prediction Confidence

---

# 📁 Project Structure

Water Potability Project
│
├── analysis.ipynb
├── app.py
├── water_potability.csv
├── scaler.pkl
├── water_potability_model.pkl
├── requirements.txt
├── README.md
│
├── images
│   ├── Boxplots of Water Quality Features.png
│   ├── Confusion_matrices.png
│   ├── Correlation Analysis of All Features.png
│   ├── Distribution of 'ph', 'Sulfate', 'Trihalomethanes' after mean imputation.png
│   ├── Distribution of Additional Water Quality Features by Potability.png
│   ├── Distribution of Selected Water Quality Features by Potability.png
│   ├── Distribution of Water Quality Features.png
│   ├── Feature_importance - Random Forest.png
│   └── roc_curves.png
│
└── WaterPotability_Report.docx


## ▶️ How to Run the Project

### Step 1: Download or Copy the Project Folder

Ensure the project folder contains the following files:

```
analysis.ipynb
app.py
water_potability.csv
water_potability_model.pkl
scaler.pkl
requirements.txt
```

---

### Step 2: Install the Required Libraries

Open a terminal or command prompt inside the project folder and run:

```bash
pip install -r requirements.txt
```

This will install all the required Python libraries.

---

### Step 3: Run the Streamlit Application

In the terminal, execute:

```bash
streamlit run app.py
```

---

### Step 4: Open the Application

After running the above command, Streamlit will automatically open the application in your default web browser.

If it does not open automatically, copy and paste the Local URL displayed in the terminal (usually `http://localhost:8501`) into your browser.

---

### System Requirements

* Python 3.10 or later
* Internet connection (only required for installing libraries)
* A modern web browser (Chrome, Edge, Firefox, etc.)

---

# 📦 Required Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- Joblib
- Streamlit

---

# 🎯 Future Improvements

- Collect more training data
- Improve prediction accuracy using additional features
- Deploy the application to the cloud
- Integrate real-time water quality sensors
- Explore deep learning models for comparison

---

# 👨‍💻 Author

**Umar**

This project was developed as part of the **Machine Learning & AI Diploma Program** at **BIA Institute**.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, model development, evaluation, hyperparameter tuning, and deployment using Streamlit.