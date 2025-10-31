🏆 Multimodal Survival Prediction Hackathon Submission

Overview

This project implements a highly regularized Cox Proportional Hazards model (specifically, `CoxnetSurvivalAnalysis`) designed to predict patient overall survival based on multimodal clinical, image, and text features.

The core challenge of this project was overcoming severe overfitting caused by the high dimensionality of the text (TF-IDF) and image features relative to the number of patient samples. The final model (Rev 26) achieved a stable C-index by aggressively forcing regularization and stability.

Final Configuration Strategy (Rev 26):

Parameter

Value

Rationale

`l1_ratio`

`0.95` (Elastic Net)

Maximize feature selection (Lasso) while injecting 5% Ridge (L2) stability to handle highly correlated multimodal features.

`alphas`

`[0.01]` (Fixed)

Use a fixed, weaker penalty strength to prevent underfitting (which occurred with $\alpha=0.2$) while maintaining the stability of a non-searching model.

Text Features

Max `500`

Drastically reduced feature count from the `pathology_report` to curb overfitting.

🚀 Usage

Prerequisites

Data: You must have the required hackathon data structure:

```
/data_dir
├── /train
│   ├── train.csv
│   └── image_features.csv
└── /test
├── test.csv
└── image_features.csv
```

Dependencies: Ensure all required Python libraries are installed. This project relies heavily on `scikit-survival` (`sksurv`), `scikit-learn`, `pandas`, and `numpy`.

```
pip install pandas numpy scikit-learn scikit-survival
```

Execution

Run the final model script (`survival_hacakthon_rev26.py`) from your terminal, providing the path to your base data directory using the `--data_dir` argument.

```
python survival_hacakthon_rev26.py --data_dir /path/to/your/data/
```

Output

The script will perform the following steps:

Load and merge clinical, image, and text data.

Apply advanced feature engineering (e.g., interaction terms and missingness indicators).

Train the highly penalized Coxnet model.

Print the local training C-Index for verification.

Generate a submission file named `survival_hackathon_submission_rev26_Fixed_Alpha_Stability.csv` in your current directory.

⚙️ Model Architecture

The model uses a `Pipeline` structure to ensure deterministic, non-leaky processing:

1. Data Loading & Feature Engineering

Loads and merges all three data sources (Clinical, Image, Text).

Applies domain-specific feature engineering (Age-Tumor Grade interaction, Missingness indicators).

2. Preprocessing (ColumnTransformer)

A single `ColumnTransformer` handles heterogeneous data types:

Feature Type

Pipeline Steps

Purpose

Numerical (Clinical/Image)

`SimpleImputer(median)` -> `RobustScaler`

Fills missing values and scales features robustly against outliers.

Categorical (Clinical)

`SimpleImputer(constant)` -> `OneHotEncoder`

Fills missing categories and converts categories into numerical columns.

Text (`pathology_report`)

`TfidfVectorizer` (max 500 features)

Converts raw text into numerical features, heavily restricted to limit dimensionality.

3. Model Training

`DenseTransformer`: Converts the output of the `ColumnTransformer` (which includes sparse TF-IDF data) into a dense array, which is required by `CoxnetSurvivalAnalysis`.

`CoxnetSurvivalAnalysis`: A regularized Cox model that performs penalized regression to select and shrink coefficients.