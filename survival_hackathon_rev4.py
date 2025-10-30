import pandas as pd
import numpy as np
import os
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis 

# Define list of strings to treat as NaN during load
NAN_VALUES = ['NX', 'NA', 'N/A', 'NaN', 'None', '?']

# --- 1. Feature Engineering Function ---
def feature_engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Creates new, informative features based on clinical domain knowledge."""
    X_out = X.copy()

    # 1. Missingness Indicators
    X_out['days_to_treatment_missing'] = X_out['days_to_treatment'].isnull().astype(int)
    X_out['age_at_diagnosis_missing'] = X_out['age_at_diagnosis'].isnull().astype(int)

    # 2. Interaction Term (Age * Tumor Grade)
    X_out['tumor_grade'] = pd.to_numeric(X_out['tumor_grade'], errors='coerce')
    age_median = X_out['age_at_diagnosis'].median()
    grade_fill = X_out['tumor_grade'].fillna(1.0) 
    age_imputed_for_int = X_out['age_at_diagnosis'].fillna(age_median)
    X_out['age_grade_interaction'] = age_imputed_for_int * grade_fill

    return X_out

# --- 2. Data Loading and Merging Function ---
def load_data(data_dir, data_type='train'):
    """Loads, merges clinical and image features, and preprocesses survival outcome."""
    
    CLINICAL_FILE = os.path.join(data_dir, data_type, f'{data_type}.csv')
    IMAGE_FILE = os.path.join(data_dir, data_type, 'image_features.csv')

    clinical_df = pd.read_csv(CLINICAL_FILE, na_values=NAN_VALUES)
    clinical_df.columns = clinical_df.columns.str.lower().str.replace(' ', '_').str.strip()

    image_df = pd.read_csv(IMAGE_FILE)
    image_df.columns = image_df.columns.str.lower().str.replace(' ', '_').str.strip()
    image_df = image_df.rename(columns={'id': 'patient_id'})

    data_df = clinical_df.merge(image_df, on='patient_id', how='left')
    
    image_features = [col for col in data_df.columns if col.startswith('feature_')]

    X = data_df.copy()

    if data_type == 'train':
        y = np.array([
            (event, time)
            for event, time in zip(
                X['overall_survival_event'].fillna(0).astype(bool), 
                X['overall_survival_days'].fillna(X['days_to_last_followup'])
            )
        ], dtype=[('event', 'bool'), ('time', 'float64')])
        
        X = X.drop(columns=['overall_survival_days', 'overall_survival_event',
                            'days_to_death', 'days_to_last_followup', 'cause_of_death'])
        return X, y, image_features
    else:
        X = X.drop(columns=[col for col in ['overall_survival_days', 'overall_survival_event'] if col in X.columns], errors='ignore')
        return X, image_features

# --- Custom Function for Text Pipeline Compatibility ---
def flatten_and_str(X):
    """Flattens the 2D array input from ColumnTransformer and converts all to string."""
    return X.ravel().astype(str)
# --------------------------------------------------------

# --- 3. Preprocessor Setup ---
def get_preprocessor(image_features):
    """Defines the ColumnTransformer for all feature types."""
    
    NUMERICAL_COLS = ['age_at_diagnosis', 'days_to_treatment']
    CATEGORICAL_COLS = ['gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade',
                        'classification_of_tumor', 'tissue_origin', 'laterality',
                        'prior_malignancy', 'synchronous_malignancy', 'disease_response',
                        'treatment_types', 'therapeutic_agents', 'treatment_outcome']
    TEXT_COL = ['pathology_report'] 
    
    NEW_NUM_COL = ['age_grade_interaction']
    NEW_CAT_COLS = ['days_to_treatment_missing', 'age_at_diagnosis_missing'] 

    all_num_cols = NUMERICAL_COLS + NEW_NUM_COL + image_features
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    text_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
        ('to_str', FunctionTransformer(flatten_and_str)), 
        # Reduced max features to 500
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=500)), 
        ('variance_threshold', VarianceThreshold(threshold=1e-3)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, all_num_cols),
            ('cat', cat_pipeline, CATEGORICAL_COLS + NEW_CAT_COLS),
            ('text', text_pipeline, TEXT_COL) 
        ],
        remainder='drop', # Excludes patient_id
    )
    
    return preprocessor

# --- 4. Main Model Pipeline ---
def get_model_pipeline(preprocessor):
    """Defines the final machine learning pipeline using Coxnet (Maximum Forced Regularization)."""
    
    # FINAL FIX: Force the regularization penalty (alpha) to be extremely strong
    coxnet_model = CoxnetSurvivalAnalysis(
        l1_ratio=0.999999,           # Maximum Sparsity/Lasso 
        alpha_min_ratio=0.1,         # CRITICAL: Force the CV to search for a much stronger penalty
        n_alphas=100,                
        fit_baseline_model=True,     
        max_iter=10000,
        tol=1e-7,
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', coxnet_model)
    ])
    
    return model_pipeline

# --- 5. Execution Logic ---
def main(data_dir):
    """Loads data, trains the model, and generates a submission file."""
    
    OUTPUT_FILE = 'survival_hackathon_submission_rev22_Max_Forced_Reg.csv'

    print("📚 Loading and Merging Multimodal Data (Clinical + Image)...")
    X_train_raw, y_train, train_image_features = load_data(data_dir, data_type='train')
    X_test_raw, test_image_features = load_data(data_dir, data_type='test')
    
    print("🔧 Running Feature Engineering...")
    X_train = feature_engineer(X_train_raw)
    X_test = feature_engineer(X_test_raw)
    
    image_features_to_use = train_image_features

    # --- 1. Setup and Train Model ---
    preprocessor = get_preprocessor(image_features_to_use)
    model_pipeline = get_model_pipeline(preprocessor)
    
    print(f"\n📈 Training Coxnet Max Forced Reg Model (alpha_min_ratio=0.1) to combat extreme overfitting...")
    model_pipeline.fit(X_train, y_train) 
    print("✅ Model trained successfully.")

    # --- 2. Local C-Index Check (on Training Data) ---
    train_risk_scores = model_pipeline.predict(X_train)
    c_index_estimate = concordance_index_censored(y_train['event'], y_train['time'], train_risk_scores)

    print(f"\n=================================================================")
    print(f"⭐️ LOCAL TRAINING C-INDEX (Maximum Forced Regularization): {c_index_estimate[0]:.4f}")
    print(f"=================================================================")
    
    # --- 3. Generate Predictions and Save Submission ---
    raw_risk_scores = model_pipeline.predict(X_test)
    
    # Invert the risk scores for submission (Higher score = Longer survival)
    survival_scores = raw_risk_scores * -1 

    submission_df = pd.DataFrame({
        'patient_id': X_test['patient_id'],
        'predicted_scores': survival_scores 
    })

    submission_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Submission file generated: {OUTPUT_FILE}. This highly restricted model should finally generalize well.")

# --- Set Submission URL ---
submission_url = "https://huggingface.co/spaces/Lab-Rasool/2025-hackathon-submission"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a high-performance Multimodal Coxnet model."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help="The path to the *base* directory containing the 'train/' and 'test/' subdirectories, where each contains a 'train.csv' (or 'test.csv') and 'image_features.csv'."
    )
    
    args = parser.parse_args()
    main(args.data_dir)