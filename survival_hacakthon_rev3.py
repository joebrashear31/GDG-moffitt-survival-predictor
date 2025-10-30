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
# Using Coxnet
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sksurv.metrics import concordance_index_censored
from sklearn.base import TransformerMixin, BaseEstimator

# Define list of strings to treat as NaN during load
NAN_VALUES = ['NX', 'NA', 'N/A', 'NaN', 'None', '?']

# Define a list of all known original clinical and survival columns
ORIGINAL_CLINICAL_COLUMNS = [
    # Numerical
    'age_at_diagnosis', 'days_to_treatment',
    # Categorical
    'gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade',
    'classification_of_tumor', 'tissue_origin', 'laterality', 
    'prior_malignancy', 'synchronous_malignancy', 'disease_response', 
    'treatment_types', 'therapeutic_agents', 'treatment_outcome',
    # Text (Removed for stability)
    'pathology_report',
    # Identifier
    'patient_id',
    # Survival & Meta (these are dropped later)
    'overall_survival_days', 'overall_survival_event', 
    'days_to_death', 'days_to_last_followup', 'cause_of_death',
    'days_to_progression', 'days_to_recurrence', 
    'progression_or_recurrence', 'vital_status'
]

# Custom transformer to explicitly convert sparse matrices to dense NumPy arrays
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

# Function to load and merge image features
def load_and_merge_image_features(df, data_dir, is_train):
    """Loads a pre-extracted image feature CSV and merges it with the main dataframe."""
    
    subset_dir = "train" if is_train else "test"
    feature_file = os.path.join(data_dir, subset_dir, "image_features.csv")

    if os.path.exists(feature_file):
        try:
            df_features = pd.read_csv(feature_file) 
            
            if 'patient_id' in df_features.columns:
                df_features = df_features.set_index('patient_id')
            else:
                 df_features = df_features.rename(columns={df_features.columns[0]: 'patient_id'}).set_index('patient_id')
                 
            df = df.set_index('patient_id').join(df_features, how='left')
            df = df.reset_index()
            
            new_cols_count = len(df_features.columns)
            print(f"✅ Merged {new_cols_count} image features from {os.path.basename(feature_file)}.")
        except Exception as e:
            print(f"❌ Warning: Could not load or merge image features from {feature_file}. Error: {e}")
    else:
        print(f"⚠️ Warning: Image feature file not found at {feature_file}. Skipping image feature loading.")

    return df

def generate_submission(data_dir):
    
    # --- 1. Define File Paths and Load Data ---
    TRAIN_CSV = os.path.join(data_dir, "train", "train.csv")
    TEST_CSV = os.path.join(data_dir, "test", "test.csv")
    OUTPUT_FILE = "hackathon_submission_final.csv"

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"❌ Error: Could not find '{os.path.basename(TRAIN_CSV)}' or '{os.path.basename(TEST_CSV)}'")
        return

    df_train = pd.read_csv(TRAIN_CSV, na_values=NAN_VALUES)
    df_test = pd.read_csv(TEST_CSV, na_values=NAN_VALUES)
    print(f"✅ Data loaded successfully. Training CSV from: {TRAIN_CSV}")

    # Load and merge image features for both train and test sets
    df_train = load_and_merge_image_features(df_train, data_dir, is_train=True)
    df_test = load_and_merge_image_features(df_test, data_dir, is_train=False)
    
    def create_survival_array(df):
        dt = np.dtype([('event', np.bool_), ('time', np.float64)])
        y = np.array(list(zip(df['overall_survival_event'].astype(bool), 
                              df['overall_survival_days'])), dtype=dt)
        return y

    y_train = create_survival_array(df_train)
    
    # CRITICAL: Drop survival columns AND pathology_report
    features_to_drop = [col for col in ORIGINAL_CLINICAL_COLUMNS if col.endswith('_days') or col.endswith('_event') or col in ['days_to_death', 'days_to_last_followup', 'cause_of_death', 'progression_or_recurrence', 'vital_status', 'pathology_report']] 

    X_train = df_train.drop(columns=[col for col in features_to_drop if col in df_train.columns], errors='ignore')
    X_test = df_test.copy().drop(columns=['pathology_report'], errors='ignore')

    # --- 2. Preprocessing and Modeling Pipeline (Clinical + Image Features ONLY) ---
    
    current_columns = set(X_train.columns)
    original_base_columns = set(
        col for col in ORIGINAL_CLINICAL_COLUMNS 
        if col not in features_to_drop and col in X_train.columns
    )
    
    image_features = sorted(list(current_columns - original_base_columns))
    print(f"🔍 Identified {len(image_features)} image features for processing.")
    print("❌ Text features have been intentionally removed to prevent overfitting.")


    numerical_features = ['age_at_diagnosis', 'days_to_treatment'] + image_features
    
    categorical_features = [
        'gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade',
        'classification_of_tumor', 'tissue_origin', 'laterality', 
        'prior_malignancy', 'synchronous_malignancy', 'disease_response', 
        'treatment_types', 'therapeutic_agents', 'treatment_outcome'
    ]
    
    # Define Numerical Pipeline 
    numerical_transformer = Pipeline(steps=[
        ('to_numeric_coerce', FunctionTransformer(
            lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors='coerce'), validate=False
        )),
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', RobustScaler()) 
    ])

    # Define Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine all feature pipelines into the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            # TEXT TRANSFORMER IS INTENTIONALLY OMITTED
        ],
        remainder='drop' 
    )

    # Build the final pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dense', DenseTransformer()), 
        ('variance_filter', VarianceThreshold(threshold=0.001)), 
        # 💥 TARGETED PENALTY: Alpha=0.2 (Optimal Generalization)
        ('coxph_regularized', CoxnetSurvivalAnalysis( 
            alphas=[0.2], 
            l1_ratio=0.5, # Elastic Net
            tol=1e-5
        ))
    ])

    print("⏳ Training Lightly Penalized Coxnet Model (Clinical + Image Only, Alpha=0.2)...")
    model_pipeline.fit(X_train, y_train) 
    print("✅ Model trained successfully. Check your C-Index!")

    # --- 3. Local C-Index Check ---
    train_risk_scores = model_pipeline.predict(X_train)
    c_index_estimate = concordance_index_censored(y_train['event'], y_train['time'], train_risk_scores)

    print(f"================================================================")
    print(f"⭐️ LOCAL TRAINING C-INDEX (Targeted Penalty Coxnet): {c_index_estimate[0]:.4f}")
    print(f"================================================================")
    
    # --- 4. Generate Predictions and Save Submission ---
    raw_risk_scores = model_pipeline.predict(X_test)
    risk_scores = raw_risk_scores * -1 

    submission_df = pd.DataFrame({
        'patient_id': X_test['patient_id'],
        'predicted_scores': risk_scores 
    })

    submission_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Submission file generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an aggressively penalized Coxnet model for the hackathon."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help="The path to the *base* directory containing the 'train/' and 'test/' subdirectories."
    )
    
    args = parser.parse_args()
    expanded_data_dir = os.path.expanduser(args.data_dir)
    
    generate_submission(expanded_data_dir)