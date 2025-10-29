import pandas as pd
import numpy as np
import os
import argparse
from sklearn.pipeline import Pipeline
# NEW IMPORT FOR TEXT PROCESSING
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# MODEL IMPORTS
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sksurv.metrics import concordance_index_censored 
from sklearn.base import TransformerMixin, BaseEstimator 

# Custom transformer to explicitly convert sparse matrices to dense NumPy arrays
class DenseTransformer(BaseEstimator, TransformerMixin):
    """Transformer to force a sparse matrix output (like from ColumnTransformer)
    into a dense NumPy array."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Only call .toarray() if the input is a sparse matrix
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

def generate_submission(data_dir):
    """
    Loads data, trains a stable multi-modal CoxPH model with max features, and generates submission.
    """
    
    # --- 1. Define File Paths ---
    TRAIN_DIR = os.path.join(data_dir, "train")
    TEST_DIR = os.path.join(data_dir, "test")
    TRAIN_CSV = os.path.join(TRAIN_DIR, "train.csv")
    TEST_CSV = os.path.join(TEST_DIR, "test.csv")
    OUTPUT_FILE = "hackathon_submission_final.csv"

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"❌ Error: Could not find '{os.path.basename(TRAIN_CSV)}' or '{os.path.basename(TEST_CSV)}'")
        return

    # --- 2. Load Data ---
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    print(f"✅ Data loaded successfully. Training CSV from: {TRAIN_CSV}")

    # FIX: Explicitly cast the pathology report column to string type (str)
    df_train['pathology_report'] = df_train['pathology_report'].astype(str)
    df_test['pathology_report'] = df_test['pathology_report'].astype(str)

    def create_survival_array(df):
        dt = np.dtype([('event', np.bool_), ('time', np.float64)])
        y = np.array(list(zip(df['overall_survival_event'].astype(bool), 
                              df['overall_survival_days'])), dtype=dt)
        return y

    y_train = create_survival_array(df_train)
    
    features_to_drop = ['overall_survival_days', 'overall_survival_event', 
                        'days_to_death', 'days_to_last_followup', 'cause_of_death',
                        'days_to_progression', 'days_to_recurrence', 
                        'progression_or_recurrence', 'vital_status']

    X_train = df_train.drop(columns=[col for col in features_to_drop if col in df_train.columns])
    X_test = df_test.copy()

    # --- 3. Preprocessing and Modeling Pipeline (Multi-Modal) ---
    numerical_features = ['age_at_diagnosis', 'days_to_treatment']
    categorical_features = [
        'gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade',
        'classification_of_tumor', 'tissue_origin', 'laterality', 
        'prior_malignancy', 'synchronous_malignancy', 'disease_response', 
        'treatment_types', 'therapeutic_agents', 'treatment_outcome'
    ]
    text_feature = ['pathology_report']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Define Text Pipeline (TF-IDF Vectorization)
    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
        ('flattener', FunctionTransformer(lambda x: x.ravel(), validate=False)),
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000 
        ))
    ])

    # Combine all feature pipelines into the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('txt', text_transformer, text_feature)
        ],
        remainder='drop' 
    )

    # Build the final pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dense', DenseTransformer()), 
        # MODEL: CoxnetSurvivalAnalysis with a small, stabilizing alpha
        ('coxph', CoxnetSurvivalAnalysis(
            l1_ratio=1.0, 
            # FIX: Set a small, non-zero alpha to prevent numerical error
            alphas=[0.001]
        ))
    ])

    print("⏳ Training Multi-Modal Stabilized CoxPH model with maximum features...")
    model_pipeline.fit(X_train, y_train) 
    print("✅ Model trained successfully.")
    
    # --- 4. Local C-Index Estimate ---
    # Coxnet outputs the log-hazard ratio (a RISK SCORE).
    train_risk_scores = model_pipeline.predict(X_train)
    
    # The local C-Index calculation expects a RISK SCORE (high = bad).
    estimate_for_local_cindex = train_risk_scores

    # Extract required components from the structured array y_train
    event_indicator = y_train['event']
    event_time = y_train['time']

    # Calculate the C-Index
    c_index_estimate = concordance_index_censored(event_indicator, event_time, estimate_for_local_cindex)

    print(f"================================================================")
    print(f"⭐️ LOCAL C-INDEX ESTIMATE (on Training Data): {c_index_estimate[0]:.4f}")
    print(f"================================================================")
    
    # --- 5. Generate Predictions for Submission ---
    raw_risk_scores = model_pipeline.predict(X_test)

    # CRITICAL: INVERT the scores for the final submission CSV, as the platform expects 
    # high scores to mean LONG survival (low risk).
    risk_scores = raw_risk_scores * -1 

    # --- 6. Format and Save Submission File ---
    submission_df = pd.DataFrame({
        'patient_id': X_test['patient_id'],
        'predicted_scores': risk_scores 
    })

    submission_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Submission file generated: {OUTPUT_FILE}")
    print("--- Submission Head ---")
    print(submission_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a survival model for the hackathon and generate a CSV submission file."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help="The path to the *base* directory containing the 'train/' and 'test/' subdirectories."
    )
    
    args = parser.parse_args()
    
    # FIX: Expand the tilde (~) to the full user home directory
    expanded_data_dir = os.path.expanduser(args.data_dir)
    
    generate_submission(expanded_data_dir)