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
# 💥 FINAL FIX: Reverting to Coxnet and controlling its fit process
# This is the correct import for the regularized Cox model
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sksurv.metrics import concordance_index_censored
from sklearn.base import TransformerMixin, BaseEstimator

# Define list of strings to treat as NaN during load (Fixes 'NX', 'NA', etc. errors)
NAN_VALUES = ['NX', 'NA', 'N/A', 'NaN', 'None', '?']

# Custom transformer to explicitly convert sparse matrices to dense NumPy arrays
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

def load_and_merge_image_features(df, data_dir, is_train):
    """Placeholder function for image feature integration."""
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

    # --- 2. Preprocessing and Modeling Pipeline (Multi-Modal) ---
    numerical_features = ['age_at_diagnosis', 'days_to_treatment']
    categorical_features = [
        'gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade',
        'classification_of_tumor', 'tissue_origin', 'laterality', 
        'prior_malignancy', 'synchronous_malignancy', 'disease_response', 
        'treatment_types', 'therapeutic_agents', 'treatment_outcome'
    ]
    text_feature = ['pathology_report']
    
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

    # Define Text Pipeline (TF-IDF Vectorization)
    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
        ('unwrapper', FunctionTransformer(lambda x: x.squeeze().tolist(), validate=False)),
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            norm=None
        )),
        ('text_scaler', RobustScaler(with_centering=False)) 
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
        ('variance_filter', VarianceThreshold(threshold=0.001)), 
        # 💥 FIX APPLIED HERE: Replaced undefined CoxLasso with imported CoxnetSurvivalAnalysis
        ('coxph_regularized', CoxnetSurvivalAnalysis( 
            # Apply fixed alpha for stability
            alpha=1.0, 
            tol=1e-5
        ))
    ])

    print("⏳ Training Multi-Modal Robust CoxLasso model...")
    model_pipeline.fit(X_train, y_train) 
    print("✅ Model trained successfully. Check your C-Index!")

    # --- 3. Local C-Index Check ---
    train_risk_scores = model_pipeline.predict(X_train)
    c_index_estimate = concordance_index_censored(y_train['event'], y_train['time'], train_risk_scores)

    print(f"================================================================")
    print(f"⭐️ LOCAL TRAINING C-INDEX (CoxLasso Fit): {c_index_estimate[0]:.4f}")
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
        description="Train a CoxPH model for the hackathon and generate a CSV submission file."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help="The path to the *base* directory containing the 'train/' and 'test/' subdirectories."
    )
    
    args = parser.parse_args()
    expanded_data_dir = os.path.expanduser(args.data_dir)
    
    # NOTE: CoxnetSurvivalAnalysis will function as CoxLasso when
    # using a single, fixed alpha value without specifying L1_ratio.
    generate_submission(expanded_data_dir)