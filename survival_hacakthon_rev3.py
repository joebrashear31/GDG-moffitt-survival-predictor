import pandas as pd
import numpy as np
import os
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# STABLE MODEL IMPORT
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sksurv.metrics import concordance_index_censored 
from sklearn.base import TransformerMixin, BaseEstimator 

# Define list of strings to treat as NaN during load
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
    """
    Loads the image_features.csv and merges it into the main DataFrame.
    Returns the original DataFrame if the file is not found or an error occurs.
    """
    
    subdir = "train" if is_train else "test"
    features_path = os.path.join(data_dir, subdir, "image_features.csv")
    
    if not os.path.exists(features_path):
        print(f"⚠️ Warning: Image feature file not found at {features_path}. Proceeding without image integration.")
        return df

    try:
        df_img = pd.read_csv(features_path)
        if 'patient_id' not in df_img.columns:
            df_img.rename(columns={df_img.columns[0]: 'patient_id'}, inplace=True)

        df_merged = df.merge(df_img, on='patient_id', how='left')
        print(f"✅ Successfully merged {df_merged.shape[1] - df.shape[1]} Pathomics features.")
        return df_merged
        
    except Exception as e:
        print(f"❌ Error loading or merging image features: {e}. Proceeding without image integration.")
        return df

def generate_submission(data_dir):
    
    # --- 1. Define File Paths ---
    TRAIN_DIR = os.path.join(data_dir, "train")
    TEST_DIR = os.path.join(data_dir, "test")
    TRAIN_CSV = os.path.join(TRAIN_DIR, "train.csv")
    TEST_CSV = os.path.join(TEST_DIR, "test.csv")
    OUTPUT_FILE = "hackathon_submission_final.csv"

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"❌ Error: Could not find '{os.path.basename(TRAIN_CSV)}' or '{os.path.basename(TEST_CSV)}'")
        return

    # --- 2. Load and Integrate Data ---
    # FIX: Use na_values to convert known missing placeholders (like 'NX') to NaN on load
    df_train = pd.read_csv(TRAIN_CSV, na_values=NAN_VALUES)
    df_test = pd.read_csv(TEST_CSV, na_values=NAN_VALUES)
    
    # Apply Pathomics Feature Integration
    df_train = load_and_merge_image_features(df_train, data_dir, is_train=True)
    df_test = load_and_merge_image_features(df_test, data_dir, is_train=False)

    df_train['pathology_report'] = df_train['pathology_report'].astype(str)
    df_test['pathology_report'] = df_test['pathology_report'].astype(str)

    def create_survival_array(df):
        dt = np.dtype([('event', np.bool_), ('time', np.float64)])
        return np.array(list(zip(df['overall_survival_event'].astype(bool), 
                              df['overall_survival_days'])), dtype=dt)

    y_train = create_survival_array(df_train)
    
    features_to_drop = ['overall_survival_days', 'overall_survival_event', 'days_to_death', 'days_to_last_followup', 'cause_of_death', 'days_to_progression', 'days_to_recurrence', 'progression_or_recurrence', 'vital_status']

    X_train = df_train.drop(columns=[col for col in features_to_drop if col in df_train.columns])
    X_test = df_test.copy()

    # --- 3. Preprocessing Pipeline ---
    all_features = set(X_train.columns)
    numerical_features = ['age_at_diagnosis', 'days_to_treatment']
    categorical_features = ['gender', 'race', 'ethnicity', 'primary_diagnosis', 'tumor_grade', 'classification_of_tumor', 'tissue_origin', 'laterality', 'prior_malignancy', 'synchronous_malignancy', 'disease_response', 'treatment_types', 'therapeutic_agents', 'treatment_outcome']
    text_feature = ['pathology_report']
    
    known_features = set(numerical_features) | set(categorical_features) | set(text_feature) | set(['patient_id'])
    image_features = list(all_features - known_features) 
    all_numerical_features = numerical_features + image_features
    
    # 💥 CRITICAL FIX: Convert 2D array input to DataFrame and apply pd.to_numeric column-wise
    numerical_transformer = Pipeline(steps=[
        ('to_numeric_coerce', FunctionTransformer(
            lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors='coerce'), validate=False
        )),
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
        ('flattener', FunctionTransformer(lambda x: x.ravel(), validate=False)),
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, all_numerical_features), 
            ('cat', categorical_transformer, categorical_features),
            ('txt', text_transformer, text_feature)
        ],
        remainder='drop' 
    )

    # --- 4. Final Stable Modeling Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dense', DenseTransformer()), 
        ('coxph', CoxnetSurvivalAnalysis(
            l1_ratio=1.0, 
            alphas=[1e-5] 
        ))
    ])

    print("⏳ Training CoxPH model (Tabular + Text + Pathomics if available)...")
    model_pipeline.fit(X_train, y_train) 
    print("✅ Model trained successfully.")
    
    # --- 5. Local C-Index Estimate ---
    train_risk_scores = model_pipeline.predict(X_train)
    c_index_estimate = concordance_index_censored(y_train['event'], y_train['time'], train_risk_scores)

    print(f"================================================================")
    print(f"⭐️ LOCAL TRAINING C-INDEX: {c_index_estimate[0]:.4f}")
    print(f"================================================================")
    
    # --- 6. Generate Submission ---
    raw_risk_scores = model_pipeline.predict(X_test)
    risk_scores = raw_risk_scores * -1 

    submission_df = pd.DataFrame({'patient_id': X_test['patient_id'], 'predicted_scores': risk_scores})
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Submission file generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a survival model for the hackathon and generate a CSV submission file.")
    parser.add_argument('--data_dir', type=str, required=True, help="The path to the *base* directory containing the 'train/' and 'test/' subdirectories.")
    args = parser.parse_args()
    expanded_data_dir = os.path.expanduser(args.data_dir)
    generate_submission(expanded_data_dir)