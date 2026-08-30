import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


def logistic_regression(X_train, y_train, X_test, y_test, weights=None, method_name="Logistic Regression"):
    """Scale numeric features and evaluate a Logistic Regression model on the test set."""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_scaled[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test_scaled[cols_to_scale])
    
    model = LogisticRegression(class_weight=weights, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n{'=' * 20} {method_name} {'=' * 20}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"]))
    print(f"=== ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f} ===")
    return model


def down_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample majority class (0) to match minority class (1) count."""
    count_class_0, count_class_1 = df["Churn"].value_counts()
    df_class_0 = df[df["Churn"] == 0]
    df_class_1 = df[df["Churn"] == 1]
    
    df_class_0_downed = df_class_0.sample(count_class_1, random_state=42)
    df_resampled = pd.concat([df_class_0_downed, df_class_1])
    return df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)


def over_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample minority class (1) with replacement to match majority class (0) count."""
    count_class_0, count_class_1 = df["Churn"].value_counts()
    df_class_0 = df[df["Churn"] == 0]
    df_class_1 = df[df["Churn"] == 1]
    
    # Upsample with replacement = True (duplicates rows so pool never runs out)
    df_class_1_upsampled = df_class_1.sample(count_class_0, replace=True, random_state=42)
    df_resampled = pd.concat([df_class_0, df_class_1_upsampled])
    return df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)


def smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Synthetic Minority Over-sampling Technique (SMOTE).
    Generates synthetic samples by interpolating between k-nearest neighbors:
    x_new = x_i + lambda * (x_neighbor - x_i), where lambda in [0, 1].
    """
    smote_model = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote_model.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Clean, encode, and preprocess the raw customer churn dataset."""
    df = pd.read_csv(filepath)
    
    # Drop customerID (non-predictive unique identifier)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
    
    # Standardize 'No internet service' and 'No phone service' to 'No'
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
    
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({'Male': 1, 'Female': 0})
        
    df_encoded = pd.get_dummies(data=df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True, dtype=int)
    return df_encoded
