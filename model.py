"""
Fast & Optimized Earthquake Damage Prediction Model
---------------------------------------------------
- Uses advanced feature engineering (regional + interaction)
- Balanced training (SMOTE)
- Lightweight high-performance models
- Weighted soft voting ensemble
- Auto-save of trained model and encoders

Runtime: ~10–15 minutes on CPU (full dataset); <5 minutes in quick mode
"""

import pandas as pd
import numpy as np
import warnings, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. Data loading & overview
# -------------------------------------------------------------------------
def load_data(path, quick_mode=True):
    df = pd.read_csv(path)
    print("="*80)
    print(f"Dataset loaded: {df.shape}")
    if quick_mode:
        df = df.sample(frac=0.3, random_state=42)   # use 30% for faster run
        print(f"⚡ Quick mode enabled: reduced to {df.shape}")
    return df

# -------------------------------------------------------------------------
# 2. Preprocessing & feature engineering
# -------------------------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # Fix target labels (shift to 0-based)
    if df["damage_grade"].dtype != "object":
        df["damage_grade"] = df["damage_grade"] - df["damage_grade"].min()

    # Fill missing values
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].fillna(df[col].mode()[0])

    # Regional mean-damage features
    geo_cols = [c for c in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"] if c in df.columns]
    for col in geo_cols:
        region_avg = df.groupby(col)["damage_grade"].mean()
        df[f"{col}_avg_damage"] = df[col].map(region_avg)

    # Target encode categorical features
    cat_cols = [c for c in ["foundation_type","roof_type","ground_floor_type","plan_configuration","position","soil_type"] if c in df.columns]
    te = TargetEncoder(cols=cat_cols)
    df[cat_cols] = te.fit_transform(df[cat_cols], df["damage_grade"])

    # Interaction features
    df["building_age_area_interaction"] = df["age"] * df["area_percentage"]
    df["floor_quake_ratio"] = df["count_floors_pre_eq"] * df["PGA_g"]
    df["age_quake_ratio"] = df["age"] * df["magnitude"]
    df["quake_depth_interaction"] = df["PGA_g"] * df["depth_km"]
    df["energy_distance_ratio"] = (df["PGA_g"] * df["magnitude"]) / (df["epicentral_distance_km"] + 1)

    return df, te

# -------------------------------------------------------------------------
# 3. Train-test split, scale, SMOTE
# -------------------------------------------------------------------------
def prepare_data(df):
    X = df.drop(["damage_grade","building_id"], axis=1, errors="ignore")
    y = df["damage_grade"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
    print(f"After SMOTE: {np.bincount(y_train_bal)}")
    return X_train_bal, X_test_s, y_train_bal, y_test, scaler

# -------------------------------------------------------------------------
# 4. Model training (lightweight versions)
# -------------------------------------------------------------------------
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=20, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=180, max_depth=6, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="mlogloss", n_jobs=-1, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=180, learning_rate=0.05, num_leaves=50,
                                   subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=-1)
    }
    for name, m in models.items():
        print(f"Training {name} ...")
        m.fit(X_train, y_train)
    return models

# -------------------------------------------------------------------------
# 5. Evaluate & ensemble
# -------------------------------------------------------------------------
def evaluate_and_ensemble(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[name] = (acc, f1)
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Weighted soft voting ensemble
    ensemble = VotingClassifier(
        estimators=list(models.items()),
        voting="soft",
        weights=[2,3,2]  # XGBoost strongest
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("="*80)
    print(f"ENSEMBLE: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ensemble_confusion_matrix.png")
    plt.close()
    return ensemble, acc, f1

# -------------------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------------------
def main():
    dataset_path = "02_final_ml_ready_dataset.csv"
    df = load_data(dataset_path, quick_mode=True)
    df, te = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    models = train_models(X_train, y_train)
    ensemble, acc, f1 = evaluate_and_ensemble(models, X_train, y_train, X_test, y_test)

    joblib.dump({
        "ensemble": ensemble,
        "scaler": scaler,
        "target_encoder": te,
        "feature_columns": list(df.drop(['damage_grade','building_id'], axis=1, errors='ignore').columns)
    }, "earthquake_ensemble.pkl")
    print("✅ Model saved to earthquake_ensemble.pkl")
    print(f"Final Accuracy={acc:.4f}, F1={f1:.4f}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

