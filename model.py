import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Upgraded ML pipeline for earthquake damage prediction
# Features added: regional mean-damage, interaction features
# Uses SMOTE, target encoding, weighted soft-voting ensemble and model saving
# Designed for a fast high-performance run (no heavy grid search by default)
# ------------------------------------------------------------------------------

# ----------------------------- 1. Load & Explore ------------------------------

def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['damage_grade'].value_counts()}")
    return df

# ----------------------------- 2. Preprocess ---------------------------------

def preprocess_data(df):
    """Enhanced preprocessing: NA handling, target encoding, engineered features."""
    df = df.copy()

    # Fill numeric missing with median (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'damage_grade' and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Normalize target to 0-based classes
    damage_mapping = {'low': 0, 'medium': 1, 'high': 2}
    if df['damage_grade'].dtype == 'object':
        df['damage_grade'] = df['damage_grade'].map(damage_mapping)
    else:
        df['damage_grade'] = df['damage_grade'] - df['damage_grade'].min()

    # Basic engineered features (existing)
    df['building_age_area_interaction'] = df['age'] * df['area_percentage']
    df['floors_height_ratio'] = df['count_floors_pre_eq'] / (df['height_percentage'] + 1)
    df['seismic_vulnerability'] = df['magnitude'] * df['PGA_g'] / (df['epicentral_distance_km'] + 1)
    df['depth_distance_ratio'] = df['depth_km'] / (df['epicentral_distance_km'] + 1)
    superstructure_cols = [col for col in df.columns if 'has_superstructure' in col]
    if len(superstructure_cols) > 0:
        df['total_superstructure_types'] = df[superstructure_cols].sum(axis=1)
    else:
        df['total_superstructure_types'] = 0

    # Advanced engineered features
    df['height_to_area_ratio'] = df['height_percentage'] / (df['area_percentage'] + 1)
    df['floor_to_height_ratio'] = df['count_floors_pre_eq'] / (df['height_percentage'] + 1)
    # Convert some categorical columns temporarily to codes to create interactions
    for col in ['soil_type', 'foundation_type', 'roof_type', 'ground_floor_type']:
        if col in df.columns:
            df[f'{col}_code'] = df[col].astype('category').cat.codes

    df['soil_foundation_interaction'] = 0
    if 'soil_type_code' in df.columns and 'foundation_type_code' in df.columns:
        df['soil_foundation_interaction'] = df['soil_type_code'] * df['foundation_type_code']

    df['roof_floor_interaction'] = 0
    if 'roof_type_code' in df.columns and 'ground_floor_type_code' in df.columns:
        df['roof_floor_interaction'] = df['roof_type_code'] * df['ground_floor_type_code']

    df['quake_energy_index'] = (df['PGA_g'] * df['magnitude']) / (df['epicentral_distance_km'] + 1)
    df['floor_quake_ratio'] = df['count_floors_pre_eq'] * df['PGA_g']
    df['age_quake_ratio'] = df['age'] * df['magnitude']
    df['quake_depth_interaction'] = df['PGA_g'] * df['depth_km']
    df['energy_distance_ratio'] = (df['PGA_g'] * df['magnitude']) / (df['epicentral_distance_km'] + 1)

    # Regional average damage features
    geo_cols = [c for c in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'] if c in df.columns]
    for col in geo_cols:
        region_avg = df.groupby(col)['damage_grade'].mean()
        df[f'{col}_avg_damage'] = df[col].map(region_avg)

    # Target encoding for categorical variables (safe/fast)
    cat_cols = [c for c in ['foundation_type', 'roof_type', 'ground_floor_type', 'plan_configuration', 'position', 'soil_type'] if c in df.columns]
    if len(cat_cols) > 0:
        te = TargetEncoder(cols=cat_cols)
        df[cat_cols] = te.fit_transform(df[cat_cols], df['damage_grade'])
    else:
        te = None

    # Drop temporary code columns
    drop_codes = [c for c in df.columns if c.endswith('_code')]
    if len(drop_codes) > 0:
        df.drop(columns=drop_codes, inplace=True)

    return df, te

# ----------------------------- 3. Prepare features ---------------------------

def prepare_features(df):
    X = df.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
    y = df['damage_grade']
    print("\n" + "="*80)
    print("FEATURE PREPARATION")
    print("="*80)
    print(f"Number of features: {X.shape[1]}")
    return X, y

# ----------------------------- 4. Baseline training --------------------------

def train_baseline_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=300, random_state=42, eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=300, random_state=42, verbose=-1, n_jobs=-1)
    }

    results = {}
    print("\n" + "="*80)
    print("BASELINE MODEL PERFORMANCE")
    print("="*80)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'model': model, 'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred}
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")

    return results

# ----------------------------- 5. Fast randomized tuning ---------------------

def fast_tune_xgboost(X_train, y_train, n_iter=20):
    param_dist = {
        'n_estimators': [300, 500, 800],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5],
        'min_child_weight': [1, 3, 5]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1)
    rs = RandomizedSearchCV(xgb, param_dist, n_iter=n_iter, cv=3, scoring='f1_weighted', n_jobs=-1, random_state=42, verbose=2)
    rs.fit(X_train, y_train)
    print(f"Best XGBoost params: {rs.best_params_}")
    return rs.best_estimator_

# ----------------------------- 6. Ensemble & Save ---------------------------

def create_weighted_ensemble(models_with_names, X_train, y_train, weights=None):
    # models_with_names: list of (name, model) tuples
    estimators = [(name, m) for name, m in models_with_names]
    if weights is None:
        # default weights: give more weight to XGBoost/LightGBM if present
        weights = []
        for name, _ in models_with_names:
            if 'XGBoost' in name:
                weights.append(3)
            elif 'LightGBM' in name or 'LGB' in name:
                weights.append(2)
            elif 'Random Forest' in name:
                weights.append(2)
            else:
                weights.append(1)

    ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    ensemble.fit(X_train, y_train)
    return ensemble

# ----------------------------- 7. Evaluation --------------------------------

def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    print("\n" + "="*80)
    print(f"{model_name.upper()} - DETAILED EVALUATION")
    print("="*80)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    return acc, y_pred

# ----------------------------- 8. Main pipeline -----------------------------

def main(filepath, tune_hyperparameters=False, save_model=True):
    df = load_and_explore_data(filepath)
    df_proc, target_encoder = preprocess_data(df)
    X, y = prepare_features(df_proc)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE balancing on training data
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    print(f"\nAfter SMOTE: {np.bincount(y_train_bal)}")

    # Baseline models
    baseline_results = train_baseline_models(X_train_bal, X_test_scaled, y_train_bal, y_test)

    # Choose top 3 models by F1
    top3 = sorted(baseline_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
    print('\nTop 3 baseline models by F1:')
    for name, res in top3:
        print(name, res['f1_score'])

    # Optionally do a fast randomized tune for XGBoost (recommended) if requested
    if tune_hyperparameters:
        print('\nRunning fast randomized tuning for XGBoost (this takes time)')
        best_xgb = fast_tune_xgboost(X_train_bal, y_train_bal, n_iter=25)
    else:
        # use baseline xgboost if present
        best_xgb = None
        if 'XGBoost' in baseline_results:
            best_xgb = baseline_results['XGBoost']['model']

    # Build weighted ensemble using the trained baseline models (or tuned XGB if present)
    models_for_ensemble = []
    for name, res in baseline_results.items():
        models_for_ensemble.append((name, res['model']))

    # If tune produced a better xgb, replace it
    if best_xgb is not None:
        # replace baseline xgboost
        models_for_ensemble = [(n, m) if n != 'XGBoost' else ('XGBoost', best_xgb) for n, m in models_for_ensemble]

    # Select top 3 models by baseline F1 for ensemble ordering
    ordered = sorted(baseline_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    selected_names = [name for name, _ in ordered[:3]]
    selected_models = [(name, dict(baseline_results)[name]['model']) for name in selected_names]

    ensemble = create_weighted_ensemble(selected_models, X_train_bal, y_train_bal)
    acc, y_pred = evaluate_model(ensemble, X_test_scaled, y_test, 'Weighted_Ensemble')

    # Feature importance pruning (optional): keep features above mean importance for tree models
    # Attempt to get importances from one of the tree models
    ref_model = None
    for n, m in selected_models:
        if hasattr(m, 'feature_importances_'):
            ref_model = m
            break
    if ref_model is not None:
        importances = ref_model.feature_importances_
        feat_names = X.columns
        mean_imp = np.mean(importances)
        important_feats = [f for f, imp in zip(feat_names, importances) if imp > mean_imp]
        print(f"\nPruning to {len(important_feats)} important features (mean importance threshold)")
        if len(important_feats) > 5:
            X_sel = X[important_feats]
            X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.2, random_state=42, stratify=y)
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            sm = SMOTE(random_state=42)
            X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_s, y_tr)
            # retrain a fast xgb on selected features and evaluate
            fast_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
            fast_xgb.fit(X_tr_bal, y_tr_bal)
            acc2, _ = evaluate_model(fast_xgb, X_te_s, y_te, 'Pruned_XGBoost')
            print(f"Pruned model accuracy: {acc2:.4f}")

    # Save ensemble + encoders + scaler
    if save_model:
        save_dict = {
            'ensemble': ensemble,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'feature_columns': list(X.columns)
        }
        joblib.dump(save_dict, 'earthquake_ensemble.pkl')
        print('\n✅ Model, scaler and encoders saved to earthquake_ensemble.pkl')

    return ensemble, acc

# ----------------------------- 9. Run ----------------------------------------

if __name__ == '__main__':
    dataset_path = '02_final_ml_ready_dataset.csv'  # update if needed
    main(dataset_path, tune_hyperparameters=False, save_model=True)
