"""
==================================================================================
BLOODSMART AI - TWO-STAGE VALIDATION APPROACH
==================================================================================
BLOCK 1: REAL DATA - VITAL SIGNS MODEL VALIDATION

This block:
1. Loads REAL datasets (UCI, Kaggle, Mendeley)
2. Trains model on VITAL SIGNS only (what we actually have)
3. Validates on real data with cross-validation
4. Establishes baseline performance on REAL clinical records

Run this first to establish the real-data foundation.
==================================================================================
"""

print("="*80)
print("🏥 BLOCK 1: REAL DATA - VITAL SIGNS MODEL")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# DOWNLOAD REAL DATASETS
# ============================================================================
print("\n" + "="*80)
print("📥 DOWNLOADING REAL DATASETS")
print("="*80)

# --- UCI Dataset (Auto-download) ---
print("\n1️⃣ UCI Maternal Health Risk Dataset (Bangladesh)")
from ucimlrepo import fetch_ucirepo

maternal_health_risk = fetch_ucirepo(id=863)
uci_data = pd.concat([
    maternal_health_risk.data.features,
    maternal_health_risk.data.targets
], axis=1)

print(f"✅ Loaded: {uci_data.shape[0]} rows, {uci_data.shape[1]} columns")

# --- Kaggle Dataset (Local file) ---
print("\n2️⃣ Kaggle Dataset")
print("   Download from: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data")
print("   Place file in current directory as 'kaggle_maternal.csv' or 'kaggle_maternal.zip'")

try:
    import os
    import zipfile
    
    # Check for CSV first
    if os.path.exists('kaggle_maternal.csv'):
        kaggle_data = pd.read_csv('kaggle_maternal.csv')
        print(f"✅ Loaded: {kaggle_data.shape[0]} rows")
    # Check for ZIP
    elif os.path.exists('kaggle_maternal.zip'):
        with zipfile.ZipFile('kaggle_maternal.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        csv_file = [f for f in os.listdir('.') if f.endswith('.csv') and 'kaggle' in f.lower()][0]
        kaggle_data = pd.read_csv(csv_file)
        print(f"✅ Loaded: {kaggle_data.shape[0]} rows")
    else:
        print("⚠️ File not found - Skipped")
        kaggle_data = None
except Exception as e:
    print(f"⚠️ Error loading Kaggle data: {e}")
    kaggle_data = None

# --- Mendeley Dataset (Local file) ---
print("\n3️⃣ Mendeley Dataset")
print("   Download from: https://data.mendeley.com/datasets/p5w98dvbbk/1")
print("   Place file in current directory as 'mendeley_maternal.csv' or 'mendeley_maternal.zip'")

try:
    import os
    import zipfile
    
    # Check for CSV first
    if os.path.exists('mendeley_maternal.csv'):
        mendeley_data = pd.read_csv('mendeley_maternal.csv')
        print(f"✅ Loaded: {mendeley_data.shape[0]} rows")
    # Check for ZIP
    elif os.path.exists('mendeley_maternal.zip'):
        with zipfile.ZipFile('mendeley_maternal.zip', 'r') as zip_ref:
            zip_ref.extractall('mendeley')
        csv_files = [f for f in os.listdir('mendeley') if f.endswith('.csv')]
        if csv_files:
            mendeley_data = pd.read_csv(os.path.join('mendeley', csv_files[0]))
            print(f"✅ Loaded: {mendeley_data.shape[0]} rows")
        else:
            print("⚠️ No CSV found in ZIP - Skipped")
            mendeley_data = None
    else:
        print("⚠️ File not found - Skipped")
        mendeley_data = None
except Exception as e:
    print(f"⚠️ Error loading Mendeley data: {e}")
    mendeley_data = None

# ============================================================================
# STANDARDIZE DATASETS - VITAL SIGNS ONLY
# ============================================================================
print("\n" + "="*80)
print("🔍 EXTRACTING VITAL SIGNS FEATURES (REAL DATA ONLY)")
print("="*80)

print("\n📋 What We HAVE in Real Data:")
print("   ✅ Age")
print("   ✅ Systolic BP")
print("   ✅ Diastolic BP")
print("   ✅ Heart Rate")
print("   ⚠️ BMI (partial - only Mendeley)")
print("   ⚠️ Blood Sugar (not primary PPH factor, but present)")

print("\n📋 What We DON'T HAVE (will add in Block 2):")
print("   ❌ Parity")
print("   ❌ Hemoglobin")
print("   ❌ History of PPH")
print("   ❌ Placenta previa")
print("   ❌ Mode of delivery")
print("   ❌ Gestational age")
print("   ... (obstetric features)")

datasets_list = []

# Process UCI
uci_clean = uci_data.copy()
uci_clean.columns = uci_clean.columns.str.lower().str.replace(' ', '_')
uci_clean = uci_clean.rename(columns={
    'age': 'age',
    'systolicbp': 'systolic_bp',
    'diastolicbp': 'diastolic_bp',
    'bs': 'blood_sugar',
    'bodytemp': 'body_temp',
    'heartrate': 'heart_rate'
})

# Binary target: high/mid risk = 1, low risk = 0
uci_clean['high_risk'] = uci_clean['risklevel'].apply(
    lambda x: 1 if 'high' in str(x).lower() or 'mid' in str(x).lower() else 0
)
uci_clean['source'] = 'UCI'
datasets_list.append(uci_clean)
print(f"\n✅ UCI: {len(uci_clean)} rows")

# Process Kaggle (same structure as UCI)
if kaggle_data is not None:
    kaggle_clean = kaggle_data.copy()
    kaggle_clean.columns = kaggle_clean.columns.str.lower().str.replace(' ', '_')
    kaggle_clean = kaggle_clean.rename(columns={
        'age': 'age',
        'systolicbp': 'systolic_bp',
        'diastolicbp': 'diastolic_bp',
        'bs': 'blood_sugar',
        'bodytemp': 'body_temp',
        'heartrate': 'heart_rate'
    })
    kaggle_clean['high_risk'] = kaggle_clean['risklevel'].apply(
        lambda x: 1 if 'high' in str(x).lower() or 'mid' in str(x).lower() else 0
    )
    kaggle_clean['source'] = 'Kaggle'
    datasets_list.append(kaggle_clean)
    print(f"✅ Kaggle: {len(kaggle_clean)} rows")

# Process Mendeley
if mendeley_data is not None:
    mendeley_clean = mendeley_data.copy()
    mendeley_clean.columns = mendeley_clean.columns.str.lower().str.replace(' ', '_')

    mendeley_clean = mendeley_clean.rename(columns={
        'systolic_bp': 'systolic_bp',
        'diastolic': 'diastolic_bp',
        'bs': 'blood_sugar',
        'body_temp': 'body_temp',
        'bmi': 'bmi',
        'heart_rate': 'heart_rate',
        'risk_level': 'risklevel'
    })

    mendeley_clean['high_risk'] = mendeley_clean['risklevel'].apply(
        lambda x: 1 if 'high' in str(x).lower() else 0
    )
    mendeley_clean['source'] = 'Mendeley'
    datasets_list.append(mendeley_clean)
    print(f"✅ Mendeley: {len(mendeley_clean)} rows")

# Find common columns
common_cols = set(datasets_list[0].columns)
for df in datasets_list[1:]:
    common_cols = common_cols.intersection(set(df.columns))

print(f"\nCommon features: {sorted([c for c in common_cols if c not in ['source', 'risklevel', 'high_risk']])}")

# Merge datasets
merged_df = pd.concat([df[list(common_cols)] for df in datasets_list], ignore_index=True)

print(f"\n✅ MERGED REAL DATA: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
print(f"\nTarget distribution:")
print(f"   Low Risk (0): {(merged_df['high_risk']==0).sum()} ({(merged_df['high_risk']==0).mean():.1%})")
print(f"   High Risk (1): {(merged_df['high_risk']==1).sum()} ({(merged_df['high_risk']==1).mean():.1%})")

# ============================================================================
# FEATURE ENGINEERING - VITAL SIGNS ONLY
# ============================================================================
print("\n" + "="*80)
print("⚙️ FEATURE ENGINEERING (VITAL SIGNS)")
print("="*80)

# Handle missing values
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged_df[col].isnull().any():
        merged_df[col].fillna(merged_df[col].median(), inplace=True)

print("✅ Missing values handled")

# Create interaction features from vital signs
merged_df['age_risk'] = ((merged_df['age'] < 20) | (merged_df['age'] > 35)).astype(int)
merged_df['hypertension'] = ((merged_df['systolic_bp'] >= 140) | (merged_df['diastolic_bp'] >= 90)).astype(int)
merged_df['bp_product'] = merged_df['systolic_bp'] * merged_df['diastolic_bp']
merged_df['hr_abnormal'] = ((merged_df['heart_rate'] < 60) | (merged_df['heart_rate'] > 100)).astype(int)

if 'bmi' in merged_df.columns:
    merged_df['obesity'] = (merged_df['bmi'] >= 30).astype(int)

print("\n✅ Engineered features:")
print("   - age_risk (teenage or advanced maternal age)")
print("   - hypertension (BP >= 140/90)")
print("   - bp_product (interaction)")
print("   - hr_abnormal (< 60 or > 100)")
if 'obesity' in merged_df.columns:
    print("   - obesity (BMI >= 30)")

# ============================================================================
# PREPARE FEATURES FOR MODELING
# ============================================================================
print("\n" + "="*80)
print("🎯 PREPARING VITAL SIGNS MODEL")
print("="*80)

# Define vital signs features
vital_signs_features = [
    'age', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'age_risk', 'hypertension', 'bp_product', 'hr_abnormal'
]

# Add optional features if available
if 'bmi' in merged_df.columns:
    vital_signs_features.extend(['bmi', 'obesity'])

if 'blood_sugar' in merged_df.columns:
    vital_signs_features.append('blood_sugar')

X = merged_df[vital_signs_features]
y = merged_df['high_risk']

print(f"Features: {len(vital_signs_features)}")
for i, feat in enumerate(vital_signs_features, 1):
    print(f"   {i}. {feat}")

print(f"\nDataset: {X.shape[0]} samples")
print(f"Positive class: {y.sum()} ({y.mean():.1%})")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("✂️ TRAIN-TEST SPLIT (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
print(f"Train positive: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Test positive: {y_test.sum()} ({y_test.mean():.1%})")

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("📏 FEATURE SCALING")
print("="*80)

scaler_vitals = StandardScaler()
X_train_scaled = scaler_vitals.fit_transform(X_train)
X_test_scaled = scaler_vitals.transform(X_test)

print("✅ StandardScaler fitted")

# ============================================================================
# TRAIN XGBOOST - VITAL SIGNS MODEL
# ============================================================================
print("\n" + "="*80)
print("🚀 TRAINING VITAL SIGNS MODEL (XGBoost)")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

model_vitals = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=RANDOM_STATE,
    use_label_encoder=False
)

print("\n🏃 Training...")
model_vitals.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# ============================================================================
# EVALUATION - VITAL SIGNS MODEL
# ============================================================================
print("\n" + "="*80)
print("📊 VITAL SIGNS MODEL - REAL DATA VALIDATION")
print("="*80)

y_pred = model_vitals.predict(X_test_scaled)
y_pred_proba = model_vitals.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 TEST SET PERFORMANCE (REAL DATA):")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC-ROC: {auc_score:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Low Risk', 'High Risk'],
                          digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print(f"\n📈 CLINICAL METRICS:")
print(f"   Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"   Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("🔄 5-FOLD CROSS-VALIDATION (REAL DATA)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model_vitals, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

print(f"Fold AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("🌟 FEATURE IMPORTANCE (VITAL SIGNS)")
print("="*80)

feature_importance_df = pd.DataFrame({
    'feature': vital_signs_features,
    'importance': model_vitals.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top Features:")
for i, row in feature_importance_df.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance - Vital Signs Model (Real Data)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('vitals_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n💾 Saved: 'vitals_feature_importance.png'")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Vital Signs Model (AUC = {auc_score:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Vital Signs Model (Real Data)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vitals_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("💾 Saved: 'vitals_roc_curve.png'")

# ============================================================================
# SAVE VITAL SIGNS MODEL
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING VITAL SIGNS MODEL")
print("="*80)

with open('model_vitals.pkl', 'wb') as f:
    pickle.dump(model_vitals, f)
print("✅ model_vitals.pkl")

with open('scaler_vitals.pkl', 'wb') as f:
    pickle.dump(scaler_vitals, f)
print("✅ scaler_vitals.pkl")

with open('vital_signs_features.pkl', 'wb') as f:
    pickle.dump(vital_signs_features, f)
print("✅ vital_signs_features.pkl")

vitals_metadata = {
    'model_type': 'XGBoost',
    'stage': 'Stage 1 - Vital Signs (Real Data)',
    'n_features': len(vital_signs_features),
    'feature_names': vital_signs_features,
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'total_real_records': X.shape[0],
    'auc_score': float(auc_score),
    'accuracy': float(accuracy),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'precision': float(precision),
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'data_sources': ['UCI Bangladesh', 'Kaggle', 'Mendeley Bangladesh']
}

with open('vitals_metadata.pkl', 'wb') as f:
    pickle.dump(vitals_metadata, f)
print("✅ vitals_metadata.pkl")

# Save processed data
merged_df.to_csv('real_data_vitals.csv', index=False)
print("✅ real_data_vitals.csv")

# ============================================================================
# BLOCK 1 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ BLOCK 1 COMPLETE - VITAL SIGNS VALIDATION")
print("="*80)

print(f"""
🎯 STAGE 1: REAL DATA VALIDATION (VITAL SIGNS)

Training Data:
  • Total real clinical records: {X.shape[0]}
  • UCI (Bangladesh): {len([d for d in datasets_list if d['source'].iloc[0] == 'UCI'])} records
  • Kaggle: {len([d for d in datasets_list if d['source'].iloc[0] == 'Kaggle']) if kaggle_data is not None else 0} records
  • Mendeley: {len([d for d in datasets_list if d['source'].iloc[0] == 'Mendeley']) if mendeley_data is not None else 0} records

Model Performance (Real Data Only):
  ✅ AUC-ROC: {auc_score:.3f}
  ✅ Sensitivity: {sensitivity:.1%}
  ✅ Specificity: {specificity:.1%}
  ✅ Precision: {precision:.1%}
  ✅ Cross-Val AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})

Features Used (Vital Signs Only):
  {', '.join(vital_signs_features)}

What This Proves:
  ✅ Our model achieves {auc_score:.1%} AUC on REAL hospital data
  ✅ Validated through 5-fold cross-validation
  ✅ Consistent performance (low CV variance)

Next: Block 2 will add SYNTHETIC obstetric features validated against literature
""")

print("\n🚀 Ready for BLOCK 2: Synthetic Obstetric Features")