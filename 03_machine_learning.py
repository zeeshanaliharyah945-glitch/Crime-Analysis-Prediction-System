# 03_machine_learning.py
# Purpose: Train machine learning models for crime prediction

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 3: MACHINE LEARNING MODEL TRAINING")
print("="*60)

# ==================== LOAD DATA ====================
print("\n📂 Loading processed data...")
df = pd.read_csv('data/processed_crime_data.csv')
print(f"✅ Loaded {len(df)} records")

# ==================== PREPARE FEATURES ====================
print("\n🔧 Preparing features for training...")

feature_columns = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'Hour',
    'Severity', 'Crime_Type_Encoded', 'Police_Station_Encoded',
    'Area_Encoded', 'TimePeriod_Encoded', 'Station_Crime_Count'
]

X = df[feature_columns]
y = df['High_Risk_Area']

print(f"   Features: {len(feature_columns)}")
print(f"   Target: High_Risk_Area (1=High Risk, 0=Low Risk)")
print(f"   Class distribution: {dict(y.value_counts())}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n✂️ Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ==================== FEATURE SCALING ====================
print("\n📊 Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('outputs/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Scaler saved")

# ==================== TRAIN RANDOM FOREST ====================
print("\n🌲 Training Random Forest Classifier...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

print(f"   Training Accuracy: {rf_train_acc:.4f}")
print(f"   Test Accuracy: {rf_test_acc:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"   Cross-validation scores: {rf_cv_scores}")
print(f"   Mean CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# ==================== TRAIN LOGISTIC REGRESSION ====================
print("\n📈 Training Logistic Regression...")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

print(f"   Training Accuracy: {lr_train_acc:.4f}")
print(f"   Test Accuracy: {lr_test_acc:.4f}")

lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
print(f"   Mean CV Accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")

# ==================== FEATURE IMPORTANCE ====================
print("\n🔍 Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 5 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

# ==================== MODEL EVALUATION ====================
print("\n📊 Evaluating models...")

# ROC AUC scores
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
print(f"   Random Forest AUC: {rf_auc:.4f}")
print(f"   Logistic Regression AUC: {lr_auc:.4f}")

# ==================== SAVE MODELS ====================
print("\n💾 Saving models...")

with open('outputs/models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✅ Random Forest model saved")

with open('outputs/models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("   ✅ Logistic Regression model saved")

# ==================== GENERATE PLOTS ====================
print("\n📈 Generating evaluation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Random Forest - Confusion Matrix', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
axes[0, 1].set_title('Logistic Regression - Confusion Matrix', fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Feature Importance
axes[1, 0].barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10), 
                color='#4ECDC4')
axes[1, 0].set_title('Top 10 Feature Importances', fontweight='bold')
axes[1, 0].set_xlabel('Importance')

# Model Comparison
models = ['Random Forest', 'Logistic Regression']
accuracies = [rf_test_acc, lr_test_acc]
bars = axes[1, 1].bar(models, accuracies, color=['#FF6B6B', '#45B7D1'])
axes[1, 1].set_title('Model Accuracy Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/06_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/charts/06_model_evaluation.png")

# ==================== GENERATE MODEL REPORT ====================
print("\n📄 Generating Model Report...")

model_report = f"""
================================================================================
                    MODEL PERFORMANCE REPORT
================================================================================

DATASET INFORMATION
--------------------------------------------------------------------------------
Total Samples: {len(df)}
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}
Features Used: {len(feature_columns)}

RANDOM FOREST MODEL
--------------------------------------------------------------------------------
Test Accuracy: {rf_test_acc:.4f}
Cross-validation Mean Accuracy: {rf_cv_scores.mean():.4f}
Cross-validation Std: {rf_cv_scores.std():.4f}
AUC Score: {rf_auc:.4f}

LOGISTIC REGRESSION MODEL
--------------------------------------------------------------------------------
Test Accuracy: {lr_test_acc:.4f}
Cross-validation Mean Accuracy: {lr_cv_scores.mean():.4f}
AUC Score: {lr_auc:.4f}

CLASSIFICATION REPORT - RANDOM FOREST
--------------------------------------------------------------------------------
{classification_report(y_test, rf_test_pred, target_names=['Low Risk', 'High Risk'])}

TOP 5 FEATURE IMPORTANCES
--------------------------------------------------------------------------------
"""
for _, row in feature_importance.head(5).iterrows():
    model_report += f"{row['feature']}: {row['importance']:.4f}\n"

model_report += """
================================================================================
"""

print(model_report)

with open('outputs/reports/model_performance_report.txt', 'w') as f:
    f.write(model_report)
print("   ✅ Saved: outputs/reports/model_performance_report.txt")

print("\n" + "="*60)
print("✅ MACHINE LEARNING TRAINING COMPLETE!")
print("="*60)
print("\nModels saved in: outputs/models/")
print("Charts saved in: outputs/charts/")