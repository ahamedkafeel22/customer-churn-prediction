import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────
df = pd.read_csv(r"C:\Users\syedk\Documents\Self Projects\Project 3\telco_churn.csv")

# ── 2. EXPLORE ────────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())
print("\nChurn Distribution:\n", df['Churn'].value_counts())
print("\nChurn %:\n", df['Churn'].value_counts(normalize=True) * 100)
print("\nSample:\n", df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE

# ── 1. LOAD ───────────────────────────────────────────
df = pd.read_csv(r"C:\Users\syedk\Documents\Self Projects\Project 3\telco_churn.csv")

# ── 2. CLEAN ──────────────────────────────────────────
# Fix TotalCharges — convert to numeric (spaces become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many NaN appeared
print("TotalCharges nulls after conversion:", df['TotalCharges'].isnull().sum())

# Fill NaN with median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID — not needed for ML
df.drop(columns=['customerID'], inplace=True)

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ── 3. FEATURE ENGINEERING ────────────────────────────
# Tenure groups
df['tenure_group'] = pd.cut(df['tenure'],
                             bins=[0, 12, 24, 48, 60, 72],
                             labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr'])

# Monthly charge category
df['charge_category'] = pd.cut(df['MonthlyCharges'],
                                bins=[0, 35, 65, 95, 120],
                                labels=['Low', 'Medium', 'High', 'Very High'])

# ── 4. ENCODE ─────────────────────────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols += ['tenure_group', 'charge_category']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── 5. SPLIT & SMOTE ──────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── 6. TRAIN BEST MODEL ───────────────────────────────
best = RandomForestClassifier(n_estimators=100, random_state=42)
best.fit(X_train, y_train)
y_pred_best = best.predict(X_test)
roc = roc_auc_score(y_test, best.predict_proba(X_test)[:,1])
print(f"Random Forest ROC-AUC: {roc:.4f}")
print(classification_report(y_test, y_pred_best))

# ── 7. PLOT 1: Confusion Matrix ───────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, ax=ax,
                                         display_labels=['No Churn', 'Churn'],
                                         colorbar=False, cmap='Blues')
ax.set_title('Random Forest — Confusion Matrix')
plt.tight_layout()
plt.savefig(r"C:\Users\syedk\Documents\Self Projects\Project 3\confusion_matrix.png", dpi=150)
plt.close()
print("✅ Confusion matrix saved!")

# ── 8. PLOT 2: ROC Curve ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(best, X_test, y_test, ax=ax, name='Random Forest')
ax.set_title('ROC Curve — Random Forest')
plt.tight_layout()
plt.savefig(r"C:\Users\syedk\Documents\Self Projects\Project 3\roc_curve.png", dpi=150)
plt.close()
print("✅ ROC curve saved!")

# ── 9. PLOT 3: Feature Importance ─────────────────────
feat_imp = pd.Series(best.feature_importances_,
                     index=df.drop(columns=['Churn']).columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 10 Feature Importances — Random Forest')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(r"C:\Users\syedk\Documents\Self Projects\Project 3\feature_importance.png", dpi=150)
plt.close()
print("✅ Feature importance saved!")

# ── 10. SAVE MODEL ────────────────────────────────────
joblib.dump(best, r"C:\Users\syedk\Documents\Self Projects\Project 3\churn_model.pkl")
joblib.dump(scaler, r"C:\Users\syedk\Documents\Self Projects\Project 3\scaler.pkl")
print("✅ Model saved!")
print("\n🎉 Project 3 Complete!")