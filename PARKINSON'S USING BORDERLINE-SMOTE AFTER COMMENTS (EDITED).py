#!/usr/bin/env python
# coding: utf-8

# In[23]:


import warnings
warnings.filterwarnings("ignore")
from boruta import BorutaPy
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import umap.umap_ as umap
from collections import Counter


# In[24]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[25]:


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


# In[26]:


# Define paths
dataset_path = r'C:\\Users\\Arkaprabha Majumdar\\Desktop\\VIGNAN\\Parkinsons\\Parkinsons Disease.csv' # Adjust to your dataset path
df = pd.read_csv(dataset_path)

df.head()


# # Use Borderline SMOTE here and make new dataset

# In[7]:


# Separate features and target
X = df.drop(columns=["name", "status"], errors='ignore')  # "status" is the target column
y = df["status"]


# In[8]:


# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Define Stratified 10-Fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Prepare results list
results = []

# Evaluate each model within pipeline (SMOTE + model)
for name, model in models.items():
    pipeline = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42)),
        ('clf', model)
    ])

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": np.mean(scores['test_accuracy']),
        "Precision": np.mean(scores['test_precision']),
        "Recall": np.mean(scores['test_recall']),
        "F1": np.mean(scores['test_f1']),
        "ROC AUC": np.mean(scores['test_roc_auc'])
    })

# Create results DataFrame
results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
print("\n===== Model Performance (10-Fold Stratified CV) =====\n")
print(results_df.to_string(index=False))


# In[9]:


# Save the new balanced dataset
results_df.to_csv("Borderline SMOTE_balanced_parkinsons_dataset.csv", index=False)


# ## Hold-out Validation (True Unseen Test Set)

# In[10]:


# ==========================================================
# You can add this to verify real generalization performance

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Use best-performing model (say, Random Forest) on hold-out test
best_model = RandomForestClassifier(random_state=42)
holdout_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', BorderlineSMOTE(random_state=42)),
    ('clf', best_model)
])

holdout_pipeline.fit(X_train, y_train)
y_pred = holdout_pipeline.predict(X_test)

print("\n===== Hold-out (Unseen 20%) Evaluation =====\n")
print(classification_report(y_test, y_pred, digits=3))


# # ============================================================

# # FINAL MODEL TRAINING

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

# ======================================================

# Train-test split

# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)

# ======================================================

# Best model

# ======================================================

best_model = RandomForestClassifier(
n_estimators=300,
random_state=42
)

# ======================================================

# CORRECT PIPELINE (imblearn)

# ======================================================

final_pipeline = Pipeline([
('scaler', StandardScaler()),
('smote', BorderlineSMOTE(random_state=42)),
('clf', best_model)
])

# Train

final_pipeline.fit(X_train, y_train)

# Predict

y_pred = final_pipeline.predict(X_test)

print("\n===== FINAL HOLD-OUT TEST PERFORMANCE =====\n")
print(classification_report(y_test, y_pred, digits=3))

print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(y_test, y_pred))


# In[10]:


from sklearn.metrics import (
classification_report,
confusion_matrix,
roc_auc_score,
roc_curve,
)

# ================================
# Predictions on the test set
# ================================

y_pred = final_pipeline.predict(X_test)

# For ROC AUC (if binary classification)

y_prob = final_pipeline.predict_proba(X_test)[:, 1]

# ================================
# Print classification results
# ================================

print("\n===== FINAL HOLD-OUT TEST PERFORMANCE =====\n")
print(classification_report(y_test, y_pred, digits=3))

# ================================
# Confusion Matrix
# ================================

cm = confusion_matrix(y_test, y_pred)
print("\n===== CONFUSION MATRIX =====\n")
print(cm)

# ================================
# Plot Confusion Matrix
# ================================

plt.figure(figsize=(6, 5))
sns.heatmap(
cm,
annot=True,
fmt='d',
cmap='Blues',
xticklabels=['Pred 0', 'Pred 1'],
yticklabels=['True 0', 'True 1']
)
plt.title("Confusion Matrix - Hold-Out Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# ================================
# ROC AUC Score
# ================================

auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC: {auc:.4f}")

# ================================
# Plot ROC Curve
# ================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hold-Out Test Set")
plt.legend()
plt.show()


# # Feature Importance Analysis (Global)

# In[12]:


# ============================================================
# 3. FEATURE IMPORTANCE ANALYSIS (GLOBAL)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Extract trained model from pipeline
# ------------------------------------------------------------
final_model = final_pipeline.named_steps['clf']

# Check if model supports feature importance
if hasattr(final_model, "feature_importances_"):

    # --------------------------------------------------------
    # Get importance scores
    # --------------------------------------------------------
    importance_scores = final_model.feature_importances_

    # Create sorted dataframe
    feature_importance_df = (
        pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importance_scores
        })
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n===== Feature Importance Ranking =====\n")
    print(feature_importance_df)

    # --------------------------------------------------------
    # Plot Feature Importance Bar Chart
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance_df,
        x="Importance",
        y="Feature"
    )
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

else:
    print("This model does not provide `feature_importances_` attribute.")


# # Boruta Feature Selection Analysis

# In[16]:


# Create a Random Forest model
rfb = RandomForestClassifier(
    n_estimators=1000,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

# Create and fit Boruta
boruta_selector = BorutaPy(
    estimator=rfb,
    n_estimators='auto',
    max_iter=100,
    random_state=42
)

boruta_selector.fit(X, y)

# Extract the selected features
selected_features = [
    col for col, is_selected in zip(df.drop('status', axis=1).columns, boruta_selector.support_)
    if is_selected
]

print("Selected features:", selected_features)


# # SHAP Explainability

# In[29]:


import shap

# Your selected feature list
selected_features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA',
    'DFA', 'spread1', 'spread2', 'D2'
]

# Extract target
y = df["status"].astype(int).values

# Extract only selected features
X = df[selected_features]

# Train-test split (same style as earlier)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply Borderline SMOTE on training data only
sm = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Convert to DataFrames for SHAP compatibility
X_train_res_df = pd.DataFrame(X_resampled, columns=selected_features)
X_test_df = pd.DataFrame(X_test, columns=selected_features)

# Train Random Forest (same as earlier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res_df, y_resampled)

# Create SHAP explainer (use predict_proba for classification)
explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_train_res_df)

# Compute SHAP values for the positive class (Parkinson's = 1)
shap_values = explainer.shap_values(X_test_df)


# In[30]:


# Summary plot
shap.summary_plot(shap_values, X_test_df)


# In[27]:


# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (for multiclass), choose the relevant class
# For binary classification using TreeExplainer, shap_values is a numpy array
# Assuming binary classification:
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)

# Get mean absolute SHAP values per feature (i.e., global importance values)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

# Print or inspect
print("Mean Absolute SHAP Values (Feature Importances):")
print(mean_abs_shap)


# In[32]:


from numpy.core import shape_base
y_pred_proba = model.predict_proba(X_test)[..., 1]
print(explainer.expected_value - y_pred_proba.mean())

shap.initjs()
for right in np.where(y_pred_proba == y_test)[0]:
  print(f'test sample {right:02d}.', "True Positive" if y_pred_proba[right] == 1 else "True Negative")
  display(shap.force_plot(
      explainer.expected_value,
      shap_values[right,:],
      X_test.iloc[right]
))


# In[33]:


from sys import base_exec_prefix
shap.initjs()
for error in np.where(y_pred_proba != y_test)[0]:
  print(f'test sample {error:02d}.', "False Positive" if y_pred_proba[error] == 1 else "False Negative")
  display(shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values
))


# # LIME Explainability

# In[30]:


# ============================================================

# OPTIONAL - LIME EXPLANATION

# ============================================================

from lime.lime_tabular import LimeTabularExplainer

# Initialize Lime Explainer

lime_explainer = LimeTabularExplainer(
training_data=np.array(X_train),
feature_names=X_train.columns,
class_names=['Class 0', 'Class 1'],
mode='classification'
)

# Choose sample for explanation

sample_id = 0

lime_exp = lime_explainer.explain_instance(
data_row=X_test.iloc[sample_id],
predict_fn=final_pipeline.predict_proba
)

# Show Explanation

lime_exp.show_in_notebook(show_table=True)


# # Error Analysis

# In[19]:


# ============================================================
# 6. ERROR ANALYSIS ON HOLD-OUT TEST SET
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Predict on test set
# ------------------------------------------------------------
y_pred = final_pipeline.predict(X_test)

# ------------------------------------------------------------
# Identify misclassified samples
# ------------------------------------------------------------
misclassified_mask = (y_pred != y_test)

misclassified_df = (
    pd.DataFrame(X_test, columns=X_test.columns)
    .assign(
        True_Label=y_test.values,
        Predicted_Label=y_pred
    )
    .loc[misclassified_mask]
)

# ------------------------------------------------------------
# Display summary
# ------------------------------------------------------------
print("\n===== Error Analysis: Misclassified Samples =====\n")
print(f"Total test samples     : {len(X_test)}")
print(f"Misclassified samples   : {misclassified_mask.sum()}")

if misclassified_mask.sum() > 0:
    print("\n===== Misclassified Records (Top 10) =====\n")
    print(misclassified_df.head(10))
else:
    print("\nPerfect prediction — No errors found!")

# ------------------------------------------------------------
# (Optional) Save to CSV for deeper manual analysis
# ------------------------------------------------------------
misclassified_df.to_csv("misclassified_records.csv", index=False)
print("\nMisclassified samples saved to 'misclassified_records.csv'.")


# # Feature Engineering

# In[15]:


# ============================================================
# 7. FEATURE ENGINEERING PIPELINE (CORRECTED)
# ============================================================

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Final Pipeline (NO nested pipelines)
# ------------------------------------------------------------
final_pipeline = ImbPipeline([
    ("smote", BorderlineSMOTE(random_state=42)),      # Handles imbalance
    ("scaler", StandardScaler()),                     # Feature scaling
    ("poly", PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )),                                               # Interaction terms
    ("pca", PCA(n_components=0.95)),                  # Dimensionality reduction
    ("clf", RandomForestClassifier(random_state=42))  # Final classifier
])

# ------------------------------------------------------------
# Fit on training data
# ------------------------------------------------------------
final_pipeline.fit(X_train, y_train)

print("\n===== Feature Engineering Complete & Model Trained =====")


# In[16]:


num_final_features = final_pipeline.named_steps["pca"].n_components_
print(f"Final number of features after PCA: {num_final_features}")


# In[34]:


# Get PCA and RF from the pipeline
pca = final_pipeline.named_steps["pca"]
rf = final_pipeline.named_steps["clf"]

# Get polynomial feature names
poly_feature_names = final_pipeline.named_steps["poly"].get_feature_names_out(input_features=X_train.columns)

# Map PCA components back to original features approximately
# Multiply RF importances by absolute PCA loadings
feature_contrib = np.dot(rf.feature_importances_, np.abs(pca.components_))  # shape: n_poly_features

# Get indices of top 7 contributing features
top7_idx = np.argsort(feature_contrib)[-7:][::-1]  # descending

# Names and scores of top 7
top7_names = poly_feature_names[top7_idx]
top7_scores = feature_contrib[top7_idx]

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.bar(top7_names, top7_scores)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Importance (approx.)")
plt.title("Top 7 Features Selected After PCA & RF")
plt.show()


# ### Variance Explained Plot

# In[17]:


# ============================================================
# PCA VARIANCE EXPLAINED PLOT
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

pca = final_pipeline.named_steps["pca"]

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA – Cumulative Variance Explained")
plt.grid()
plt.show()


# ### Correlation Heatmap (Before Feature Engineering)

# In[12]:


# ============================================================
# FEATURE CORRELATION HEATMAP (RAW DATA)
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    annot=True,        # Show numbers
    fmt=".2f",         # Number format (2 decimal places)
    annot_kws={"size": 10}  # Label font size
)
plt.title("Feature Correlation Heatmap – Before Engineering")
plt.show()


# ### Feature Distribution Before vs After Scaling

# In[25]:


# ============================================================
# DISTRIBUTION COMPARISON FOR SCALING
# ============================================================

import pandas as pd

scaled = final_pipeline.named_steps["scaler"].transform(X_train)
scaled_df = pd.DataFrame(scaled, columns=X_train.columns)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.histplot(X_train.iloc[:,0], ax=axes[0])
axes[0].set_title("Raw Feature Distribution")

sns.histplot(scaled_df.iloc[:,0], ax=axes[1])
axes[1].set_title("After Standard Scaling")

plt.show()


# ### Visualize PCA Components as Heatmap

# In[20]:


# ============================================================
# PCA COMPONENTS VS FEATURES HEATMAP (Correct for PolynomialFeatures Pipeline)
# ============================================================

# Step 1: get names after polynomial expansion
poly = final_pipeline.named_steps["poly"]
poly_feature_names = poly.get_feature_names_out(X_train.columns)

# Step 2: get PCA components with correct column labels
components = pd.DataFrame(
    final_pipeline.named_steps["pca"].components_,
    columns=poly_feature_names
)

plt.figure(figsize=(14, 6))
sns.heatmap(components, cmap="viridis")
plt.title("PCA Components – Feature Contribution Strength")
plt.xlabel("Engineered Features")
plt.ylabel("PCA Components")
plt.show()


# ### Visualize Data in 2D (First 2 PCA Components)

# In[27]:


# ============================================================
# 2D PCA PROJECTION SCATTER PLOT
# ============================================================

from sklearn.decomposition import PCA

pca_2 = PCA(n_components=2)
proj = pca_2.fit_transform(
    final_pipeline.named_steps["scaler"].transform(X_train)
)

plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], c=y_train, cmap='coolwarm')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Dataset Projected into 2 PCA Components")
plt.show()


# ### Distribution Comparison Between Classes

# In[28]:


# ============================================================
# FEATURE DISTRIBUTION BY CLASS
# ============================================================

feature = X_train.columns[0]   # pick 1st feature, or loop later

plt.figure(figsize=(7,5))
sns.kdeplot(X_train[feature][y_train==0], label="Class 0", shade=True)
sns.kdeplot(X_train[feature][y_train==1], label="Class 1", shade=True)
plt.title(f"Distribution of {feature} by Class")
plt.legend()
plt.show()


# # Polynomial Interaction Feature Visualizations

# ### Extract Polynomial Features from Pipeline

# In[24]:


# ============================================================
# EXTRACT POLYNOMIAL FEATURES FROM PIPELINE
# ============================================================

poly = final_pipeline.named_steps["poly"]
scaler = final_pipeline.named_steps["scaler"]

# Apply transformations step-by-step
X_scaled = scaler.transform(X_train)
X_poly = poly.transform(X_scaled)

# Get readable feature names
feature_names = poly.get_feature_names_out(X_train.columns)

print("Polynomial Features extracted from pipeline.")


# ### Top N Most Variable Polynomial Features

# In[25]:


# ============================================================
# MOST VARIABLE POLYNOMIAL FEATURES
# ============================================================

X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
variances = X_poly_df.var().sort_values(ascending=False)

topN = 20         # Change N as needed
print(f"\nTop {topN} Most Variable Polynomial Features:\n")
print(variances.head(topN))


# ### Visual Distribution of a Polynomial Feature

# In[27]:


X_poly_df = X_poly_df.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)


# In[29]:


# ============================================================
# KDE PLOT FOR ONE POLYNOMIAL FEATURE (FIXED)
# ============================================================

# Reset indexes so boolean masks align
X_poly_df = X_poly_df.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

# Choose the feature with highest variance
feature = variances.index[0]

plt.figure(figsize=(8,5))
sns.kdeplot(
    X_poly_df.loc[y_train_reset==0, feature],
    shade=True, label="Class 0"
)
sns.kdeplot(
    X_poly_df.loc[y_train_reset==1, feature],
    shade=True, label="Class 1"
)

plt.title(f"Distribution of {feature} by Class")
plt.xlabel(feature)
plt.ylabel("Density")
plt.legend()
plt.show()


# In[ ]:




