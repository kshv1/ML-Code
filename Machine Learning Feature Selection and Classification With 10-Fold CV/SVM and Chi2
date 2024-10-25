import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, RFE
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

# Set the random seed for reproducibility
random_seed = 50
np.random.seed(random_seed)

# Load your dataset
df = pd.read_csv('imputed_data.csv')

# Create the group column based on the first five characters of 'TargetID'
df['GroupID'] = df['TargetID'].str[:5]

# Separate features and target variable
X = df.drop(columns=['TargetID', 'class'])
y = df['class']
groups = df['GroupID']
feature_names = X.columns

# 1. Reduce the number of features to 2500 using LinearSVC
lsvc = LinearSVC(C=2, penalty="l2", dual=False, max_iter=10000, random_state=random_seed).fit(X, y)
num_features = min(2500, X.shape[1])
model = SelectFromModel(lsvc, prefit=True, max_features=num_features)
X_lsvc = model.transform(X)
selected_lsvc_features = feature_names[model.get_support()]
print(f"Features after LinearSVC: {X_lsvc.shape[1]}")

# 2. Further feature selection using chi-squared and SVM-RFE

# Chi-Squared
num_features = min(2000, X_lsvc.shape[1])
chi2_selector = SelectKBest(chi2, k=num_features).fit(X_lsvc, y)
X_chi2 = chi2_selector.transform(X_lsvc)
selected_chi2_features = selected_lsvc_features[chi2_selector.get_support()]
print(f"Features after Chi-Squared: {X_chi2.shape[1]}")

# SVM-RFE
svc = SVC(kernel="linear", C=2, random_state=random_seed)
rfe_selector = RFE(estimator=svc, n_features_to_select=640)
X_rfe = rfe_selector.fit_transform(X_chi2, y)
selected_rfe_features = selected_chi2_features[rfe_selector.get_support()]
print(f"Features after SVM-RFE: {X_rfe.shape[1]}")

# Print the names of the selected features after SVM-RFE
print("Selected features after SVM-RFE:")
print(selected_rfe_features)

# Save the names of the selected features to a CSV file
selected_features_df = pd.DataFrame(selected_rfe_features, columns=["Selected Features"])
selected_features_df.to_csv("final_selected_features640.csv", index=False)

# 3. Classification with SVM using 10-fold group cross-validation
def evaluate_model(X, y, groups):
    svm_classifier = SVC(kernel='linear', random_state=random_seed)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    cv = GroupKFold(n_splits=10)

    accuracy_scores = cross_val_score(svm_classifier, X, y, cv=cv, groups=groups, scoring='accuracy')
    precision_scores = cross_val_score(svm_classifier, X, y, cv=cv, groups=groups, scoring='precision')
    recall_scores = cross_val_score(svm_classifier, X, y, cv=cv, groups=groups, scoring='recall')

    print(f"Accuracy: {accuracy_scores.mean()}")
    print(f"Precision: {precision_scores.mean()}")
    print(f"Recall: {recall_scores.mean()}")

# Evaluate the final feature set
print("\nEvaluation using features selected by SVM-RFE:")
evaluate_model(X_rfe, y, groups)
