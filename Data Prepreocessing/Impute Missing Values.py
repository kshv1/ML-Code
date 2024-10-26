import pandas as pd
from sklearn.impute import KNNImputer

csv_file = 'data_with_binary_class.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Separate features and target variable
X = df.iloc[:, 1:-1]  # Exclude 'TargetID' and the last column
y = df.iloc[:, -1]    # Last column is the target variable

# Perform KNN imputation on features
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)

# Concatenate imputed features with 'TargetID' and target variable
df_imputed = pd.DataFrame(X_imputed, columns=X.columns)
df_imputed['TargetID'] = df['TargetID']
df_imputed['class'] = y

# Print the first few rows of the updated DataFrame
print("First few rows of the updated DataFrame:")
print(df_imputed.head())

# Save the updated DataFrame to a new CSV file
updated_csv_file = 'imputed_data.csv'
df_imputed.to_csv(updated_csv_file, index=False)
