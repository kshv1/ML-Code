import pandas as pd
from google.colab import files

csv_file = 'transposed_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Extract class information from 'TargetID' column
df['class'] = df['TargetID'].apply(lambda x: 0 if '.s1.' in x else (1 if '.p1.' in x else None))

# Drop rows where 'class' is None (i.e., neither 's1' nor 'p1' in 'TargetID')
df = df.dropna(subset=['class'])

# Convert 'class' column to integer type
df['class'] = df['class'].astype(int)

# Display the first few rows with the 'class' column added
print("First few rows with binary class column:")
print(df.head())

# Save the updated DataFrame to a new CSV file
updated_csv_file = 'data_with_binary_class.csv'
df.to_csv(updated_csv_file, index=False)
