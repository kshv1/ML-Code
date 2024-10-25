import pandas as pd

# Read the TXT file into a DataFrame
df = pd.read_csv('GSE27044_Matrix_Normalized_AllSampleBetaPrime.txt', delimiter='\t')

# Transpose the DataFrame
df_transposed = df.T

# Reset the index to keep the row numbers
df_transposed = df_transposed.reset_index()

# Make the first row the header
new_header = df_transposed.iloc[0]  # the first row for the header
df_transposed = df_transposed[1:]  # take the data less the header row
df_transposed.columns = new_header  # set the header row as the df header

# Save the transposed DataFrame to a new CSV file
df_transposed.to_csv('transposed_data.csv', index=False)

# Display the first few rows of the transposed DataFrame
print("First few rows of the transposed dataset:")
print(df_transposed.head())
