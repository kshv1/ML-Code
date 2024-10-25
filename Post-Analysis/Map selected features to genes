import pandas as pd

# Load the file with methylation sites and symbols
file_path = 'GPL8490-65.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Load the file with the list of methylation sites of interest
sites_file_path = 'final_selected_features640.csv'  # Replace with your file path
sites_df = pd.read_csv(sites_file_path)

# Extract the list of methylation sites
methylation_sites = sites_df['Selected Features'].tolist()

# Filter the dataframe to get the rows where the methylation site is in the list
filtered_df = df[df['ID'].isin(methylation_sites)]

# Create the final table with 'Methylation Site' and 'Symbol' columns
result_table = filtered_df[['ID', 'Symbol']]

# Print the result
print(result_table)

# Save the result to a new CSV file if needed
result_table.to_csv('gene_list.csv', index=False)
