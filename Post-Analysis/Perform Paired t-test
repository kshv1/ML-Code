import pandas as pd
from scipy.stats import ttest_rel
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = 'final_selected_features640.csv'  # replace with your actual file path
selected_features = pd.read_csv(file_path)['Selected Features'].tolist()

# Load the main data CSV (assuming it contains 'TargetID' and 'class' columns)
main_data_path = 'imputed_data.csv'  # replace with your actual main data file path
data = pd.read_csv(main_data_path)

# Drop the 'class' column as it's not needed for the analysis
data.drop(columns=['class'], inplace=True)

# Extract identifiers from the TargetID to identify sibling pairs
data['Identifier'] = data['TargetID'].apply(lambda x: x.split('.')[0])
data['Type'] = data['TargetID'].apply(lambda x: x.split('.')[1])

# Separate the affected and unaffected individuals
affected = data[data['Type'] == 'p1']
unaffected = data[data['Type'] == 's1']

# Initialize lists to store paired data
paired_affected = []
paired_unaffected = []

# Pair the data based on the identifier and selected features
for identifier in affected['Identifier']:
    affected_row = affected[affected['Identifier'] == identifier][selected_features]
    unaffected_row = unaffected[unaffected['Identifier'] == identifier][selected_features]
    if not unaffected_row.empty:
        paired_affected.append(affected_row.values[0])
        paired_unaffected.append(unaffected_row.values[0])

# Convert lists to DataFrame for paired analysis
paired_affected_df = pd.DataFrame(paired_affected, columns=selected_features)
paired_unaffected_df = pd.DataFrame(paired_unaffected, columns=selected_features)

# Perform paired t-test
t_stats, p_values = ttest_rel(paired_unaffected_df, paired_affected_df)

# Adjust p-values using the Benjamini-Hochberg method
reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method='fdr_bh')

# Create a DataFrame for the results
results = pd.DataFrame({
    'Methylation_Site': selected_features,
    't_statistic': t_stats,
    'p_value': p_values,
    'p_value_corrected': pvals_corrected,
    'reject_null': reject
})

# Sort results by corrected p-value
results.sort_values(by='p_value_corrected', inplace=True)

# Save results to a CSV file
results.to_csv('methylation_sites_t_test_results.csv', index=False)

print("Paired t-test, p-value correction, and graph creation completed. Results saved to 'methylation_sites_t_test_results.csv' and 'top_15_methylation_sites.png'.")
