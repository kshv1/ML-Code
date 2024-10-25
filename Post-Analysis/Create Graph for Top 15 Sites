import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv('imputed_data.csv')

# Separate the data into two groups based on the 'class' column
asd_data = data[data['class'] == 1]
non_asd_data = data[data['class'] == 0]

# Drop the 'class' column to focus only on methylation sites
asd_data = asd_data.drop(columns=['class'])
non_asd_data = non_asd_data.drop(columns=['class'])

# Define the list of features to be used
selected_sites = [
   'cg00744433',
    'cg04223956',
    'cg10094277',
    'cg12010995',
    'cg13703941',
    'cg23477967',
    'cg11161873',
    'cg13459560',
    'cg21660392',
    'cg27032352',
    'cg13059335',
    'cg21942438',
    'cg16650125',
    'cg04689061',
    'cg19612574'
]

# Initialize lists to store means and standard errors
means_asd = []
means_non_asd = []
std_errors_asd = []
std_errors_non_asd = []

# Calculate mean and 2 times the standard error for plotting for each of the selected sites
for site in selected_sites:
    asd_mean = asd_data[site].mean()
    non_asd_mean = non_asd_data[site].mean()
    asd_std_error = asd_data[site].sem() * 2
    non_asd_std_error = non_asd_data[site].sem() * 2

    means_asd.append(asd_mean)
    means_non_asd.append(non_asd_mean)
    std_errors_asd.append(asd_std_error)
    std_errors_non_asd.append(non_asd_std_error)

# Plotting the bar graph with error bars representing 2SEM
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(selected_sites))

# Plot bars with adjusted error bar linewidth
bar1 = ax.bar(index - bar_width/2, means_asd, bar_width, yerr=std_errors_asd, label='ASD', capsize=3, color='tab:blue', error_kw=dict(lw=1))
bar2 = ax.bar(index + bar_width/2, means_non_asd, bar_width, yerr=std_errors_non_asd, label='Unaffected Sibling', capsize=3, color='tab:orange', error_kw=dict(lw=1))

# Customize the font sizes
ax.set_xlabel('Methylation Sites', fontsize=14)
ax.set_ylabel('Average Normalized Methylation Level', fontsize=14)
ax.set_title('Methylation Levels for Selected Sites', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(selected_sites, fontsize=12, rotation=45, ha='right')
ax.legend(fontsize=10, bbox_to_anchor=(0.005, 1), loc='upper left', borderaxespad=0.)

# Set y-axis ticks to 0.0, 0.4, and 0.8
ax.set_yticks([0.0, 0.4, 0.8])
ax.set_yticklabels(['0.0', '0.4', '0.8'], fontsize=12)

plt.tight_layout()
plt.show()
