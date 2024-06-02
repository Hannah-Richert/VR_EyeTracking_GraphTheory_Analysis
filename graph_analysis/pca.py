
"""
This script performs PCA on graph features for the graph measurement data for multiple participants and plots/saves the results.

Steps:
1. Load and Transform Data
1.5 Parameters for Plotting:
2. Perform PCA & Plot first two principal components
3. Extract necessary components until explained variance < 95%
4. Plot the contribution/correlation of the 11 features in the feature components

Adjustable Variables:
- savepath (str): Directory to save generated plots.
- part_list (list): List of participant IDs for analysis.
- save_dpi (int): Pesolution for the saved figures (best: 300 or 600 )

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/pca/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]

# resolution of the saved figure 
save_dpi = 300 #600 



################################### 1. Load and transform data ###################################

dataGraphMeasures =  pd.read_csv('parts_summary_stats.csv')

# selecting the 11 features fr pca
dataGraphMeasures = dataGraphMeasures[['MeanDiameter', 'EndDiameter', 'MaxDiameter', 'MaxDiameterIndex',
                                       'MeanAvgShortPath', 'EndAvgShortPath', 'MaxAvgShortPath',
                                       'MeanExplorationRate', 'MeanDiscoveryRate',
                                       'NumEndNodes', 'NumEndEdges']]



################################### 1.5 Parameters for Plotting ###################################

# gloabal settings for all figures
mpl.rcParams.update({'font.size': 16,  # for normal text
                     'axes.labelsize': 16,  # for axis labels
                     'axes.titlesize': 16,  # for title
                     'xtick.labelsize': 14,  # for x-axis tick labels
                     'ytick.labelsize': 14})  # for y-axis tick labels



################################### 2. Perform PCA & Plot first two principle components ###################################

# Extract the features
X = dataGraphMeasures.values

# Standardize the feature matrix
X_std = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(X_std)

# Plot PCA
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c='#103F71', alpha=1)
#plt.title('PCA with 11 features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(f'{savepath}pca_2comp', dpi=save_dpi, bbox_inches='tight')
plt.show()



################################### 2. Extract necessary components until explained variance < 95% ###################################

# Step 3: Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Find the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Determine the number of components to explain 95% variance
n_components_95 = np.argmax(cumulative_variance_ratio > 0.95) + 1

# Extract the components explaining 95% variance
components_95 = principal_components[:, :n_components_95]

print("Number of components to explain 95% variance:", n_components_95)
print("Explained variance ratios for each component:")
print(explained_variance_ratio[:])



################################### 4. Plot the contribution/correlation of the 11 features in the feature components ###################################

# Create a DataFrame for important components with feature names
components_df = pd.DataFrame(pca.components_[:n_components_95], columns=dataGraphMeasures.columns)
#components_df = pd.DataFrame(pca.components_, columns=dataGraphMeasures.columns)
components_df.index = range(1, len(components_df) + 1)

# Plot the heatmap
custom_cmap = LinearSegmentedColormap.from_list('custom', [(0,'#103f71'),(0.25,'#69a2ae'),(0.5,'#ffffff'),(0.75, '#fbac63'), (1, '#FF0000')])
plt.figure(figsize=(8, 6))

sns.heatmap(components_df.transpose(), cmap=custom_cmap, annot=True, fmt=".2f")
#plt.title('Feature Contributions to Principal Components')
plt.ylabel('Feature')
plt.xlabel('Principal Component')
plt.tight_layout()
plt.savefig(f'{savepath}pca_features_comp', dpi=save_dpi, bbox_inches='tight')
plt.show()



################################### 

print('End')