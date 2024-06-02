"""
This script performs data analysis and visualization on the HIERARCHY-INDEX of gaze-graphs for a set of participants. 
The script will save plots and summary statistics in the specified directories.

The script consists of the following main sections:
1. Adjustable Variables: Define file paths, participant lists, and figure resolution.
2. Load and Transform Data: Load data from CSV files and transform columns containing arrays into DataFrames.
3. Extracting Summary Statistics for Participants: Calculate and save summary statistics for each participant.
4. Plotting: Generate various plots to visualize the data.
5. Statistics: Perform statistical tests to analyze the relationships and differences between summary statistics.

Adjustable Variables:
    - savepath (str): Path to the directory where figures will be saved.
    - summary_file_path (str): Path to the directory a summary file is stored and saved (it needs to be the same across scripts).
    - part_list (list): List of all participants.
    - examples_idx (list): Example participant indices, for plots using only a few participants.
    - save_dpi (int): Resolution of saved PNGs (best: 300 or 600).
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)


################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/hierarchy_final/'
summary_file_path = 'D:/WestbrueckData/Analysis/parts_summary_stats.csv'  
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]

# part_list indice of three example participants
examples_idx = [5,9,14]

# resolution of any saved figures
save_dpi = 300 #600 



################################### 1. Load and transform data ###################################

# load data
g_measures_df = pd.read_csv('overview_graph_measures_final.csv')


# convert read-in array-strings into arrays
g_measures_df['Diameter'] = g_measures_df['Diameter'].apply(ast.literal_eval).apply(np.array)
g_measures_df['HierarchyIndex'] = g_measures_df['HierarchyIndex'].apply(ast.literal_eval).apply(np.array)

# Extract Data data from a column with array-elements into an own dfs
# each column on TS, each row one Part
diameter_array = np.array(g_measures_df['Diameter'].tolist())
hierarchy_array = np.array(g_measures_df['HierarchyIndex'].tolist())

diameter_df = pd.DataFrame(diameter_array, columns=[i+1 for i in range(diameter_array.shape[1])])
hierarchy_df = pd.DataFrame(hierarchy_array, columns=[i+1 for i in range(hierarchy_array.shape[1])])



################################### 2. Extracting summary statistics for particpants ###################################

parts_sum_measures = pd.DataFrame(columns=['ParticipantID',
                                            'MeanHierarchy',
                                            'EndHierarchy',
                                            'MaxHierarchy'])



parts_sum_measures['ParticipantID'] = g_measures_df['Participant']

# Calculate summary statistics for each participant

parts_sum_measures['EndHierarchy'] = hierarchy_df.iloc[:, -1].astype(float) # last column



################################### 3. Saving summary statistics ###################################

# Check if the file exists
if os.path.exists(summary_file_path):
    # Load existing file
    saved_measures = pd.read_csv(summary_file_path)

    # Identify columns that don't already exist in the merged DataFrame
    new_columns = [col for col in parts_sum_measures.columns if col not in saved_measures.columns]

    # Concatenate the new columns by 'ParticipantID'
    new_saved_measures = pd.merge(saved_measures, parts_sum_measures[['ParticipantID'] + new_columns], on='ParticipantID', how='inner')

else:
    new_saved_measures = parts_sum_measures

# Save the updated DataFrame to the file
new_saved_measures.to_csv(summary_file_path, index=False)





################################### 4. Plotting ###################################

# gloabal settings for all figures
mpl.rcParams.update({'font.size': 16,  # for normal text
                     'axes.labelsize': 16,  # for axis labels
                     'axes.titlesize': 16,  # for title
                     'xtick.labelsize': 14,  # for x-axis tick labels
                     'ytick.labelsize': 14})  # for y-axis tick labels



################################### 4.1. Plotting Hierarchy Index over time ###################################

plt.figure(figsize=(9, 5))

for _, row in hierarchy_df.iterrows():
    plt.plot(row.index, row.values, alpha=0.6)

# means per ts
mean_values_per_ts = hierarchy_df.mean(axis=0)
# Plot the mean line
plt.plot(mean_values_per_ts.index, mean_values_per_ts.values, color='#FF0000', linestyle='-', linewidth=2, label='Mean')

#plotting horizontal threhold line
plt.axhline(y=2, color='#103F71', linestyle='--', label='Threshold Index = 2', linewidth=2)

# Customize labels and title
plt.xlabel('Time Step')
plt.ylabel('Hierarchy Index')
#plt.title('Temporal Evolution of the Gaze-Graphs Hierarchy Index with Threshold')
plt.legend()

xticks = np.arange(10,len(hierarchy_df.columns)+1, 10)  # Show every 10th time step
xticks = np.array([1] + xticks.tolist()) # include the first one
plt.xticks(xticks)
plt.xlim(0, 151)

# Show the plot
plt.tight_layout()
plt.savefig(savepath + 'Hierarchy_ot', dpi=save_dpi, bbox_inches='tight')

plt.show()



################################### 4.2. Binary Heatplot of HierarchyIndex over time ###################################

# Create a binary mask based on your criteria

plt.figure(figsize=(7,6))
mask = hierarchy_df >= 2

# Plot the heatmap
sns.heatmap(mask, cmap=['lightgrey','#103F71'], cbar=False, alpha=0.8)


# Create custom legend
legend_elements = [
    Patch(facecolor='lightgrey', edgecolor='black', label='Below Threshold'),
    Patch(facecolor='#103F71', edgecolor='black', label='Above Threshold')
]
plt.legend(handles=legend_elements, fontsize=14)

plt.xlabel('Time Step')
plt.ylabel('Participant ID')

# Set the y-axis labels to be Participant IDs
plt.yticks(ticks=np.arange(len(g_measures_df['Participant'])), labels=g_measures_df['Participant'], rotation=0, va='center', fontsize=14)

# Set the x-axis labels to represent the correct time steps

xticks = np.arange(20, len(hierarchy_df.columns) + 1, 20)
xticks = np.array([1] + xticks.tolist())

plt.xticks(xticks, labels=xticks, rotation=45, fontsize=14)

#plt.title('Hierarchy Index above or below the Threshold of 2', fontsize=13)
# Show the plot
plt.tight_layout()
plt.savefig(savepath + 'hierarchy_binary', dpi=save_dpi, bbox_inches='tight')
plt.show()


##########add legend

################################### 4.3. Plotting three Participants Hierarchy Index over time ###################################

colors_plot = ['#103F71','#179999','#E1A315']
plt.figure(figsize=(11,5.5)) 

for i, e_index in enumerate(examples_idx):
    plt.plot(hierarchy_df.iloc[e_index], alpha=0.9, label= g_measures_df['Participant'].iloc[i], color= colors_plot[i])

# Customize subplot
plt.xlabel('Time Step')
plt.ylabel('Hierarchy Index')

xticks = np.arange(10, len(diameter_df.columns) + 1, 10)
xticks = np.array([1] + xticks.tolist())
plt.xticks(xticks, rotation=45)
plt.xlim(0, 151)


#plt.title('Temporal Evolution of the Gaze-Graphs Hierarchy Index of Three Participants')
plt.legend(title = 'Participant ID')


# Save and show the plot
# Adjust layout to prevent clipping of ylabel
plt.tight_layout()
plt.savefig(savepath + f'hierarchy_samples_plot', dpi=save_dpi, bbox_inches='tight')
plt.show()



###################################

print('End')
