"""
This script performs data analysis and visualization on the DIAMETER of gaze-graphs for a set of participants. 
Custom functions from analysis_plotting_functions are used for some plotting tasks.
The script will save plots and summary statistics in the specified directories.

The script consists of the following main sections:
1. Adjustable Variables: Define file paths, participant lists, and figure resolution.
2. Load and Transform Data: Load data from CSV files and transform columns containing arrays into DataFrames.
3. Extracting Summary Statistics for Participants: Calculate and save summary statistics for each participant.
4. Plotting: Generate various plots to visualize the data.
5. Statistics: Perform statistical tests to analyze the relationships and differences between summary statistics.

Adjustable Variables:
    - savepath (str): Path to the directory where figures will be saved.
    - summary_file_path (str): Path to the directory a summary file is stored and saved (it needs to be the same across scripts)
    - part_list (list): list of all participants
    - examples_idx (list): Exymple participant indices, for plots using only a few participants
    - save_dpi (int): resoslution of saved pngs (best :300 or 600)

"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import pearsonr

from analysis_plotting_functions import distribution_hist, data_and_avg_line_plot

warnings.simplefilter(action='ignore', category=FutureWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/diameter_final/'
summary_file_path = 'D:/WestbrueckData/Analysis/parts_summary_stats.csv'  

os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]

# part_list indice of three example participants
examples_idx = [5,9,14]

# resolution of the saved figure 
save_dpi = 300 #600 



################################### 1. Load and transform data ###################################

# load data
g_measures_df = pd.read_csv('overview_graph_measures_final.csv')

# convert read-in array-strings into arrays
g_measures_df['Diameter'] = g_measures_df['Diameter'].apply(ast.literal_eval).apply(np.array)

# Extract data from a column with array-elements into an own dfs
# each column on time step, each row one partisipant
diameter_array = np.array(g_measures_df['Diameter'].tolist())

diameter_df = pd.DataFrame(diameter_array, columns=[i+1 for i in range(diameter_array.shape[1])])



################################### 2. Extracting summary statistics for particpants ###################################

parts_sum_measures = pd.DataFrame(columns=['ParticipantID',
                                            'MeanDiameter',
                                            'EndDiameter',
                                            'MaxDiameter',
                                            'MaxDiameterIndex'])



parts_sum_measures['ParticipantID'] = g_measures_df['Participant']

# Calculate summary statistics for each participant

parts_sum_measures['MeanDiameter'] = diameter_df.mean(axis=1).astype(float)
parts_sum_measures['EndDiameter'] = diameter_df.iloc[:, -1].astype(int) # last column
parts_sum_measures['MaxDiameter'] = diameter_df.max(axis=1).astype(int)
parts_sum_measures['MaxDiameterIndex'] = diameter_df.idxmax(axis=1).astype(int)



################################### 3. Saving summary statistics ###################################

# Check if the summary file already exists
if os.path.exists(summary_file_path):
    
    # Load existing file
    saved_measures = pd.read_csv(summary_file_path)

    # Identify columns that do not exist in the loaded dataframe
    new_columns = [col for col in parts_sum_measures.columns if col not in saved_measures.columns]

    # Concatenate the new columns/ data by 'ParticipantID'
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



################################### 4.1. Plotting Diameter over time ###################################

data_and_avg_line_plot(diameter_df,
                       x_label = 'Time Step',
                       y_label = 'Diameter',
                       fig_title = None, #"Temporal Evolution of Gaze-Graph Diameters",
                       savepath = savepath + f'Diameter_ot',
                       color ='#FF0000',
                       show = True,
                       save_dpi = save_dpi)



################################### 4.2. Plotting three Participants Diameter over time with Max. Diameter ###################################

colors_plot = ['#103F71','#179999','#E1A315']

plt.figure(figsize=(11,5.5)) 

for i, e_index in enumerate(examples_idx):
    plt.plot(diameter_df.iloc[e_index], alpha=0.9, label = g_measures_df['Participant'].iloc[i], color= colors_plot[i], linewidth=1.5,zorder=1)

# scatter max diameter values on top of it
max_diameter = diameter_df.iloc[examples_idx].max(axis=1)
max_diameter_ts = diameter_df.iloc[examples_idx].idxmax(axis=1)

plt.scatter(x= max_diameter_ts, y=max_diameter,color='#FF0000', marker='*', s=100, label = 'Max. Value',zorder=2)

plt.xlabel('Time Step')
plt.ylabel('Diameter')
#plt.title('Temporal Evolution of Gaze-Graph Diameters & Max. Diameters of Three Participants with Diverse Trends', fontsize=14)

xticks = np.arange(10,len(diameter_df.columns)+1, 10)  # Show every 10th time step
xticks = np.array([1] + xticks.tolist()) # include the first one
plt.xticks(xticks)
plt.xlim(0, 151)

plt.legend(title = 'Participant ID')

plt.tight_layout()
plt.savefig(savepath +'diameter_subset_ot', dpi=save_dpi, bbox_inches='tight')
plt.show()



################################### 4.3. Plot of distributions of End Diameter ###################################

distribution_hist(parts_sum_measures['EndDiameter'],
                             x_label= 'Diameter',
                             y_label= 'Count',
                             fig_title = None, #'Distribution of Gaze-Graph Diameters at t=150',
                             savepath = savepath + f'EndDiameter_dist',
                             kde = False,
                             discrete=True,
                             bins = 3,
                             color='#103F71', 
                             show = True,
                             save_dpi = save_dpi)



################################### 4.4. Scatterplot of Max Diameter & MaxIndex with Distributions/Marginals and coloured by End Diameter ###################################

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

# Scatter plot
ax_scatter = fig.add_subplot(gs[1, 0])
colors_plot = ['#103F71','#179999','#E1A315']
markers = ['o','^','s','D']

for i, end_value in enumerate(sorted(parts_sum_measures['EndDiameter'].unique())):
    mask = parts_sum_measures['EndDiameter'] == end_value
    ax_scatter.scatter(parts_sum_measures['MaxDiameterIndex'][mask], parts_sum_measures['MaxDiameter'][mask], label=end_value, alpha = 1, color = colors_plot[i], marker = markers[i])

# Calculate mean values
mean_max_diameter = parts_sum_measures['MaxDiameter'].mean()
mean_max_diameter_idx = parts_sum_measures['MaxDiameterIndex'].mean()

# Add vertical and horizontal lines to the scatter plot
ax_scatter.axvline(mean_max_diameter_idx, color='darkgrey', linestyle='-', label='Mean Max. Diameter TS')
ax_scatter.axhline(mean_max_diameter, color='darkgrey', linestyle='--', label='Mean Max. Diameter')


# top histogram
ax_hist_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
hist_data_top = [parts_sum_measures['MaxDiameterIndex'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_top.hist(hist_data_top, bins=50, edgecolor='black', alpha=0.9, label=[7,8,9], color=colors_plot, stacked=True)

ax_hist_top.set(yticklabels=[], xlabel='')
ax_hist_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


# Right histogram
ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
hist_data_right = [parts_sum_measures['MaxDiameter'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_right.hist(hist_data_right, bins=50, edgecolor='black', alpha=0.9,orientation='horizontal', label=[7,8,9], color=colors_plot, stacked=True)

ax_hist_right.set(xticklabels=[], ylabel='')
ax_hist_right.legend()
ax_hist_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


# Add labels and legend
ax_scatter.set_xlabel('Time Step with Max. Diameter')
ax_scatter.set_ylabel('Max. Diameter')

ax_scatter.legend(title='EndDiameter Value')

# Title
#fig.suptitle('Relationship of Max, MaxTS and End Diameter Values', fontsize=15, y=0.93)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(savepath + 'diameter_sum_measures_relationship', dpi=save_dpi, bbox_inches='tight')
plt.show()



################################### 4.5. Scatterplot of Max & Mean Diameter with Distributions/Marginals and coloured by End Diameter ###################################

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

# Scatter plot
ax_scatter = fig.add_subplot(gs[1, 0])
colors_plot = ['#103F71','#179999','#E1A315'] #['#00008B','#6A5ACD','#C1B9FF'] #

for i, end_value in enumerate(sorted(parts_sum_measures['EndDiameter'].unique())):
    mask = parts_sum_measures['EndDiameter'] == end_value
    ax_scatter.scatter(parts_sum_measures['MeanDiameter'][mask], parts_sum_measures['MaxDiameter'][mask], label=end_value, alpha = 1, color = colors_plot[i],marker = markers[i])

# Calculate mean values
mean_max_diameter = parts_sum_measures['MaxDiameter'].mean()
mean_max_diameter_idx = parts_sum_measures['MeanDiameter'].mean()


# top histogram
ax_hist_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
hist_data_top = [parts_sum_measures['MeanDiameter'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_top.hist(hist_data_top, bins=50, edgecolor='black', alpha=0.9, label=[7,8,9], color=colors_plot, stacked=True)

ax_hist_top.set(yticklabels=[], xlabel='')
ax_hist_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


# Right histogram
ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
hist_data_right = [parts_sum_measures['MaxDiameter'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_right.hist(hist_data_right, bins=50, edgecolor='black', alpha=0.9, orientation='horizontal', label=[7,8,9], color=colors_plot, stacked=True)

ax_hist_right.set(xticklabels=[], ylabel='')
ax_hist_right.legend()
ax_hist_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


# Add labels and legend
ax_scatter.set_xlabel('Mean Diameter')
ax_scatter.set_ylabel('Max. Diameter')
ax_scatter.legend()

ax_scatter.legend(title='EndDiameter Value')

# Title
#fig.suptitle('Relationship of Max, Mean and End Diameter Values', fontsize=15, y= 0.93)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(savepath + 'diameter_sum_measures_relationship_mean', dpi=save_dpi, bbox_inches='tight')
plt.show()





################################### 5. Statistics ###################################


################################### 5.1. Linear Correlation between Max, MaxTS, Mean and End Values of Diameter ###################################

correlation_matrix = parts_sum_measures.corr()

columns = parts_sum_measures.columns

# Create an empty matrix for p-values
p_values_matrix = pd.DataFrame(index=columns, columns=columns)

# Fill the p-values matrix
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        corr_coeff, p_value = pearsonr(parts_sum_measures[columns[i]], parts_sum_measures[columns[j]])
        p_values_matrix.at[columns[i], columns[j]] = p_value
        p_values_matrix.at[columns[j], columns[i]] = p_value

# Print or inspect the correlation matrix and p-values matrix
print("Correlation Matrix:")
print(correlation_matrix)
print("\nP-Values Matrix:")
print(p_values_matrix)



################################### 5.2. MWU/t-test Group Tests between Diameter Values of EndDiameter Groups ###################################

for col1 in ['EndDiameter']:
    for col2 in ['MaxDiameter','MaxDiameterIndex','MeanDiameter']:
        if col1 != col2:
            print(col1,col2)

            
            if col1 == 'EndDiameter':
                # Assuming Groups with 2 different EndDiameters are have significant different distributions
                for i in [(7,8),(8,9),(9,7)]:
                    a,b = i
                    print(a,b)
                    t_stat, p_value = ttest_ind(parts_sum_measures[parts_sum_measures['EndDiameter']==a][col2],
                                                parts_sum_measures[parts_sum_measures['EndDiameter']==b][col2])

                    if p_value < 0.05:
                        print("The distributions are significantly different.",p_value)
                    else:
                        print("The distributions are not significantly different.",p_value)

                    # Assuming dist1 and dist2 are your two distributions
                    stat, p_value = mannwhitneyu(parts_sum_measures[parts_sum_measures['EndDiameter']==a][col2],
                                                parts_sum_measures[parts_sum_measures['EndDiameter']==b][col2])

                    if p_value < 0.05:
                        print("The distributions are significantly different.",p_value)
                    else:
                        print("The distributions are not significantly different.",p_value)



###################################

print('End')
