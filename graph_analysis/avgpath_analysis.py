"""
This script performs data analysis and visualization of the AVERAGESHORTESTPATH and DIAMETER measure. 
Custom functions from analysis_plotting_functions are used for some plotting tasks.
The script will save plots and summary statistics in the specified directories.

The steps include:
1. Adjustable Variables: Defining paths, participant lists, and plot resolution settings.
2. Data Loading and Transformation: Reading and transforming the data from a CSV file, and calculating derived metrics.
3. Extracting Summary Statistics: Calculating and compiling summary statistics for each participant.
4. Saving Summary Statistics: Saving the summary statistics to a CSV file.
5. Plotting: Generating and saving various plots, including line plots, heatmaps, and histograms, to visualize the data.

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
import seaborn as sns

from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import pearsonr

from analysis_plotting_functions import   data_and_avg_line_plot, scatter_corr

warnings.simplefilter(action='ignore', category=FutureWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/avgshortestpath_final/'
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
g_measures_df['AvgShortestPath'] = g_measures_df['AvgShortestPath'].apply(ast.literal_eval).apply(np.array)


# Extract Data data from a column with array-elements into an own dfs
# each column on TS, each row one Part
diameter_array = np.array(g_measures_df['Diameter'].tolist())
avgshortpath_array = np.array(g_measures_df['AvgShortestPath'].tolist())

diameter_df = pd.DataFrame(diameter_array, columns=[i+1 for i in range(diameter_array.shape[1])])
avgshortpath_df = pd.DataFrame(avgshortpath_array, columns=[i+1 for i in range(avgshortpath_array.shape[1])])



################################### 2. Extracting summary statistics for particpants ###################################

parts_sum_measures = pd.DataFrame(columns=['ParticipantID',
                                            'MeanDiameter',
                                            'EndDiameter',
                                            'MaxDiameter',
                                            'MaxDiameterIndex',
                                            'MeanAvgShortPath',
                                            'EndAvgShortPath',
                                            'MaxAvgShortPath',
                                            'MaxAvgShortPathIndex'])


parts_sum_measures['ParticipantID'] = g_measures_df['Participant']

# Calculate summary statistics for each participant

parts_sum_measures['MeanDiameter'] = diameter_df.mean(axis=1).astype(float)
parts_sum_measures['EndDiameter'] = diameter_df.iloc[:, -1].astype(int) # last column
parts_sum_measures['MaxDiameter'] = diameter_df.max(axis=1).astype(int)
parts_sum_measures['MaxDiameterIndex'] = diameter_df.idxmax(axis=1).astype(int)

parts_sum_measures['MeanAvgShortPath'] = avgshortpath_df.mean(axis=1).astype(float)
parts_sum_measures['EndAvgShortPath'] = avgshortpath_df.iloc[:, -1].astype(float) # last column
parts_sum_measures['MaxAvgShortPath'] = avgshortpath_df.max(axis=1).astype(float)
parts_sum_measures['MaxAvgShortPathIndex'] = avgshortpath_df.idxmax(axis=1).astype(int)



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



################################### 4.1. Plotting AverageSHortestPath over time ###################################

data_and_avg_line_plot(avgshortpath_df,
                       x_label= 'Time Step',
                       y_label= 'Average Shortest Path',
                       fig_title = None, #'Temporal Evolution of Gaze-Graph AvgShortestPath',
                       savepath = savepath + f'AvgShortPath_ot',
                       color='#FF0000',
                       show = True,
                       save_dpi = save_dpi)



################################### 4.2. Plotting three Participants Diameter & Average Shortest Path over time for comparison ###################################

# Create subplots
fig, axes = plt.subplots(1, len(examples_idx), figsize=(15, 5), sharey=True, sharex=True)

# Plot each sample on its own subplot
for idx,sample_idx in enumerate(examples_idx):

    # Plot measures on each subplot
    axes[idx].plot(diameter_df.iloc[sample_idx], alpha=1, label='Diameter', color='#179999')
    axes[idx].plot(avgshortpath_df.iloc[sample_idx], alpha=1, label='AvgShortestPath', color='#103F71')

    axes[idx].set_title(f'ParticipantID {g_measures_df["Participant"].iloc[sample_idx]}')

    xticks = np.arange(10, len(diameter_df.columns) + 1, 20)
    #xticks = np.array([1] + xticks.tolist())
    axes[idx].set_xticks(xticks)
    axes[idx].tick_params(axis='x')

axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Path Length')
axes[0].legend(title='Measure', loc='upper right', fontsize=14)

#fig.suptitle('Temporal Evolution Comparison of Gaze-Graph Diameter and AvgShortestPath Lengths')

# Save and show the plot
plt.tight_layout()
plt.savefig(savepath + 'diameter_avgpath_plot',  dpi=save_dpi, bbox_inches='tight')
plt.show()





################################### 5. Statistics ###################################


################################### 5.1. Linear Correlation of Diameter and AvgShortPath curves ###################################

corr_list = []
p_list = []
for i in range(26):
    data_1 = avgshortpath_df.iloc[i]
    data_2 = diameter_df.iloc[i]

    correlation_coefficient = data_1.corr(data_2)
    corr, p = pearsonr (data_1, data_2)
    corr_list.append(corr)
    p_list.append(p)
    print(f'Correlation Coefficient:{corr}, {p}')


################################### 5.2. Linear Correlation between Max, Mean, End Values of Diameter & AvgShortest Path and MWU/tTest Group Tests ###################################

cols_diameter = ['EndDiameter','MaxDiameter', 'MeanDiameter']
cols_avgshortpath = ['EndAvgShortPath','MaxAvgShortPath', 'MeanAvgShortPath']
for col1, col2 in zip(cols_diameter,cols_avgshortpath):
    print(col1,col2)
    scatter_corr(x_data = parts_sum_measures[col1],
                y_data = parts_sum_measures[col2],
                x_label =col1,
                y_label =col2,
                fig_title = None, #f'Relationship of {col1} and {col2} Values',
                savepath = savepath + f'PartCorr_{col1}_{col2}_avgshortpath_diameter',
                regression= True,
                color = '#179999',
                show = True,
                save_dpi = save_dpi)

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
