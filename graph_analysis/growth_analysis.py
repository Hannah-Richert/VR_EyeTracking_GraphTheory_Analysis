"""
This script performs data analysis and visualization of the GROWTH measures (Nodes, Edges, Exploration Rate & Discovery Rate). 
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
from matplotlib.colors import LinearSegmentedColormap

from analysis_plotting_functions import distribution_hist, data_and_avg_line_plot, heatmap

warnings.simplefilter(action='ignore', category=FutureWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/growth_final/'
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

g_measures_df = pd.read_csv('overview_graph_measures_final.csv')

for col in g_measures_df.columns:
    if col != 'Participant':
        g_measures_df[col] = g_measures_df[col].apply(ast.literal_eval).apply(np.array)

def make_df(data_col):
    # Convert array column into a 2D numpy array
    data_array = np.array(data_col.tolist())

    # Create a DataFrame for boxplotting
    data_df = pd.DataFrame(data_array, columns=[i+1 for i in range(data_array.shape[1])])

    return data_df

unique_node_df = make_df(g_measures_df['NumNodesInLastSeg'])
new_node_df = make_df(g_measures_df['NumNewNodesInLastSeg'])

unique_edge_df = make_df(g_measures_df['NumEdgesInLastSeg'])
new_edge_df = make_df(g_measures_df['NumNewEdgesInLastSeg'])

discovery_rate_df = new_node_df.div(unique_edge_df)
exploration_rate_df = new_edge_df.div(unique_edge_df)

diameter_df = make_df(g_measures_df['Diameter'])



################################### 2. Extracting summary statistics for particpants ###################################

parts_sum_measures = pd.DataFrame(columns=['ParticipantID',
                                           'EndDiameter',
                                            'MeanExplorationRate',
                                            'MeanDiscoveryRate',
                                            'NumEndNodes',
                                            'NumEndEdges'])


parts_sum_measures['ParticipantID'] = g_measures_df['Participant']

# Calculate summary statistics for each participant
parts_sum_measures['EndDiameter'] = diameter_df.iloc[:, -1].astype(int)

parts_sum_measures['MeanDiscoveryRate'] = discovery_rate_df[discovery_rate_df > 0].mean(axis=1).astype(float)
parts_sum_measures['MeanExplorationRate'] = exploration_rate_df[exploration_rate_df > 0].mean(axis=1).astype(float) 
parts_sum_measures['NumEndNodes'] = new_node_df.cumsum(axis=1).iloc[:, -1].astype(int) # last column
parts_sum_measures['NumEndEdges'] = new_edge_df.cumsum(axis=1).iloc[:, -1].astype(int) # last column



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



################################ 4. Plotting ###################################

# gloabal settings for all figures
mpl.rcParams.update({'font.size': 16,  # for normal text
                     'axes.labelsize': 16,  # for axis labels
                     'axes.titlesize': 16,  # for title
                     'xtick.labelsize': 14,  # for x-axis tick labels
                     'ytick.labelsize': 14})  # for y-axis tick labels



################################### 4.1. Plotting number of Nodes and Edges over time ###################################

data_and_avg_line_plot(new_node_df.cumsum(axis=1),
                    x_label= 'Time Steps',
                    y_label= 'Number of Nodes',
                    fig_title = None, #'Temporal Evolution of Number of Gaze-Graph Nodes',
                    savepath = savepath + f'Nodes_ot',
                    color='#FF0000',
                    show = False,
                    save_dpi=save_dpi)


data_and_avg_line_plot(new_edge_df.cumsum(axis=1),
                    x_label= 'Time Steps',
                    y_label= 'Number of Edges',
                    fig_title = None, #'Temporal Evolution of Number of Gaze-Graph Edges',
                    savepath = savepath + f'Edges_ot',
                    color='#FF0000',
                    show = False,
                    save_dpi=save_dpi)



################################### 4.2 Discovery Rate and Exploration Rate over time with heatplots ###################################

discovery_rate_cmap = LinearSegmentedColormap.from_list('custom', [(0,'#ffffff'),(0.15, '#66b6b5'),(0.3, '#33718d'),(1, '#103f71')])

heatmap(discovery_rate_df,
        x_label = 'Time Step',
        y_label = 'Participant Indices',
        fig_title = None, #'Temporal Evolution of the Dicovery Rate',
        savepath = savepath + f'DiscRate_heatplot',
        cmap=discovery_rate_cmap,
        cbar=True,
        show = True,
        save_dpi=save_dpi)


custom_cmap_explor = LinearSegmentedColormap.from_list('custom', [(0,'#FF0000'),(0.25,'#fbac63'),(0.5,'#ffffff'),(0.75, '#69a2ae'), (1, '#103f71')])


heatmap(exploration_rate_df,
        x_label = 'Time Step',
        y_label = 'Participant Indices',
        fig_title = None, #'Temoral Evolution of the Exploration Rate',
        savepath = savepath + f'ExplorRate_heatplot',
        cmap=custom_cmap_explor,
        cbar=True,
        show = True,
        save_dpi=save_dpi)



################################### 4.3. Plot of distributions of number of nodes/edges at the end ###################################

distribution_hist(parts_sum_measures['NumEndNodes'],
                             x_label= 'Number of Nodes',
                             y_label= 'Count',
                             fig_title = None, #'Distribution of Number of EndNodes',
                             savepath = savepath + f'end_nodes_dist',
                             bins=10,
                             color='#103F71', 
                             show = True,
                             save_dpi=save_dpi)

distribution_hist(parts_sum_measures['NumEndEdges'],
                             x_label= 'Number of Edges',
                             y_label= 'Count',
                             fig_title = None, #'Distribution of Number of EndEdges',
                             savepath = savepath + f'end_edges_dist',
                             bins=10,
                             color='#103F71', 
                             show = True,
                             save_dpi=save_dpi)


################################### 4.4. Example-Participants' Discovery & Exploration Rate with the Diameter ###################################

# Create subplots
fig, axes = plt.subplots(1, len(examples_idx), figsize=(15, 5), sharey=False, sharex=False)
colors_plot = ['#103F71','#179999','#E1A315']
# Plot each sample on its own subplot
for idx,sample_idx in enumerate(examples_idx):
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    # Plot measures on each subplot
    ax1.plot(discovery_rate_df.iloc[sample_idx], alpha=0.9, label='Discovery Rate' if idx == 0 else '', color='#179999')
    ax1.plot(exploration_rate_df.iloc[sample_idx], alpha=0.9, label='Exploration Rate' if idx == 0 else '', color='#E1A315')
    ax2.plot(diameter_df.iloc[sample_idx], alpha=0.9, label='Diameter' if idx == 0 else '', color='#103F71')


    # Customize subplot
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Rate Value')
    ax2.set_ylabel('Diameter', color='#103F71')

    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y',color='#103F71')

    xticks = np.arange(10, len(diameter_df.columns) + 1, 20)
    #xticks = np.array([1] + xticks.tolist())
    ax1.set_xticks(xticks, labels=xticks, rotation=45)

    ax1.set_title(f'Participant {g_measures_df["Participant"].iloc[sample_idx]}')
    
#fig.suptitle('Temporal Evolution Discovery Rate, Exploration Rate and Diameter of Three Participants')
fig.legend(title='Measure', loc=(0.79, 0.65), fontsize= 14)

# Save and show the plot
plt.tight_layout()
plt.savefig(savepath + 'diameter_disc_explor_plot', dpi=save_dpi, bbox_inches='tight')
plt.show()



###################################

print('End')
