"""
This script performs data analysis and visualization on the gaze-graph-defined LANDMARKS of gaze-graphs for a set of participants.
Custom functions from analysis_plotting_functions are used for some plotting tasks.
The script will save plots and summary statistics in the specified directories.

The script consists of the following main sections:
1. Adjustable Variables: Define file paths, participant lists, common landmarks, number of nodes, and figure resolution.
2. Load and Prepare Part of the Data: Load data from CSV files and prepare necessary DataFrames.
3. Extracting Landmarks and Summary Statistics for Participants: Get the landmarks from the node-degree distributions.
4. Plotting: Generate various plots to visualize the data.
5. Statistics: Perform statistical tests to analyze the relationships and differences between summary statistics.

Adjustable Variables:
    - savepath (str): Path to the directory where figures will be saved.
    - node_pos_path (str): Path to the directory where node positions are stored.
    - summary_file_path (str): Path to the directory where a summary file is stored and saved.
    - part_list (list): List of all participants.
    - common_lm (list): List of common landmarks in the environment.
    - num_nodes (int): Total number of nodes/buildings in the environment.
    - save_dpi (int): Resolution of saved PNGs (recommended: 300 or 600).

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

from analysis_plotting_functions import  categorical_stacked_barplot

warnings.simplefilter(action='ignore', category=FutureWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/lm_final/'
node_pos_path = 'D:/WestbrueckData/Pre-processing/'
summary_file_path = 'D:/WestbrueckData/Analysis/parts_summary_stats.csv'

os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]

# common Lm in westbrook, based on analysis by Jasmin L. Walter
common_lm = ['Building_154','Building_176','TaskBuilding_35','Building_214','HighSilo-TaskBuilding_49']

# absolute number of nodes/  buildings in the environment
num_nodes = 244

# resolution of any saved figures
save_dpi = 300 #600 




################################### 1. Load and prepaire part of the data ###################################

g_measures_df = pd.read_csv('overview_graph_measures_final.csv')

# Hierarchy Index to determine if hierarchical structure exitsts at t => if some nodes are more important => graph defined landmarks
g_measures_df['HierarchyIndex'] = g_measures_df['HierarchyIndex'].apply(ast.literal_eval).apply(np.array)

h_idx_data = g_measures_df['HierarchyIndex']
h_idx_array = np.array(h_idx_data.tolist())
h_idx_df = pd.DataFrame(h_idx_array, columns=[i for i in range(h_idx_array.shape[1])])

# diameter, for later comparison
g_measures_df['Diameter'] = g_measures_df['Diameter'].apply(ast.literal_eval).apply(np.array)
diameter_data = g_measures_df['Diameter']
diameter_array = np.array(diameter_data.tolist())
diameter_df = pd.DataFrame(diameter_array, columns=[i for i in range(diameter_array.shape[1])])



################################### 2. Extracting Landmarks and summary statistics for particpants ###################################

parts_sum_measures = pd.DataFrame(index=range(26), columns = ['ParticipantID','CommonLM','IndivLM','TempLM','EndLM','EndDiameter'])

parts_sum_measures['ParticipantID'] = g_measures_df['Participant']
parts_sum_measures['EndDiameter'] = diameter_df.iloc[:,-1]

building_status_df = pd.DataFrame(index=['TempLM','EndLM'])

all_landmarks = []
end_landmarks = []

# for each participants iterate though node degree over time and determine landmarks
for i, part in enumerate((part_list)):

    measures_df = pd.read_csv(f'{part}_node_measurements.csv')
    measures_df.fillna(0, inplace=True)

    node_measure_df = measures_df[measures_df['Measure'] == 'NodeDegree'].copy()
   
    node_measure_df = node_measure_df.iloc[:,2:-1]
    node_measure_df.astype(int)
    
    
    # Identifying degree distribution at each time step to determine threshold for nodes being Gaze-graph-defined landmarks
    degree_means_per_ts = node_measure_df[node_measure_df > 0].mean(axis=1)
    degree_std_per_ts = node_measure_df[node_measure_df > 0].std(axis=1)
    degree_2sigma_threshold_per_ts = 2 * degree_std_per_ts


    # each node gets a category: a regular building [1] or a landmark [2], if the degree is above the threshold
    category_landmark_df = node_measure_df.copy()
    category_landmark_df = category_landmark_df.apply(lambda x: np.select(
    [
        (x > degree_means_per_ts+degree_2sigma_threshold_per_ts),
        (x > 0) & (x <= degree_means_per_ts+degree_2sigma_threshold_per_ts)
    ],
    [2,1],
    default=np.nan))

    # if the hierarchy index is to low, we set the landmarks back to regular buildings
    for j in range(150):
        hier_idx = h_idx_df.iloc[i,j]

        # no building is a landmark if no hierarchy exists
        if hier_idx < 2:
            category_landmark_df.iloc[j, :] = 1


    all_landmarks_part = category_landmark_df.columns[category_landmark_df.eq(2).any()]
    end_landmarks_part = category_landmark_df.columns[category_landmark_df.iloc[-1].eq(2)]
    
    
    parts_sum_measures.iloc[i,1] = len(set(end_landmarks_part).intersection(set(common_lm))) # CommonLM
    parts_sum_measures.iloc[i,2] = len(set(end_landmarks_part)) - parts_sum_measures.iloc[i,1] # IndivLM
    parts_sum_measures.iloc[i,3] = len(set(all_landmarks_part)) -len(set(end_landmarks_part)) # TempLM
    parts_sum_measures.iloc[i,4] = len(set(end_landmarks_part)) # EndLM


    # add the lm status for each building from this partipcipant
    all_present_buildings = set(all_landmarks_part)

    # Check if any new buildings need to be added
    new_buildings = all_present_buildings - set(building_status_df.columns)
    if new_buildings:
        # Create a DataFrame with zeros for new buildings
        new_buildings_df = pd.DataFrame(0, index=building_status_df.index, columns=list(new_buildings))
        # Concatenate the new DataFrame with the existing one
        building_status_df = pd.concat([building_status_df, new_buildings_df], axis=1)

    # Increment counts for each building
    for building in set(all_landmarks_part):
        
        if building in set(end_landmarks_part):
            building_status_df.at['EndLM', building] += 1
        else:
            building_status_df.at['TempLM', building] += 1

    # save landmarks in lists
    end_landmarks.append(set(end_landmarks_part))
    all_landmarks.append(set(all_landmarks_part))



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



################################### 4.1. LM Status of Buildings across all participants ###################################

building_status_df = building_status_df[building_status_df.loc['EndLM'].sort_values().index]

categorical_stacked_barplot(building_status_df.T,
                            x_label = 'Building',
                            y_label = 'Count',
                            colours = ['#E1A315', '#103F71'],
                            fig_title = None, #'Amount of Gaze-Graphs with certain Building as a LM', 
                            savepath = savepath + 'all_lm_status_hist', 
                            show = True,
                            save_dpi = save_dpi)


common_status_df = building_status_df[common_lm]

categorical_stacked_barplot(common_status_df.T,
                            x_label = 'Building',
                            y_label = 'Count',
                            colours = ['#E1A315','#103F71'],
                            fig_title = None, #'Amount of Gaze-Graphs with certain CommonLM-Buildings as a LM', 
                            savepath = savepath + 'common_lm_status_hist', 
                            show = True,
                            save_dpi = save_dpi)



################################### 4.2. Status of Buildings overall ###################################

all_lm = set([x for sublist in all_landmarks for x in sublist])
end_lm = set([x for sublist in end_landmarks for x in sublist])

indiv_lm = [x for x in end_lm if x not in common_lm]
temp_lm = [x for x in all_lm if x not in end_lm]

num_common_lm = len(common_lm)
num_end_lm = len(end_lm)
num_indiv_lm = len(indiv_lm)
num_temp_lm = len(temp_lm)


# Example data
labels = ['CommonLM', 'IndivLM', 'TempLM', 'Never a LM']
sizes = [num_common_lm, num_indiv_lm, num_temp_lm, num_nodes-num_temp_lm-num_end_lm]

# Create a pie chart with numbers
plt.pie(sizes, labels=labels,  autopct=lambda p: f'{p:.1f}%\n({p * sum(sizes) / 100:.0f})',
         startangle=90, colors=['#179999','#6555CB','#E1A315','lightgrey'])



#plt.title('Buildings in Westbrook - graph-defined Landmarks')

# Display the pie chart
plt.savefig(savepath + 'landmark_types', dpi=save_dpi, bbox_inches='tight')
plt.show()



################################### 4.3. Building Status on the WB Map ###################################

# load data
wb_image = plt.imread('map_natural_500mMarker.png') 
node_data = pd.read_csv(node_pos_path + "node_positions.csv")


# assign each status to the node/building in the data
node_data['LM_category'] = 0

node_data.loc[node_data['hitObjectColliderName'].isin(common_lm), 'LM_category'] = 3
node_data.loc[node_data['hitObjectColliderName'].isin(indiv_lm), 'LM_category'] = 2
node_data.loc[node_data['hitObjectColliderName'].isin(temp_lm), 'LM_category'] = 1

#['#3D3D9C','#6A5ACD','#857DE6','#9991FA']

fig,ax = plt.subplots(figsize=(8.8, 6.2))

colors = ['lightgrey','#E1A315','#6555CB','#179999']
markers = ['o', 's', '^', 'D']
labels = ['Never a LM','TempLM', 'IndivLM','CommonLM' ]

    
ax.imshow(wb_image, extent=[node_data['pos_x'].min()-17,node_data['pos_x'].min() + 969.988,
                            node_data['pos_z'].min()-65, node_data['pos_z'].min()+693.07], alpha=1)


for i, category in enumerate([0,1,2,3]):
    category_data = node_data[node_data['LM_category'] == category]
    plt.scatter(category_data['pos_x'], category_data['pos_z'], 
                c=colors[i], marker=markers[i], 
                s=42, alpha=1, label=labels[i], edgecolor= '#103F71')
    
# Create a scatter plot
#plt.scatter(node_data['pos_x'], node_data['pos_z'], c=node_data['LM_category'].map(color_mapping),  s=40, alpha=1, edgecolor='#00008B')

# Add labels and title
ax.set_xlim(node_data['pos_x'].min()-17,node_data['pos_x'].min() + 969.988)
ax.set_ylim(node_data['pos_z'].min()-65, node_data['pos_z'].min()+693.07)
plt.xticks([])
plt.yticks([])
plt.legend()
#plt.title('Map of Building Landmark Status')

# Show the plot
plt.savefig(f'{savepath}lm_map', dpi=save_dpi, bbox_inches='tight')
plt.show()




################################### 4.4. Common and individual Landmarks ###################################

sorted_df = parts_sum_measures.sort_values(by='CommonLM')
sorted_df.index = sorted_df['ParticipantID']

print(sorted_df[['CommonLM','IndivLM']].index)

categorical_stacked_barplot(sorted_df[['CommonLM','IndivLM']],
                            x_label = 'ParticipantID',
                            y_label = 'Count',
                            colours = ['#179999','#6555CB'],
                            fig_title = 'Amount of Global and Individual Landmarks per Participant',
                            savepath = savepath + f'Part_lm_dist',
                            show = True,
                            save_dpi = save_dpi)





################################### 5. Statistics ###################################

################################### 5.1 Correlation / Comparison LM and EndDiameter ###################################


for col1 in ['EndDiameter',]:
    for col2 in ['CommonLM','IndivLM','EndLM']:

        if col1 == 'EndDiameter':
            # Assuming dist1 and dist2 are your two distributions
            for i in [(7,8),(8,9),(9,7)]:
                a,b = i
                print(a,b)
                print(parts_sum_measures[parts_sum_measures['EndDiameter']==a][col2],
                                parts_sum_measures[parts_sum_measures['EndDiameter']==b][col2])
                t_stat, p_value = ttest_ind(parts_sum_measures[parts_sum_measures['EndDiameter']==a][col2].astype(int),
                                            parts_sum_measures[parts_sum_measures['EndDiameter']==b][col2].astype(int))

                if p_value < 0.05:
                    print("The distributions are significantly different.",p_value)
                else:
                    print("The distributions are not significantly different.",p_value)

                # Assuming dist1 and dist2 are your two distributions
                stat, p_value = mannwhitneyu(parts_sum_measures[parts_sum_measures['EndDiameter']==a][col2].astype(int),
                                            parts_sum_measures[parts_sum_measures['EndDiameter']==b][col2].astype(int))

                if p_value < 0.05:
                    print("The distributions are significantly different.",p_value)
                else:
                    print("The distributions are not significantly different.",p_value)

            plt.figure(figsize=(7.5, 5.5))
            sns.boxplot(x=parts_sum_measures[col1], y= parts_sum_measures[col2].astype(float), palette=['#103F71','#179999','#E1A315'])
           
            # Customize labels and title
            plt.legend()
            plt.xlabel(f'{col1}')
            plt.ylabel(f'{col2}')
            #plt.title(f'Relationship {col1} and {col2}')

            # Show the plot

            plt.savefig(savepath + f'PartCorr_{col1}_{col2}', dpi=save_dpi, bbox_inches='tight')

            plt.show()
            plt.close()



################################### 5.2. Save LM-stats (how many participants have it as a endLM) of individual buildings in participants for performance ###################################
            
end_lm_df = pd.DataFrame(index=range(26),columns = list(set(end_lm)))


for i,lm_list in enumerate(end_landmarks):
    for lm in end_lm:

        if lm in lm_list:
            end_lm_df.loc[i,lm] = 1
        else:
            end_lm_df.loc[i,lm] = 0


end_lm_df.to_csv(summary_file_path + 'end_lm.csv', index=False)



###################################

print('End')

