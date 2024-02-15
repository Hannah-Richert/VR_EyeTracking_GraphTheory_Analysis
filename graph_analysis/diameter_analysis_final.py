import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import pearsonr

from analysis_plots import  violinplot,smooth_groups, smooth_line_plot, continuous_distribution_hist, data_and_avg_line_plot, heatmap, categorical_stacked_barplot, boxplot, scatter_corr, identify_peaks

warnings.simplefilter(action='ignore', category=FutureWarning)

# Adjustable variables
savepath = 'D:/WestbrueckData/Analysis/Plots/diameter_final/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]


########## 0. Load and prepaire data ##########

# load data
g_measures_df = pd.read_csv('overview_graph_measures_new.csv')


# convert read-in array-strings into arrays
g_measures_df['Diameter'] = g_measures_df['Diameter'].apply(ast.literal_eval).apply(np.array)
g_measures_df['AvgShortestPath'] = g_measures_df['AvgShortestPath'].apply(ast.literal_eval).apply(np.array)


# Extract Data data from a column with array-elements into an own dfs
# each column on TS, each row one Part
diameter_array = np.array(g_measures_df['Diameter'].tolist())
avgshortpath_array = np.array(g_measures_df['AvgShortestPath'].tolist())

diameter_df = pd.DataFrame(diameter_array, columns=[i+1 for i in range(diameter_array.shape[1])])
avgshortpath_df = pd.DataFrame(avgshortpath_array, columns=[i+1 for i in range(avgshortpath_array.shape[1])])



########## 1. Plotting Diameter (and AvgShortPath) over time ##########


data_and_avg_line_plot(diameter_df,
                       x_label= 'Time Step',
                       y_label= 'Diameter',
                       fig_title = 'Diameter of all Participants over time with Mean',
                       savepath = savepath + f'Diameter_ot',
                       color='blue',
                       show = False)


data_and_avg_line_plot(avgshortpath_df,
                       x_label= 'Time Step',
                       y_label= 'Average Shortest Path Length',
                       fig_title = 'Avg. Shortest Path Length of all Participants over time with Mean',
                       savepath = savepath + f'AvgShortPath_ot',
                       color='blue',
                       show = False)



########## 2. Plotting three Participants Diameter over time with Max. Diameter to show differences ##########

#examples_idx = [3,5,14,9,13,16]

examples_idx = [5,9,14]

plt.figure(figsize=(12, 6))


for i in examples_idx:
    plt.plot(diameter_df.iloc[i], alpha=0.75, label = g_measures_df['Participant'].iloc[i],zorder=1)

# scatter max diameter values on top of it
max_diameter = diameter_df.iloc[examples_idx].max(axis=1)
max_diameter_ts = diameter_df.iloc[examples_idx].idxmax(axis=1)

plt.scatter(x= max_diameter_ts, y=max_diameter,color='blue', marker='*', s=100, label = 'Max Values',zorder=2)

plt.xlabel('Time Step')
plt.ylabel('Diameter')
plt.title('Diameter over time & Max. Diameter of three Participants with varying Diameter development')
xticks = np.arange(10,len(diameter_df.columns)+1, 10)  # Show every 10th time step
xticks = np.array([1] + xticks.tolist()) # include the first one
plt.xticks(xticks)
plt.legend(title = 'PartID')

plt.savefig(savepath +'diameter_subset_ot')
plt.show()



########## 3. Summary Statistics for Particpants ##########


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

parts_sum_measures['MeanAvgShortPath'] = diameter_df.mean(axis=1).astype(float)
parts_sum_measures['EndAvgShortPath'] = diameter_df.iloc[:, -1].astype(float) # last column
parts_sum_measures['MaxAvgShortPath'] = diameter_df.max(axis=1).astype(float)
parts_sum_measures['MaxAvgShortPathIndex'] = diameter_df.idxmax(axis=1).astype(float)


########## 4. Plot of distributions of End Diameter ##########

continuous_distribution_hist(parts_sum_measures['EndDiameter'],
                             x_label= 'Diameter',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Diameter at t=150',
                             savepath = savepath + f'EndDiameter_dist',
                             bins=10,
                             color='orange', 
                             show = False)


########## 5. Scatterplot of Max Diameter & MaxIndex with Distributions/Marinals and coloured by End Diameter ##########


fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

# Scatter plot
ax_scatter = fig.add_subplot(gs[1, 0])
colors_plot = ['orange','green', 'blue']

for i, end_value in enumerate(sorted(parts_sum_measures['EndDiameter'].unique())):
    mask = parts_sum_measures['EndDiameter'] == end_value
    ax_scatter.scatter(parts_sum_measures['MaxDiameterIndex'][mask], parts_sum_measures['MaxDiameter'][mask], label=end_value, alpha = 0.5, color = colors_plot[i])

# Calculate mean values
mean_max_diameter = parts_sum_measures['MaxDiameter'].mean()
mean_max_diameter_idx = parts_sum_measures['MaxDiameterIndex'].mean()

# Add vertical and horizontal lines to the scatter plot
ax_scatter.axvline(mean_max_diameter_idx, color='purple', linestyle='-', label='Avg. Max. Diameter TS')
ax_scatter.axhline(mean_max_diameter, color='purple', linestyle='--', label='Avg. Max. Diameter')



# top histogram
ax_hist_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
hist_data_top = [parts_sum_measures['MaxDiameterIndex'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_top.hist(hist_data_top, bins=50, edgecolor='black', alpha=0.5, label=[7,8,9], color=colors_plot, stacked=True)

#ax_hist_top.hist(parts_sum_measures['MaxDiameterIndex'], bins=30, edgecolor='black', alpha=0.5, label=end_value, color='C0')  # Use the same color as scatter plot

ax_hist_top.set(yticklabels=[], xlabel='')
ax_hist_top.legend()
ax_hist_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


# Right histogram
ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
hist_data_right = [parts_sum_measures['MaxDiameter'][parts_sum_measures['EndDiameter'] == end_value] for end_value in [7,8,9]]
ax_hist_right.hist(hist_data_right, bins=50, edgecolor='black', alpha=0.5,orientation='horizontal', label=[7,8,9], color=colors_plot, stacked=True)

#ax_hist_right.hist(parts_sum_measures['MaxDiameter'], bins=30, edgecolor='black', orientation='horizontal', alpha=0.5, label=end_value, color='C0')  # Use the same color as scatter plot

ax_hist_right.set(xticklabels=[], ylabel='')
ax_hist_right.legend()
ax_hist_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)



# Add labels and legend
ax_scatter.set_xlabel('Timestep with Max. Diameter')
ax_scatter.set_ylabel('Max. Diameter')
ax_scatter.legend()

ax_scatter.legend(title='Diameter Value at TS=150')
#plt.title('Relationship of Max, MaxTS and End Diameter')
# Title
fig.suptitle('Relationship of Max, MaxTS and End Diameter', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(savepath + 'diameter_sum_measures_relationship')

plt.show()


########## 6. Visualization of the groups (A;B;C based on the scatter plot and the others ased on the end diameter)##########


A = set([10, 11, 12, 14, 15, 18, 19, 23, 24])
B = set([1, 2, 3, 5, 6, 7, 8, 20, 25])
C = set([0, 4, 9, 13, 16, 17, 21, 22])


seven = set([1,2,8,9,11,14,17,19,21,22,25])
eight = set([3,4,5,6,7,13,18])
nine = set([0,10,12,15,16,20,23,24])

smooth_groups(num_groups=3, 
              group_idx_list = [A,B,C],
              data_df = diameter_df, 
              subtitle_list = ['Group A: big, early Max Diameter',
                               'Group B: small, early Max Diameter',
                               'Group C: small, late Max Diameter'], 
                               x_label = 'Time Step', 
                               y_label = 'Diameter', 
                               fig_title = 'Diameter OT by Groups (based on Max Diameter-MaxDiameterIndex Relationship)')


smooth_groups(num_groups=3, 
              group_idx_list = [seven, eight, nine],
              data_df = diameter_df, 
              subtitle_list = ['End Diameter = 7',
                               'End Diameter = 8',
                               'End Diameter = 9'], 
                               x_label = 'Time Step', 
                               y_label = 'Diameter', 
                               fig_title = 'Diameter OT by Groups (based on End Diameter Value)')



########## 7. Correlation of Diameter/ GRaph Extend  and AvgShortPath / GRaph COmpactness ##########


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

continuous_distribution_hist(corr_list,
                             x_label= 'Pearson Correlation Value',
                             y_label= 'Count',
                             fig_title = 'Distribution of Correlation between ExplorRate and DiameterGradient',
                             savepath = savepath + f'corr_avgshortpath_diameter',
                             bins=50,
                             color='orange', 
                             show = True)

continuous_distribution_hist(p_list,
                             x_label= 'P-Value',
                             y_label= 'Count',
                             fig_title = 'Distribution of P-Value between ExplorRate and DiameterGradient',
                             savepath = savepath + f'p_avgshortpath_diameter',
                             bins=50,
                             color='orange', 
                             show = True)

 ############### 8. Correlation between Max, Min Values of Diameter & AvgShortest Path ###########


for col1 in ['EndDiameter','MaxDiameter']:
    for col2 in ['EndAvgShortPath','MaxAvgShortPath']:
        if col1 != col2:
            scatter_corr(x_data = parts_sum_measures[col1],
                        y_data = parts_sum_measures[col2],
                        x_label =col1,
                        y_label =col2,
                        fig_title = f'Realtionship {col1} and {col2}',
                        savepath = savepath + f'PartCorr_{col1}_{col2}_avgshortpath_diameter',
                        regression= True,
                        show = True)
            
            print(col1,col2)

            
            if col1 == 'EndDiameter':
                # Assuming dist1 and dist2 are your two distributions
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





            sns.violinplot(x=parts_sum_measures[col1], y= parts_sum_measures[col2])
            #sns.scatterplot(x = participant_sum_measures[col1], y= participant_sum_measures[col2], color='blue', marker='.', s=100, label= 'Max Value')

            # Customize labels and title
            plt.legend()
            plt.xlabel(f'{col1}')
            plt.ylabel(f'{col2}')
            plt.title(f'Relationship {col1} and {col2}')

            # Show the plot

            plt.savefig(savepath + f'PartCorrViolin_{col1}_{col2}_avgshortpath_diameter')

            plt.show()
            plt.close()

########### #####################
print('End')

