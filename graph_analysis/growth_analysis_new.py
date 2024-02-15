import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from analysis_plots import smooth_line_plot,smooth_groups,categorical_barplot, violinplot, continuous_distribution_hist, data_and_avg_line_plot, heatmap, categorical_stacked_barplot, boxplot, scatter_corr, identify_peaks
from matplotlib.colors import LinearSegmentedColormap

warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Adjustable variables
savepath = 'D:/WestbrueckData/Analysis/Plots/explor_new/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]

g_measures_df = pd.read_csv('overview_graph_measures_new_explor.csv')


for col in g_measures_df.columns:
    if col != 'Participant':
        g_measures_df[col] = g_measures_df[col].apply(ast.literal_eval).apply(np.array)


g_measures_df[['NodeHits', 'UniqueNodeHits', 'NewNodes']] = g_measures_df['NewNodesMeasures'].apply(lambda x: pd.Series(zip(*x)))
g_measures_df[['EdgeHits', 'UniqueEdgeHits', 'NewEdges']] = g_measures_df['NewEdgesMeasures'].apply(lambda x: pd.Series(zip(*x)))


# Drop the original column with tuples if needed
g_measures_df = g_measures_df.drop('NewNodesMeasures', axis=1)
g_measures_df = g_measures_df.drop('NewEdgesMeasures', axis=1)

g_measures_df['NewNodesPercentage'] = ''
g_measures_df['EndNodesPercentage'] = ''
g_measures_df['MiddleNodesPercentage'] = ''

#g_measures_df['node_gradient'] =''
#g_measures_df['edge_gradient'] =''

for i in range(len(part_list)):

    num_nodes_ot = g_measures_df['NumNodes'].iloc[i]

    #session_over_time = g_measures_df['Session'].iloc[i]

    num_end_nodes = num_nodes_ot[149]
    num_middle_nodes = num_nodes_ot[74]

    g_measures_df['NewNodesPercentage'].iloc[i] = [a / num_end_nodes for a in g_measures_df['NewNodes'].iloc[i]]

    g_measures_df['EndNodesPercentage'].iloc[i] = num_end_nodes/244

    g_measures_df['MiddleNodesPercentage'].iloc[i] = num_middle_nodes/244

   


print('Percentage of Discovery:',g_measures_df['EndNodesPercentage'])
print('Percentage of MiddleDiscovery:',g_measures_df['MiddleNodesPercentage'])


def make_df(data_col):
    # Convert GrowthPercentage lists to a 2D numpy array
    data_array = np.array(data_col.tolist())

    # Create a DataFrame for boxplotting
    data_df = pd.DataFrame(data_array, columns=[i for i in range(data_array.shape[1])])

    return data_df



new_node_perc_df = make_df(g_measures_df['NewNodesPercentage'])

node_hits_df = make_df(g_measures_df['NodeHits'])
unique_node_df = make_df(g_measures_df['UniqueNodeHits'])
new_node_df = make_df(g_measures_df['NewNodes'])

edge_hits_df = make_df(g_measures_df['EdgeHits'])
unique_edge_df = make_df(g_measures_df['UniqueEdgeHits'])
new_edge_df = make_df(g_measures_df['NewEdges'])

new_rep_edge = new_edge_df - new_node_df
new_new_edge = new_node_df

node_new_div_all = new_node_df.div(node_hits_df)
node_new_div_unique = new_node_df.div(unique_node_df)

edge_new_div_all = new_edge_df.div(edge_hits_df)
edge_new_div_unique = new_edge_df.div(unique_edge_df)

node_er_df = new_node_df.div(unique_edge_df)

edge_er_df = new_edge_df.div(unique_edge_df)

comb_er_df = node_er_df + edge_er_df

#node_per_edge_df  = new_node_df.div(unique_edge_df)
#node_per_edge_df  = new_node_df.div(edge_hits_df)

node_er_grad_df = node_er_df.copy()
edge_er_grad_df = node_er_df.copy()

for i in range(len(part_list)):
    node_er_grad_df.iloc[i] = np.gradient(node_er_df.iloc[i])
    edge_er_grad_df.iloc[i] = np.gradient(edge_er_df.iloc[i])

data_and_avg_line_plot(node_er_grad_df,
                        x_label= 'Time Steps',
                        y_label= 'Discovery Rate Gradient',
                        fig_title = 'Discovery Rate Gradient of all Participants over time with Mean',
                        savepath = savepath + f'DiscoverRate_ot',
                        color='red',
                        show = True)

data_and_avg_line_plot(edge_er_grad_df,
                        x_label= 'Time Steps',
                        y_label= 'Exploration Rate Gradient',
                        fig_title = 'Exploration Rate Gradient of all Participants over time with Mean',
                        savepath = savepath + f'{i}_ExplorRate_ot',
                        color='red',
                        show = True)

time_sum_measures = pd.DataFrame(index = range(150),
                                 columns=['Timestep', 'MeanNodePercent',
                                          'Mean_NumNodesHits','Mean_NumUniqueNodes','Mean_NumNewNodes', 
                                          'Mean_NumEdgeHits','Mean_NumUniqueEdges','Mean_NumNewEdges',
                                          'MeanNodeDivAll','MeanNodeDivUnique',
                                          'MeanEdgeDivAll','MeanNodeDivAll',
                                          'DiscoveryRatio','ExplorationRatio'])

time_sum_measures['Timestep'] = new_node_perc_df.columns


continuous_distribution_hist(g_measures_df['EndNodesPercentage'],
                             x_label= 'Discovered Buildings [%]',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant End Discovery Percentage',
                             savepath = savepath + f'EndExplorVals_dist',
                             bins=13,
                             color='purple', 
                             show = False)
continuous_distribution_hist(g_measures_df['MiddleNodesPercentage'],
                             x_label= 'Discovered Buildings [%]',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Middle Dicovery Percentage',
                             savepath = savepath + f'MiddleExplorVals_dist',
                             bins=13,
                             color='purple', 
                             show = False)


for i,df in enumerate([new_node_perc_df, 
                       #node_hits_df, unique_node_df, new_node_df, 
                       #edge_hits_df, unique_edge_df, new_edge_df,
                       #node_new_div_all, node_new_div_unique,
                       #edge_new_div_all, edge_new_div_unique,
                       node_er_df, edge_er_df]):
   
    time_sum_measures.iloc[:,i+1] = df.mean(axis=0)

    
    data_and_avg_line_plot(df,
                        x_label= 'Time Steps',
                        y_label= 'Exploration Rate',
                        fig_title = 'Exploration Rate of all Participants over time with Mean',
                        savepath = savepath + f'{i}_ExplorValue_ot',
                        color='red',
                        show = False)
    
    smooth_line_plot(df,
                 x_label= 'Time Steps',
                 y_label= 'Diameter',
                 fig_title = 'Smoothed Diameter of all Participants over time with Mean',
                 savepath = savepath + f'Diameter_smooth_ot',
                 color='blue',
                 show = True)

'''
    data_and_avg_line_plot(df.cumsum(axis=1),
                        x_label= 'Time Steps',
                        y_label= 'Cummulative Exploration',
                        fig_title = f'{i}_Cummulative Exploration Rate of all Participants over time with Mean',
                        savepath = savepath + f'{i}_CumExplorRate_ot',
                        color='red',
                        show = False)
    
    # Create boxplots 

    boxplot(df,
            x_label = 'Time Step',
            y_label = 'Exploration Rate',
            fig_title = 'Distributions of Exploration Rate acorss participants over time with Mean Line',
            savepath = savepath + f'{i}_ExplorRate_Distributions_ot_violin',
            show = False)


    
    # Calculate the mean for each time step
    mean_values = np.mean(df, axis=0)
    # Calculate the differences from the mean for each participant
    difference_array = df - mean_values
    # Create a DataFrame for boxplotting
    mean_difference_df = pd.DataFrame(difference_array, columns=[i+1 for i in range(df.shape[1])])

    boxplot(mean_difference_df,
            x_label = 'Time Step',
            y_label = 'Exploration Rate Difference from individuals to mean',
            fig_title = 'Distributions of Exploration Rate-Difference to the Mean across participants over time with Mean Line',
            savepath = savepath + f'{i}_ExplorRateDiff_Distributions_ot_violin',
            show = False)
    

    
    violinplot(df[df > 0].T,
            x_label = 'Participant',
            y_label = 'Exploration Rate',
            fig_title = f'{i}_Distributions of Exploration Rate across time for each participant',
            savepath = savepath + f'{i}_ExplorationRate_Distributions_part',
            show = False)


    heatmap(df,
            x_label = 'Time Step',
            y_label = 'Participants',
            fig_title = 'Exploration Rate over time across Participants',
            savepath = savepath + f'{i}_exploration_heatplot',
            cmap='viridis',
            cbar=True,
            show = False)'''
'''
print(time_sum_measures)
for i in range(1,14):
    x_label= 'Time Steps',
    y_label= 'Exploration Rate',
    fig_title = 'Normalized Exploration Rate of all Participants over time with Mean',
    color='red',
    show = False
    plt.plot(time_sum_measures['Timestep'], time_sum_measures.iloc[:,i].values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    xticks = np.arange(0, 150, 10)  # Show every 10th time step
    plt.xticks(xticks)
    plt.legend()

    # Show the plot
    plt.savefig( savepath + f'{i}_Mean_ot',)
    
    if show == True:
        plt.show()
    plt.close()'''

#---------------------line plot nodes

for i, a in enumerate(['Mean_NumNodesHits','Mean_NumUniqueNodes','Mean_NumNewNodes']):
    label = ['Mean_AllHits','Mean_Unique','Mean_New']
    plt.plot(time_sum_measures['Timestep'], time_sum_measures[a].values, label = label[i])

x_label= 'Time Steps'
y_label= 'Num Nodes / Hits'
fig_title = 'Num of Nodes per TS averaged over Participants'
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(fig_title)
xticks = np.arange(0, 150, 10)  # Show every 10th time step
plt.xticks(xticks)
plt.legend()

# Show the plot
plt.savefig( savepath + f'Mean_ot_nodes',)
    
plt.show()
plt.close()

#---------------------line plot edges
for i, a in enumerate(['Mean_NumEdgeHits','Mean_NumUniqueEdges','Mean_NumNewEdges']):
    label = ['Mean_AllHits','Mean_Unique','Mean_New']
    plt.plot(time_sum_measures['Timestep'], time_sum_measures[a].values, label= label[i])
x_label= 'Time Steps'
y_label= 'Num Edges / Hits'
fig_title = 'Num of Edges per TS averaged over Participants'

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(fig_title)
xticks = np.arange(0, 150, 10)  # Show every 10th time step
plt.xticks(xticks)
plt.legend()

# Show the plot
plt.savefig( savepath + f'Mean_ot_edges',)
    
plt.show()
plt.close()

#---------------------line plot NER/EER
for a in ['ExplorationRatio']:#,'DiscoveryRatio']:
    plt.plot(time_sum_measures['Timestep'], time_sum_measures[a].values, label= a)

'''for i in range(1):
    plt.plot(time_sum_measures['Timestep'], node_er_df.iloc[i])
    plt.plot(time_sum_measures['Timestep'], edge_er_df.iloc[i])
    plt.plot(time_sum_measures['Timestep'], node_er_df.iloc[i]+edge_er_df.iloc[i])'''

x_label= 'Time Steps'
y_label= 'Exploration Ratio'
fig_title = 'Node, Edge, Exploration Ratios per TS averaged over Participants'
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(fig_title)
xticks = np.arange(0, 150, 10)  # Show every 10th time step
plt.xticks(xticks)
plt.legend()

# Show the plot
plt.savefig( savepath + f'{i}_Mean_ot_ExplorRatios_c',)
    
plt.show()
plt.close()

#---------------------line plot NER/EER
for i in range(1):
    plt.plot(time_sum_measures['Timestep'], edge_er_df.iloc[i], label = 'ExplorationRatio')
    plt.plot(time_sum_measures['Timestep'], node_er_df.iloc[i], label = 'DiscoveryRatio')

    #plt.plot(time_sum_measures['Timestep'], node_per_edge_df.iloc[i], label = 'DiscoveryRatio', alpha = 1)

    x_label= 'Time Steps'
    y_label= 'Exploration Ratio'
    fig_title = 'Node, Edge, Combined Exploration Ratios per TS for Part: 1004'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    xticks = np.arange(0, 150, 10)  # Show every 10th time step
    plt.xticks(xticks)
    plt.legend()

    # Show the plot
    plt.savefig( savepath + f'1004_ot_ExplorRatios_c',)
        
    plt.show()
    plt.close()


'''categorical_barplot(time_sum_measures[['Mean_NumNodesHits','Mean_NumUniqueNodes','Mean_NumNewNodes']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewNodes over time',
                            savepath = savepath + f'node_dist',
                            show = True)

categorical_barplot(time_sum_measures[['MeanNumEdgeHits','MeanNumUniqueEdges','MeanNumNewEdges']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewEdges over time',
                            savepath = savepath + f'edge_dist',
                            show = True)'''



time_sum_measures['Mean_OldNodes'] = time_sum_measures['Mean_NumUniqueNodes']-time_sum_measures['Mean_NumNewNodes']
time_sum_measures['Mean_OldEdges'] = time_sum_measures['Mean_NumUniqueEdges']-time_sum_measures['Mean_NumNewEdges']
time_sum_measures['Mean_NewNodes'] = time_sum_measures['Mean_NumNewNodes']
time_sum_measures['Mean_NewEdges'] = time_sum_measures['Mean_NumNewEdges']
categorical_stacked_barplot(time_sum_measures[['Mean_NewNodes','Mean_OldNodes']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Num. Unique Nodes hit per TS averaged over Participants',
                            savepath = savepath + f'node_dist_stacked_mean',
                            show = False)

categorical_stacked_barplot(time_sum_measures[['Mean_NewEdges','Mean_OldEdges']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Num. Unique Edges hit per TS averaged over Participants',
                            savepath = savepath + f'edge_dist_stacked_mean',
                            show = False)

# Concatenate the DataFrames as columns
#result_df = pd.concat([node_hits_df.head(1).T, unique_node_df.head(1).T, new_node_df.head(1).T], axis=1)
result_df = pd.concat([new_node_df.head(1).T, unique_node_df.head(1).T - new_node_df.head(1).T], axis=1)
# Optional: You may want to reset the index if needed
result_df.reset_index(inplace=True, drop=True)
result_df.columns = ['New','Old']

categorical_stacked_barplot(result_df,
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Num. Unique Nodes hit per TS for Part: 1004',
                            savepath = savepath + f'node_dist_1004',
                            show = False)


#result_df = pd.concat([edge_hits_df.head(1).T, unique_edge_df.head(1).T, new_edge_df.head(1).T], axis=1)
result_df = pd.concat([new_edge_df.head(1).T, unique_edge_df.head(1).T - new_edge_df.head(1).T], axis=1)
# Optional: You may want to reset the index if needed
result_df.reset_index(inplace=True, drop=True)
result_df.columns = ['New','Old']

categorical_stacked_barplot(result_df,
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Num. Unique Edges hit per TS for Part: 1004',
                            savepath = savepath + f'edge_dist_1004',
                            show = False)




#--------------------- NER and EER in categorical heatplots

category_mapping = {
    'no': 0,
    'low': 1,
    #'moderate': 2,
    'high': 2,
    #'very_high': 4,
}

mean_NER = node_er_df.mean(axis=1)
mean_EER = edge_er_df.mean(axis=1)

print(mean_NER, mean_EER)

categorized_NER_df = node_er_df.copy()
categorized_NER_df = categorized_NER_df.apply(lambda x: np.select(
    [
        (x > 0.80),
        (x > 0.60) & (x <=0.80),
        (x > 0.40) & (x <=0.60),
        #(x < participant_mean_when_explor-participant_std_when_explor) & (x <= participant_mean_when_explor+participant_std_when_explor),
        (x > 0.20) & (x <=0.40),
        (x > 0) & (x <=0.20),
        #(x > 0) & (x <= participant_mean_when_explor/2),
        (x <=0)
    ],
    [5,4,3,2,1,0],
    default=np.nan
))

categorized_EER_df = edge_er_df.copy()
categorized_EER_df = categorized_EER_df.apply(lambda x: np.select(
    [
        (x > mean_EER),
        #(x < participant_mean_when_explor-participant_std_when_explor) & (x <= participant_mean_when_explor+participant_std_when_explor),
        (x > 0) & (x <=mean_EER),
        #(x > 0) & (x <= participant_mean_when_explor/2),
        (x <= 0)
    ],
    [2,1,0],
    default=np.nan
))

categorized_comb_df = categorized_NER_df + categorized_EER_df # 4= both, 3= one of them high, 2= both a bit, 1= only one, 0 neither

custom_cmap = LinearSegmentedColormap.from_list('custom', [(0,'white'),(0.15, 'red'), (0.25, 'purple'), (1, 'darkblue')])
custom_cmap_explor = LinearSegmentedColormap.from_list('custom', [(0,'red'), (0.25, 'orange'),(0.5,'white'),(0.75,'blue'), (1, 'darkblue')])

heatmap(node_er_df,
        x_label = 'Time Step',
        y_label = 'Participants',
        fig_title = 'Dicovery Ratio over time across Participants',
        savepath = savepath + f'dicsRatio_heatplot',
        cmap=custom_cmap,
        cbar=True,
        show = True)
heatmap(edge_er_df,
        x_label = 'Time Step',
        y_label = 'Participants',
        fig_title = 'Exploration Rate over time across Participants',
        savepath = savepath + f'eer_heatplot',
        cmap=custom_cmap_explor,
        cbar=True,
        show = True)

heatmap(categorized_NER_df,
        x_label = 'Time Step',
        y_label = 'Participant',
        fig_title = 'Categorized Dicovery Ratio over time across Participants',
        savepath = savepath + f'cat_discratio_heatplot_b',
        cmap=['brown','purple','red','green', 'orange','blue'],
        cbar= False,
        show = True)

heatmap(categorized_EER_df,
        x_label = 'Time Step',
        y_label = 'Participants',
        fig_title = 'Exploration Intensities over time across Participants',
        savepath = savepath + f'cat_eer_heatplot',
        cmap=['white', 'pink', 'purple'],
        cbar= False,
        show = False)

'''heatmap(categorized_comb_df,
        x_label = 'Time Step',
        y_label = 'Participants',
        fig_title = 'Exploration Intensities over time across Participants',
        savepath = savepath + f'cat_comb_er_heatplot',
        cmap=['white', 'orange','red','purple', 'blue'],
        cbar= False,
        show = True)'''

participant_sum_measures = pd.DataFrame(index= range(26))

cat = ['No', 'VeryLow', 'Low','Moderate','High','VeryHigh']
for category in [0,1,2,3,4,5]:
    participant_sum_measures[f'{cat[category]}_disc_num'] = categorized_NER_df[categorized_NER_df == category].count(axis=1)
    #participant_sum_measures[f'{cat[category]}_expl_num'] = categorized_EER_df[categorized_EER_df == category].count(axis=1)
    participant_sum_measures[f'{cat[category]}_disc_avg'] = np.where((categorized_NER_df == category), np.arange(categorized_NER_df.shape[1]), 0).mean(axis=1)
    #participant_sum_measures[f'{cat[category]}_expl_avg'] = np.where((categorized_EER_df == category), np.arange(categorized_EER_df.shape[1]), 0).mean(axis=1)



categorical_stacked_barplot(participant_sum_measures[['VeryHigh_disc_num','High_disc_num','Moderate_disc_num',
                                                      'Low_disc_num','VeryLow_disc_num','No_disc_num']],
                            x_label = 'Participants',
                            y_label = 'Count',
                            fig_title = 'Number of TS with High,Low or No Discovery Rate',
                            savepath = savepath + f'Part_disc_num_cat',
                            show = True)

'''
categorical_stacked_barplot(participant_sum_measures[['High_expl_num','Low_expl_num','No_expl_num']],
                            x_label = 'Participants',
                            y_label = 'Count',
                            fig_title = 'Number of TS with High,Low or No Exploration Rate',
                            savepath = savepath + f'Part_expl_num_cat',
                            show = True)


continuous_distribution_hist(participant_sum_measures['High_disc_num'],
                             x_label= 'TS spend without exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploitation time',
                             savepath = savepath + f'TimeExploit_dist',
                             bins=10,
                             color='purple', 
                             show = True)

continuous_distribution_hist(participant_sum_measures['High_expl_num'],
                             x_label= 'TS spend with Exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploration time',
                             savepath = savepath + f'TimeExplor_dist',
                             bins=10,
                             color='purple', 
                             show = True)

continuous_distribution_hist(participant_sum_measures['High_expl_avg'],
                             x_label= 'TS spend with Exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploration time',
                             savepath = savepath + f'TimeExplor_dist',
                             bins=10,
                             color='purple', 
                             show = True)
'''
continuous_distribution_hist(participant_sum_measures['High_disc_avg'],
                             x_label= 'TS spend with Exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploration time',
                             savepath = savepath + f'TimeExplor_dist',
                             bins=10,
                             color='purple', 
                             show = True)

for list in [['VeryHigh_disc_num','High_disc_num','Moderate_disc_num','Low_disc_num','VeryLow_disc_num','No_disc_num'],
             ['High_disc_num','Low_disc_num','No_disc_num'],['High_expl_num','Low_expl_num','No_expl_num'],
             ['High_disc_avg','Low_disc_avg','No_disc_avg'],['High_expl_avg','Low_expl_avg','No_expl_avg']]:

    sns.violinplot(data = participant_sum_measures[list])

    # Customize labels and title
    plt.xlabel(f'Dicosvery Ratio Intervall')
    plt.ylabel(f'Count per Participant')
    plt.title(f'Distribution of TS spend with a certain Discovery Ratio Intervall')

    # Show the plot
    #plt.savefig(savepath + f'glob_indiv_lm_violin_3')
    plt.show()
    plt.close()

#-----------------------------------------------#
# Plot distribution of categorys over time and across participants
'''
categorical_stacked_barplot(participant_sum_measures[['0_BinCat','1_BinCat']],
                            x_label = 'Participants',
                            y_label = 'Count',
                            fig_title = 'Distribution of Binary Exploration Rate for each Participant',
                            savepath = savepath + f'Part_bin_expor_dist',
                            show = False)

categorical_stacked_barplot(participant_sum_measures[['no_cat','low_cat', 'high_cat']],
                            x_label = 'Participants',
                            y_label = 'Count',
                            fig_title = 'Distribution of Categorical Exploratiom Rate for each Participant',
                            savepath = savepath + f'Part_cat_expor_dist',
                            show = False)


categorical_stacked_barplot(time_sum_measures[['0_BinCat','1_BinCat']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of Binary Exploration Rate over time',
                            savepath = savepath + f'ot_bin_expor_dist',
                            show = False)

categorical_stacked_barplot(time_sum_measures[['no_cat','low_cat', 'high_cat']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of Categorical Exploration Rate over time',
                            savepath = savepath + f'ot_cat_expor_dist',
                            show = False)



continuous_distribution_hist(participant_sum_measures['0_BinCat'],
                             x_label= 'TS spend without exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploitation time',
                             savepath = savepath + f'TimeExploit_dist',
                             bins=10,
                             color='purple', 
                             show = False)

continuous_distribution_hist(participant_sum_measures['1_BinCat'],
                             x_label= 'TS spend with Exploration',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploration time',
                             savepath = savepath + f'TimeExplor_dist',
                             bins=10,
                             color='purple', 
                             show = False)

continuous_distribution_hist(participant_sum_measures['low_cat'],
                             x_label= 'TS spend with Low Exploration Rate ',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant High Exploration Rate Duration',
                             savepath = savepath + f'TimeExplorLow_dist',
                             bins=10,
                             color='purple', 
                             show = True)
continuous_distribution_hist(participant_sum_measures['high_cat'],
                             x_label= 'TS spend with High Exploration Rate',
                             y_label= 'Count',
                             fig_title = 'Distribution of Participant Exploration Rate Duration',
                             savepath = savepath + f'TimeExplorHigh_dist',
                             bins=10,
                             color='purple', 
                             show = True)
#correlation_coefficient_a = participant_sum_measures['1_BinCat'].corr(participant_sum_measures['MaxExplor'])
#correlation_coefficient_b = participant_sum_measures['1_BinCat'].corr(participant_sum_measures['MaxExplorIndex'])
#correlation_coefficient_c = participant_sum_measures['1_BinCat'].corr(participant_sum_measures['EndExplor'])
#correlation_coefficient_d = participant_sum_measures['1_BinCat'].corr(participant_sum_measures['MeanExplor'])

#print(correlation_coefficient_a, correlation_coefficient_b, correlation_coefficient_c, correlation_coefficient_d)


for col1 in ['MeanExplor','EndExplor','MaxExplor','high_cat','low_cat','no_cat', '1_BinCat']:
    for col2 in ['MeanExplor','EndExplor','MaxExplor','high_cat','low_cat','no_cat', '1_BinCat']:
        if col1 != col2:
            scatter_corr(x_data = participant_sum_measures[col1],
                        y_data = participant_sum_measures[col2],
                        x_label =col1,
                        y_label =col2,
                        fig_title = f'PartCorrelation {col1} and {col2}',
                        savepath = savepath + f'Partcorr_{col1}_{col2}',
                        show = True)


for col1 in ['Timestep']:
    for col2 in ['high_cat','low_cat','no_cat','1_BinCat']:
        if col2 == '1_BinCat':
            scatter_corr(x_data = time_sum_measures[col1],
                        y_data = time_sum_measures[col2],
                        x_label =col1,
                        y_label = 'Percentage of people Exploring',
                        fig_title = f'Raltion of the percentage of people Exploring the city at the time steps',
                        savepath = savepath + f'Relation_{col1}_{col2}',
                        show = True)
        if col2 == 'high_cat':
            scatter_corr(x_data = time_sum_measures[col1],
                        y_data = time_sum_measures[col2],
                        x_label =col1,
                        y_label = 'Percentage of people Exploring with a higher Rate than their Average',
                        fig_title = f'Relation of the percentage of people with a High Exploring Rate at the time steps',
                        savepath = savepath + f'Relation_{col1}_{col2}',
                        show = True)
        if col2 == 'low_cat':
            scatter_corr(x_data = time_sum_measures[col1],
                        y_data = time_sum_measures[col2],
                        x_label =col1,
                        y_label = 'Percentage of people exploring with a lower or average Exploration Rate',
                        fig_title = f'Relation of the percentage of people with a low Exploring Rate at the time steps',
                        savepath = savepath + f'Relation_{col1}_{col2}',
                        show = True)
        if col2 == 'no_cat':
            scatter_corr(x_data = time_sum_measures[col1],
                        y_data = time_sum_measures[col2],
                        x_label =col1,
                        y_label = 'Percentage of people not Exploring',
                        fig_title = f'Relation of the percentage of people not exploring the city at the time steps',
                        savepath = savepath + f'Relation_{col1}_{col2}',
                        show = True)


    plt.plot(time_sum_measures['high_cat'],alpha = 0.6, label = 'Explor.Rate > Avg')
    plt.plot(time_sum_measures['low_cat'],alpha = 0.6, label = 'Explor.Rate <= Avg')
    #plt.plot(time_sum_measures['no_cat'],alpha = 0.6, label = 'Explor.Rate == 0')
    plt.plot(time_sum_measures['1_BinCat'],alpha = 0.6, label = 'Explor.Rate > 0')
    
    window_size = 10

    # Berechnung des gleitenden Durchschnitts
    #smooth_values = df['Values'].rolling(window=window_size, center=True).mean()
    plt.plot(time_sum_measures['high_cat'].rolling(window=window_size, center=True).mean(),alpha = 0.8, label = 'Smoothed Line')
    plt.plot(time_sum_measures['low_cat'].rolling(window=window_size, center=True).mean(),alpha = 0.8, label = 'Smoothed Line')
    #plt.plot(time_sum_measures['no_cat'].rolling(window=window_size, center=True).mean(),alpha = 1, label = 'Smoothed Line')
    plt.plot(time_sum_measures['1_BinCat'].rolling(window=window_size, center=True).mean(),alpha =1, label = 'Smoothed Line')

   
    plt.xlabel('Timestep')
    plt.ylabel('% pf Participants')
    plt.title('% of certain Exploration Rate Categories over time')

    plt.legend()
    plt.show()
    #plt.close()



'''





















'''

#--------------------------------devided by unique nodes

time_sum_measures['Mean_NumOldNodesP'] = (time_sum_measures['Mean_NumUniqueNodes']-time_sum_measures['Mean_NumNewNodes']).div(time_sum_measures['Mean_NumUniqueNodes'])
time_sum_measures['Mean_NumOldEdgesP'] = (time_sum_measures['Mean_NumUniqueEdges']-time_sum_measures['Mean_NumNewEdges']).div(time_sum_measures['Mean_NumUniqueEdges'])
time_sum_measures['Mean_NumNewNodesP'] = time_sum_measures['Mean_NumNewNodes'].div(time_sum_measures['Mean_NumUniqueNodes'])
time_sum_measures['Mean_NumNewEdgesP'] = time_sum_measures['Mean_NumNewEdges'].div(time_sum_measures['Mean_NumUniqueEdges'])


categorical_stacked_barplot(time_sum_measures[['Mean_NumNewNodesP','Mean_NumOldNodesP']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewNodes over time',
                            savepath = savepath + f'node_dist_stacked',
                            show = True)

categorical_stacked_barplot(time_sum_measures[['Mean_NumNewEdgesP','Mean_NumOldEdgesP']],
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewNodes over time',
                            savepath = savepath + f'edge_dist_stacked_P',
                            show = True)

# Concatenate the DataFrames as columns
#result_df = pd.concat([node_hits_df.head(1).T, unique_node_df.head(1).T, new_node_df.head(1).T], axis=1)
result_df = pd.concat([new_node_df.head(1).T.div(unique_node_df.head(1).T), (unique_node_df.head(1).T - new_node_df.head(1).T).div(unique_node_df.head(1).T)], axis=1)
# Optional: You may want to reset the index if needed
result_df.reset_index(inplace=True, drop=True)
result_df.columns = ['New','Old']

categorical_stacked_barplot(result_df,
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewNodes over time',
                            savepath = savepath + f'node_dist_1004_P',
                            show = True)


#result_df = pd.concat([edge_hits_df.head(1).T, unique_edge_df.head(1).T, new_edge_df.head(1).T], axis=1)
result_df = pd.concat([new_edge_df.head(1).T.div(unique_edge_df.head(1).T), (unique_edge_df.head(1).T - new_edge_df.head(1).T).div(unique_edge_df.head(1).T)], axis=1)
# Optional: You may want to reset the index if needed
result_df.reset_index(inplace=True, drop=True)
result_df.columns = ['New','Old']

categorical_stacked_barplot(result_df,
                            x_label = 'Time Step',
                            y_label = 'Count',
                            fig_title = 'Distribution of NewEdges over time',
                            savepath = savepath + f'edge_dist_1004_P',
                            show = True)'''