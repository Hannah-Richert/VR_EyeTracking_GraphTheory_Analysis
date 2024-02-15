import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.optimize import curve_fit
from upsetplot import UpSet, from_contents
from scipy.stats import ttest_ind, mannwhitneyu
#import holoviews as hv
#from holoviews import opts, dim
#from bokeh.sampledata.les_mis import data

#hv.extension('bokeh')
#hv.output(size=200)

# Your piecewise functionconda a

from analysis_plots import  smooth_line_plot, grouped_barplot, violinplot, continuous_distribution_hist, data_and_avg_line_plot, heatmap, categorical_stacked_barplot, boxplot, scatter_corr, identify_peaks

warnings.simplefilter(action='ignore', category=FutureWarning)

# Adjustable variables
savepath = 'D:/WestbrueckData/Analysis/Plots/graph_hierarchy/'
pos_path = 'D:/WestbrueckData/Pre-processing/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]
#part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1079, 1080]

g_measures_df = pd.read_csv('overview_graph_measures_new.csv')
for col in g_measures_df.columns:
    if col != 'Participant':
        g_measures_df[col] = g_measures_df[col].apply(ast.literal_eval).apply(np.array)

# Extract GrowthPercentage data
h_idx_data = g_measures_df['HierarchyIndex']
#h_slope_data = g_measures_df['h_slope']
# Convert GrowthPercentage lists to a 2D numpy array
h_idx_array = np.array(h_idx_data.tolist())
#h_slope_array = np.array(h_slope_data.tolist())
# Create a DataFrame for boxplotting
h_idx_df = pd.DataFrame(h_idx_array, columns=[i for i in range(h_idx_array.shape[1])])
#h_slope_df = pd.DataFrame(h_slope_array, columns=[i for i in range(h_slope_array.shape[1])])

print(h_idx_df[149])
data_and_avg_line_plot(h_idx_df,
                       x_label= 'Time Steps',
                       y_label= 'Diameter',
                       fig_title = 'Diameter of all Participants over time with Mean',
                       savepath = savepath + f'Diameter_ot',
                       color='red',
                       show = False)

'''
# Create boxplots 

boxplot(h_idx_df,
        x_label = 'Time Step',
        y_label = 'Diameter',
        fig_title = 'Distributions of Diameter across participants over time with Mean Line',
        savepath = savepath + f'Diameter_Distributions_ot',
        show = True)


violinplot(h_idx_df.T,
        x_label = 'Participant',
        y_label = 'Diameter Gradient',
        fig_title = 'Distributions of Diameter Grad across time for each participant with Mean Line',
        savepath = savepath + f'DiameterGrad_Distributions_part',
        show = False)'''



# node analysis

num_nodes = 244
num_glob_lm = 5
num_indiv_lm = 0
num_temp_lm = 0

sum_lm_df = pd.DataFrame(index=range(26), columns = ['GlobalLM','IndivLM','TempLM','DiscoveredBuildings','EndLM',
                                                     'MeanGlobalLMDuration', 'MeanIndivLMDuration','MeanLMDuration',
                                                     'MeanGlobalLMfirstTS','MeanIndivLMfirstTS','MeanLMfirstTS'])

lm_duration_df = pd.DataFrame(index=range(26))
lm_start_df = pd.DataFrame(index=range(26))

lm_count_ot_df = pd.DataFrame(index=range(26), columns = range(150))

end_lm_per_person = []
landmark_per_person = []
global_landmarks = ['Building_154','Building_176','TaskBuilding_35','Building_214','HighSilo-TaskBuilding_49']



for i, part in enumerate((part_list)):

    measures_df = pd.read_csv(f'{part}_node_measurements.csv')
    measures_df.fillna(0, inplace=True)

    node_measure_df = measures_df[measures_df['Measure'] == 'NodeDegree'].copy()
   
    node_measure_df = node_measure_df.iloc[:,2:-1]
    node_measure_df.astype(int)

    degree_means_per_ts = node_measure_df[node_measure_df > 0].mean(axis=1)
    degree_std_per_ts = node_measure_df[node_measure_df > 0].std(axis=1)
    degree_2sigma_threshold_per_ts = 2 * degree_std_per_ts


    category_landmark_df = node_measure_df.copy()
    category_landmark_df = category_landmark_df.apply(lambda x: np.select(
    [
        (x > degree_means_per_ts+degree_2sigma_threshold_per_ts),
        #(x < participant_mean_when_explor-participant_std_when_explor) & (x <= participant_mean_when_explor+participant_std_when_explor),
        (x > 0) & (x <= degree_means_per_ts+degree_2sigma_threshold_per_ts),
        #(x > 0) & (x <= participant_mean_when_explor/2),
        (x == 0)
    ],
    [2,1,0],
    default=np.nan))

    for j in range(150):
        hier_idx = h_idx_df.iloc[i,j]
        if hier_idx < 2:
            category_landmark_df.iloc[j, :] = -1
            
        

    

    landmarks = category_landmark_df.columns[category_landmark_df.eq(2).any()]

    final_landmarks = category_landmark_df.columns[category_landmark_df.iloc[-1].eq(2)]
    


    sum_lm_df.iloc[i,0] = len(set(final_landmarks).intersection(set(global_landmarks)))
    sum_lm_df.iloc[i,1] = len(set(final_landmarks)) - sum_lm_df.iloc[i,0]
    sum_lm_df.iloc[i,2] = len(set(landmarks)) -len(set(final_landmarks))
    sum_lm_df.iloc[i,3] = len(category_landmark_df.columns)
    sum_lm_df.iloc[i,4] = len(set(final_landmarks))


    end_lm_per_person.append(set(final_landmarks))
    landmark_per_person.append(set(landmarks))
    
    category_landmark_global_df = pd.DataFrame(index = range(150), columns= global_landmarks)
    for glm in global_landmarks:
        try:
            category_landmark_global_df[glm] = category_landmark_df[glm]
        except Exception as e:
            category_landmark_global_df[glm] = 0




    #category_landmark_df = category_landmark_global_df
    
    category_landmark_df = category_landmark_df[final_landmarks]
    
    lm_count_ot_df.iloc[i] = category_landmark_df.eq(2).sum(axis=1)

    

    for col in category_landmark_df.columns:
        # Count occurrences of "2" in each column
        count = category_landmark_df[col].eq(2).sum()
        first_occurence = category_landmark_df.index[category_landmark_df[col] == 2].min()
    

        if col not in lm_duration_df.columns:
            
            #lm_duration_df.iloc[i] = pd.concat([lm_duration_df.iloc[i], pd.DataFrame({col: count})], axis=1)

            empty_column = pd.Series([''] * len(lm_duration_df), name=col)
            # Add the empty column to the DataFrame
            lm_duration_df = pd.concat([lm_duration_df, empty_column], axis=1)
        
        lm_duration_df.loc[i,col] = count


        if col not in lm_start_df.columns:
            empty_column = pd.Series([''] * len(lm_start_df), name=col)
            # Add the empty column to the DataFrame
            lm_start_df = pd.concat([lm_start_df, empty_column], axis=1)

        lm_start_df.loc[i,col] = first_occurence



    


lm_count_ot_df = lm_count_ot_df.sort_values(by=lm_count_ot_df.columns[149])


seven = set([1,2,8,9,11,14,17,19,21,22,25])
eight = set([3,4,5,6,7,13,18])
nine = set([0,10,12,15,16,20,23,24])

'''for set in [seven,eight,nine]:
        set = list(set)
        heatmap(lm_count_ot_df.loc[set].astype(float),
                x_label = 'Timestep',
                y_label = 'Participant',
                fig_title = f'Number of LM over time',
                savepath = savepath +  f'{set[0]}_num_landmark_heatplot',
                cmap='viridis',
                cbar=True,
                show = True)'''
    



heatmap(lm_count_ot_df.astype(float),
                x_label = 'Timestep',
                y_label = 'Participant (sorted my No. of LM)',
                fig_title = f'Number of LM over time',
                savepath = savepath +  f'num_landmark_heatplot_global',
                cmap='viridis',
                cbar=True,
                show = False)

smooth_line_plot(lm_count_ot_df.astype(float),
                       x_label= 'Time Steps',
                       y_label= 'Num of Landmarks',
                       fig_title = 'Diameter of all Participants over time with Mean',
                       savepath = savepath + f'landmarks_ot_global_smooth',
                       color='blue',
                       show = False)
















end_lm = set()
for sublist in end_lm_per_person:
    for element in sublist:
        end_lm.add(element)

num_indiv_lm = len(end_lm) - num_glob_lm

temp_lm = set()
for sublist in landmark_per_person:
    for element in sublist:
        temp_lm.add(element)

num_temp_lm = len(temp_lm) - len(end_lm)

print(num_nodes, num_glob_lm, num_indiv_lm, num_temp_lm)


# Example data
labels = ['Global LM', 'Indiv. LM', 'Temp. LM', 'Never a LM']
sizes = [num_glob_lm, num_indiv_lm, num_temp_lm, num_nodes-num_temp_lm-num_glob_lm-num_indiv_lm]

# Create a pie chart with numbers
plt.pie(sizes, labels=labels,  autopct=lambda p: f'{p:.1f}%\n({p * sum(sizes) / 100:.0f})', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
plt.title('Buildings in Westbrook - graph-definedn-Landmarks')

# Display the pie chart
plt.show()

end_lm = list(end_lm)
temp_lm = list(temp_lm)




global_landmarks
indiv_landmarks = [x for x in end_lm if x not in global_landmarks]
temp_landmarks = [x for x in temp_lm if x not in end_lm]



 #file in os.chdir folder
node_data = pd.read_csv(pos_path + "node_positions.csv")

#node_data = dict(zip(pos_data['hitObjectColliderName'], zip(pos_data['pos_x'], pos_data['pos_z'])))


node_data['LM_category'] = 0

node_data.loc[node_data['hitObjectColliderName'].isin(global_landmarks), 'LM_category'] = 3
node_data.loc[node_data['hitObjectColliderName'].isin(indiv_landmarks), 'LM_category'] = 2
node_data.loc[node_data['hitObjectColliderName'].isin(temp_landmarks), 'LM_category'] = 1


#    print(pos_data['pos_x'].max()-pos_data['pos_x'].min())
#    print(pos_data['pos_z'].max()-pos_data['pos_z'].min())

color_mapping = {0: 'orange', 1: 'red', 2: 'green', 3: 'blue'}  # Add more categories as needed

# Create a scatter plot
plt.scatter(node_data['pos_x'], node_data['pos_z'], c=node_data['LM_category'].map(color_mapping),  s=50, alpha=0.7)

# Add labels and title
plt.xlim(node_data['pos_x'].min()-10, node_data['pos_x'].max()+10)
plt.ylim(node_data['pos_z'].min()-10, node_data['pos_z'].max()+10)
plt.xlabel('X Position')
plt.ylabel('Z Position')

plt.title('Map-Plot of Landmarks')

# Add colorbar
#cbar = plt.colorbar()
#cbar.set_label('Content')

# Show the plot
plt.show()





