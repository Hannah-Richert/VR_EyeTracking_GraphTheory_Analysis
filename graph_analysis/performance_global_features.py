"""
This script analyzes the relationship between participants' mean performance in a VR city task 
and various graph measures extracted from their navigation data.

The script performs the following steps:
1. Sets adjustable variables, including file paths and participant list.
2. Loads and transforms data from CSV files.
3. Plots the distribution of participant's average performance.
4. Calculates the linear correlation between mean performance and each graph measure feature.
5. Performs linear regression modeling with different sets of graph measures as predictor variables.

Adjustable Variables:
- savepath (str): Path to save the plots.
- part_list (list): List of participant IDs.
- save_dpi (int): Resolution of saved figures.

"""

import os
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import pearsonr

warnings.simplefilter(action='ignore', category=DeprecationWarning)



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/performance/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 
             1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 
             1073, 1074, 1075, 1077, 1079, 1080
             ]
#part_list = [1004]

# resolution of any saved figures
save_dpi = 300 #600 



################################### 1. Load and transform data ###################################

dataPerformance = pd.read_csv("Performance/overviewPerformance.csv")

dataGraphMeasures =  pd.read_csv('parts_summary_stats.csv')

# Combine selected columns from dataPerformance and dataGraphMeasures
dataGraphM = pd.concat([dataPerformance.loc[:, ['Participants', 'meanPerformance']], 
                        dataGraphMeasures], axis=1)

dataGraphM = dataGraphM.astype(float)
dataGraphM.drop(['Participants','ParticipantID'],  axis=1, inplace=True)



################################### 2. Plotting MeanPerformance Distribution ###################################

# gloabal settings for all figures
mpl.rcParams.update({'font.size': 14,  # for normal text
                     'axes.labelsize': 16,  # for axis labels
                     'axes.titlesize': 16,  # for title
                     'xtick.labelsize': 14,  # for x-axis tick labels
                     'ytick.labelsize': 14})  # for y-axis tick labels

plt.figure(figsize=(4, 4))

# Create the box plot
sns.boxplot(data=dataGraphM['meanPerformance'], color='#103F71', boxprops=dict(alpha=0.6))
# Overlay the swarm plot
sns.swarmplot(data=dataGraphM['meanPerformance'], color='#103F71', size=8)

# Remove x-axis tick labels
plt.gca().set_xticklabels([])
# Add title and labels
plt.title("Distribution of Participant's Avg. Performance")
plt.xlabel('Participants')
plt.ylabel('Mean Task Performance Error [Angular degrees]')

# Show the plot
plt.tight_layout()
plt.savefig(savepath + 'performance_distri', dpi=save_dpi, bbox_inches='tight')

plt.show()



################################### 2. Linear Correlation of MeanPerformance with each Gaze-Graph Feature ###################################

for col1 in dataGraphM.columns:
        
        print(col1)

        data_2 = dataGraphM['meanPerformance']
        data_1 = dataGraphM[col1]

        corr, p = pearsonr(data_1, data_2)

        print(f'Feature: {col1} / Correlation:{corr}, {p}')
  


################################### 3. Linear Regression Model with all Graph Diameter/AvgShortPath features as model variables ###################################

# Model Variables: 'EndDiameter','MaxDiameter','MaxDiameterIndex','MeanDiameter', 'EndAvgShortPath','MaxAvgShortPath','MeanAvgShortPath'
# Predictable Variabe: meanPerformance

# all
model_single = sm.OLS.from_formula(f'meanPerformance ~ MeanDiameter + EndDiameter + MaxDiameter + MaxDiameterIndex + EndAvgShortPath + MaxAvgShortPath + MeanAvgShortPath', data=dataGraphM)
result_single = model_single.fit()
print('AllStruc',result_single.summary())

# Without End Diameter
model_single = sm.OLS.from_formula(f'meanPerformance ~ MeanDiameter + MaxDiameter + MaxDiameterIndex + EndAvgShortPath + MaxAvgShortPath + MeanAvgShortPath', data=dataGraphM)
result_single = model_single.fit()
print('StrucNotEndDiam',result_single.summary())


#Without Mean Diameter
model_single = sm.OLS.from_formula(f'meanPerformance ~ MaxDiameter + MaxDiameterIndex + EndAvgShortPath + MaxAvgShortPath + MeanAvgShortPath', data=dataGraphM)
result_single = model_single.fit()
print('StrucNotEndAndMeanDiam',result_single.summary())



################################### 4. Linear Model with all 'End' features as variables ###################################
        
# Model Variables: 'EndDiameter', 'EndAvgShortPath','EndNumNodes','EndNumEdges'
# Predictable Variabe: meanPerformance

# all
model_single = sm.OLS.from_formula(f'meanPerformance ~ EndAvgShortPath + EndDiameter + NumEndNodes + NumEndEdges', data=dataGraphM)
result_single = model_single.fit()
print('AllEnd',result_single.summary())

# Without End Diameter
model_single = sm.OLS.from_formula(f'meanPerformance ~ EndAvgShortPath + NumEndNodes + NumEndEdges', data=dataGraphM)
result_single = model_single.fit()
print('EndNotEndDiam',result_single.summary())



##################################

print('End')