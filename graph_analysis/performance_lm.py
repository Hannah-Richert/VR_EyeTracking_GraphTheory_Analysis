"""
This script analyzes the correlation between participants' performance in a VR training environment 
and the presence of landmarks (LMs) in their navigation data.

The script performs the following steps:
1. Sets adjustable variables, including file paths and participant list.
2. Loads and transforms data from CSV files.
3. Computes the Pearson correlation between the mean performance of participants and the LM status of various buildings.
4. Prints the correlation coefficients and p-values for each building, along with the count of participants having/not having each LM.

Adjustable Variables:
- savepath (str): Path to save any plots.
- part_list (list): List of participant IDs.
"""
import os
import pandas as pd
import warnings
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



################################### 1. Load and transform data ###################################

dataPerformance = pd.read_csv("Performance_data/overviewPerformance.csv")

dataGraphLM =  pd.read_csv('end_lm.csv')

# Combine selected columns from dataPerformance and dataGraphLM
            
dataGraphM = pd.concat([dataPerformance.loc[:, ['meanPerformance']], 
                        dataGraphLM], axis=1)

dataGraphM = dataGraphM.astype(float)



################################### 2. Linear Correlation of MeanPerformance with each Buidlings LMStatus distribution (how many people lm, how many notLM) ###################################

# iterate through each building and their lm/notLm distribution across participants
for col1 in dataGraphM.columns:
        
        print(col1)

        data_2 = dataGraphM['meanPerformance']
        data_1 = dataGraphM[col1]

        corr, p = pearsonr(data_1, data_2)
        print(f'Correlation Coefficient:{corr}, {p}')
        
        if col1 != 'meanPerformance':
                lm_popularity = data_1.value_counts()

                print('Num of Participants having this LM:',lm_popularity[1])
                print('Num of Participants not having this LM:',lm_popularity[0])
  
'''
Significant:
TaskBuilding_35 -> if present smaller mean performance error
Building_131 -> if present higher mean performance error

All other buidlings Not stat. significant.
'''



###################################

print('End')