import os
import pandas as pd


# Adjustable variables
savepath = 'D:/WestbrueckData/Pre-processing/'
os.chdir('D:/WestbrueckData/Pre-processing/')

# 26 participants with 5x30min VR training less than 30% data loss
#part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
part_list = [1004]

global_pos_data_df = pd.DataFrame(columns=['hitObjectColliderName','hitObjectColliderBoundsCenter_x','hitObjectColliderBoundsCenter_y','hitObjectColliderBoundsCenter_z'])

for part in part_list:
    
    print(f"Processing participant {part}")

    # Load data from .csv file
    data_df = pd.read_csv(f"{part}_gazes_data_WB.csv")

    pos_data_df = data_df[['hitObjectColliderName','hitObjectColliderBoundsCenter_x','hitObjectColliderBoundsCenter_y','hitObjectColliderBoundsCenter_z']]
   
    pos_data_df = pos_data_df.drop_duplicates(subset=['hitObjectColliderName'])

    # Concatenate df with global_df vertically
    global_pos_data_df = pd.concat([global_pos_data_df, pos_data_df], ignore_index=True)

    global_pos_data_df = global_pos_data_df.drop_duplicates(subset=['hitObjectColliderName'])

    global_pos_data_df.reset_index(drop=True, inplace=True)
    

global_pos_data_df['pos_x'] =  global_pos_data_df['hitObjectColliderBoundsCenter_x'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
global_pos_data_df['pos_y'] =  global_pos_data_df['hitObjectColliderBoundsCenter_y'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
global_pos_data_df['pos_z'] =  global_pos_data_df['hitObjectColliderBoundsCenter_z'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
global_pos_data_df.drop(['hitObjectColliderBoundsCenter_x','hitObjectColliderBoundsCenter_y','hitObjectColliderBoundsCenter_z'], axis=1, inplace=True)

# Remove rows with 'NH'(no house), noData and newSession in the 'hitObjectColliderName' column
values_to_exclude = ['noData', 'newSession', 'NH'] 
global_pos_data_df = global_pos_data_df[~global_pos_data_df['hitObjectColliderName'].isin(values_to_exclude)]

global_pos_data_df.to_csv(savepath + 'node_positions.csv', index=False)


print('Finished')

    