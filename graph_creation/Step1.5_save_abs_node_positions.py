"""
This script generates and saves the (x,y,z) positions of the center of all buildings/nodes.

Parameters:
- data_dir (str): Directory with the X_gaze_data_WB.csv files (X = participant id)
- save_dir (str): Directory for the result files
- part_list (list<int>): IDs of the participants

Saved Results:
- node_positions.csv (csv-file): containing 3D positions of all buildings looked at by any participant
"""

import os
import pandas as pd

# Start of the main script
if __name__ == "__main__": # only execute if file is called directly not as a module


    # Directory of the input gaze-files
    data_dir = 'D:/WestbrueckData/Pre-processing/'

    # Directory for saving graph measures
    save_dir = 'D:/WestbrueckData/Pre-processing/'

    os.chdir(data_dir)  # Set directory for input data

    # 26 participants with 5x30min VR training less than 30% data loss
    part_list = [
        1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021,
        1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072,
        1073, 1074, 1075, 1077, 1079, 1080
    ]

    # Make a DataFrame to store node positions
    global_pos_data_df = pd.DataFrame(columns=[
        'hitObjectColliderName', 'hitObjectColliderBoundsCenter_x',
            'hitObjectColliderBoundsCenter_y', 'hitObjectColliderBoundsCenter_z'
    ])


    for part in part_list:
        print(f"Processing participant {part}")

        # Load data from .csv file
        data_df = pd.read_csv(f"{part}_gazes_data_WB.csv")

        # Extract hit objects with their positions
        pos_data_df = data_df[[
            'hitObjectColliderName', 'hitObjectColliderBoundsCenter_x',
            'hitObjectColliderBoundsCenter_y', 'hitObjectColliderBoundsCenter_z'
        ]]

        # Remove duplicates
        pos_data_df = pos_data_df.drop_duplicates(subset=['hitObjectColliderName'])

        # Concatenate DataFrame from participant with global DataFrame
        global_pos_data_df = pd.concat([global_pos_data_df, pos_data_df],
                                    ignore_index=True)

        global_pos_data_df = global_pos_data_df.drop_duplicates(
            subset=['hitObjectColliderName']
        )

        global_pos_data_df.reset_index(drop=True, inplace=True)

    # Extract the first number/position in the list of positions
    global_pos_data_df['pos_x'] = global_pos_data_df['hitObjectColliderBoundsCenter_x'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
    global_pos_data_df['pos_y'] = global_pos_data_df['hitObjectColliderBoundsCenter_y'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
    global_pos_data_df['pos_z'] = global_pos_data_df['hitObjectColliderBoundsCenter_z'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False).astype(float)
    global_pos_data_df.drop(['hitObjectColliderBoundsCenter_x', 'hitObjectColliderBoundsCenter_y', 'hitObjectColliderBoundsCenter_z'], axis=1, inplace=True)

    # Remove rows with 'NH' (no house), 'noData', and 'newSession' in the 'hitObjectColliderName' column
    values_to_exclude = ['noData', 'newSession', 'NH']
    global_pos_data_df = global_pos_data_df[~global_pos_data_df['hitObjectColliderName'].isin(values_to_exclude)]

    # Save node positions
    global_pos_data_df.to_csv(f'{save_dir}node_positions.csv', index=False)

    print('Finished')