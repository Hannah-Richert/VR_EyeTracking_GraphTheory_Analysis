"""
The script selects certain columns from gaze-data and divides the samples into time segments.
An added column indicates the time segment of each gaze/sample.

Parameters:
- data_dir (str): Directory with the X_gaze_data_WB.csv files (X = participant id)
- save_dir (str): Directory for the result files
- part_list (list<int>): IDs of the participants 
- segment_time (int): Duration of a time segment in seconds

Saved Results:
- X_segmented_gaze_data_WB.csv: Gaze data with columns: 'hitObjectColliderName', 'segment', 'firstTimeStamp'
"""

import os
import ast
import numpy as np
import pandas as pd

    
def segmentation(segment_duration, part):
    """
    Add time-segment indices to the gaze data for a participant.

    Parameters:
    - segment_duration (int): Duration of a segment in seconds.
    - part (int): Participant ID.

    Returns:
    - gaze_data_df (pd.DataFrame): Gaze data DataFrame with additional 'segment' and 'firstTimeStamp' columns.
    """
     
    # Load data from .csv file
    file = f'{part}_gazes_data_WB.csv'
    data_df = pd.read_csv(file)

    # Extract relevant fields from the CSV
    gazes_data_df = data_df[['hitObjectColliderName', 'timeStampDataPointStart', 'clusterDuration', 'Session']].copy()

    # Add column for the segment indices
    gazes_data_df['segment'] = np.nan

    # Convert timestamps from string to arrays
    gazes_data_df['timeStampDataPointStart'] = gazes_data_df['timeStampDataPointStart'].replace({' ': ','}, regex=True).apply(ast.literal_eval).apply(np.array)
    gazes_data_df['firstTimeStamp'] = gazes_data_df['timeStampDataPointStart'].str[0]

    # Get the timestamp of the first sample
    first_ts_in_session = gazes_data_df['firstTimeStamp'].iloc[0]

    # Assign the segment to the first sample
    segment_index = 0
    gazes_data_df.at[0, 'segment'] = segment_index

    # Variables to deal with time jumps between sessions
    sum_previous_sessions = 0

    # iterate through all gaze-samples, besides the first
    for gaze_index in range(1, len(gazes_data_df)): 

        # If start of a new 10min session  = end of a 10min session 
        if gazes_data_df['hitObjectColliderName'].iloc[gaze_index] == 'newSession':

            # If not the end of last (15th) session
            if gaze_index < len(gazes_data_df) - 1:

                # time between the first and last timestap in the session
                last_session_sum = gazes_data_df['firstTimeStamp'].iloc[gaze_index-1] - first_ts_in_session
                sum_previous_sessions += last_session_sum

                # update the timestamp indicating the beginn of a session
                first_ts_in_session = gazes_data_df['firstTimeStamp'].iloc[gaze_index + 1]

        # if not the start of a new 10min session 
        else:
            # time between the current gaze-sample and the start of the session
            time_in_session = gazes_data_df['firstTimeStamp'].iloc[gaze_index] - first_ts_in_session
            
            # total time from 1st session until current gaze-sample
            current_sum = sum_previous_sessions + time_in_session

            # if passed time is longer than the segment-time + all previous segments-time, increase the segment index
            if current_sum > (segment_duration * (segment_index + 1)):
                segment_index += 1

        # assign the segment index to the gaze sample
        gazes_data_df.loc[gaze_index, 'segment'] = segment_index
    

    return gazes_data_df
    

# Start of the main script
if __name__ == "__main__": # only execute if file is called directly not as a module

    # Adjustable variables
    data_dir = 'D:/WestbrueckData/Pre-processing/'  # Path for input file
    save_dir = 'D:/WestbrueckData/Pre-processing/'  # Path for the output file
    os.chdir(data_dir)  # Change directory to the input file directory

    # Participant list
    part_list = [
        1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021,
        1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072,
        1073, 1074, 1075, 1077, 1079, 1080
    ]
    segment_time = 60  # Duration of a segment in seconds

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Process each participant
    for part in part_list:

        print(f"Processing participant {part}")

        segmented_df = segmentation(segment_time, part)

        # Keep only relevant columns 
        segmented_df = segmented_df[['hitObjectColliderName','segment','firstTimeStamp']]
        
        # Save the data
        output_path = os.path.join(save_dir, f'{part}_segmented_gaze_data_WB.csv')
        segmented_df.to_csv(output_path)

    print("Finished")