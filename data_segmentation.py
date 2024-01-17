import os
import ast
import numpy as np
import pandas as pd

def segmentation(segment_time, part):
    """
    Add time-segemnt indices to the gaze gaze data for a  participant.

    Parameters:
    - segment_time (int): duration of a segment; in seconds
    - part (int): Participant ID

    """

    # increasing index, indicating the time segment (starting with 0 for the first segement)
    segment_index = 0
    
    # Load data from .csv file
    file = f'{part}_gazes_data_WB.csv'
    data_df = pd.read_csv(file)

    # Extract relevant fields from the csv
    gazes_data_df = data_df[['hitObjectColliderName','timeStampDataPointStart','clusterDuration','Session']].copy()

    # add column for the segment indices
    gazes_data_df['segment'] = np.nan

    # make extra column, containing the very first Start-timestap from each gaze cluster    
    gazes_data_df['timeStampDataPointStart'] =  gazes_data_df['timeStampDataPointStart'].replace({' ':','}, regex=True).apply(ast.literal_eval).apply(np.array)
    gazes_data_df['firstTimeStamp'] =  gazes_data_df['timeStampDataPointStart'].str[0]

    # get the timestamp of the first sample
    first_ts_in_session = gazes_data_df['firstTimeStamp'].iloc[0]

    # assign the segment to the first sample
    gazes_data_df.at[0,'segment'] = segment_index

    # variables to deal with time jumps between sessions
    sum_previous_sessions = 0
    last_session_sum = 0


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
            if current_sum >= (segment_time + segment_time * segment_index):
                segment_index += 1

        # assign the segment index to the gaze sample
        gazes_data_df.loc[gaze_index,'segment'] = segment_index
    
    return gazes_data_df
    




# Adjustable variables
savepath = 'D:/WestbrueckData/Pre-processing/' # path for theoutput file
os.chdir('D:/WestbrueckData/Pre-processing/') # directory with the input file

# 26 participants with 5x30min VR training less than 30% data loss
#part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
part_list = [1004]

segment_time = 60  # in seconds

# Ensure the save directory exists
os.makedirs(savepath, exist_ok=True)  


for part in part_list:

    print(f"Processing participant {part}")

    segmented_df = segmentation(segment_time, part)

    # only keep relevant cols 
    segmented_df = segmented_df[['hitObjectColliderName','segment','firstTimeStamp']]
    
    # save the data
    output_path = os.path.join(savepath, f'{part}_segmented_gaze_data_WB.csv')
    segmented_df.to_csv(output_path)

print("Finished")