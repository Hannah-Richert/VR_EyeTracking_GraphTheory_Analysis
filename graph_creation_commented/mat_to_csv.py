import os
import matgrab # Install: https://pypi.org/project/matgrab/ Credits: https://github.com/Aaronearlerichardson/matgrab


def mat_to_csv(savepath, part):
    """
    Converts and saves data from .mat file to .csv file.

    Parameters:
    - savepath (str): Path to the directory where processed data will be saved.
    - part (int): Participant ID 

    Returns: None

    """
    # load input file
    input_file = f'{part}_gazes_data_WB.mat'
    
    output_path = os.path.join(savepath, f'{part}_gazes_data_WB.csv')

    # Use matgrab module to convert .mat file to DataFrame and save it as CSV
    matgrab.mat2df(input_file).to_csv(output_path)

        
    

# Adjustable variables
savepath = 'D:/WestbrueckData/Pre-processing/' # path for the .csv files
os.chdir('D:/WestbrueckData/Pre-processing/gazes_vs_noise/') # directory with the .mat files

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
#part_list = [1004]

# Ensure the save directory exists
os.makedirs(savepath, exist_ok=True)  


# Iterate through the list of participants
for part in part_list:
    
    print(f"Processing participant {part}")
    
    mat_to_csv(savepath, part)

print("Finished")













"""
    explicit transformation of all cols, with this one could only select a few cols instead of all
    matgrab.mat2df(file, 
                        ["sampleNr",
                        "clusterDuration",
                        "timeStampDataPointStart",
                        "timeStampDataPointEnd",
                        "timeStampGetVerboseData",
                        "hitObjectColliderName",
                        "hitPointOnObject_x",
                        "hitPointOnObject_y",
                        "hitPointOnObject_z",
                        "hitObjectColliderBoundsCenter_x",
                        "hitObjectColliderBoundsCenter_y",
                        "hitObjectColliderBoundsCenter_z",
                        "hitObjectColliderisGraffiti",
                        "hmdPosition_x",
                        "hmdPosition_y",
                        "hmdPosition_z",
                        "hmdDirectionForward_x",
                        "hmdDirectionForward_y",
                        "hmdDirectionForward_z",
                        "hmdRotation_x",
                        "hmdRotation_y",
                        "hmdRotation_z",
                        "playerBodyPosition_x",
                        "playerBodyPosition_y",
                        "playerBodyPosition_z",
                        "bodyTrackerPosition_x",
                        "bodyTrackerPosition_y",
                        "bodyTrackerPosition_z",
                        "bodyTrackerRotation_x",
                        "bodyTrackerRotation_y",
                        "bodyTrackerRotation_z",
                        "eyeOpennessLeft",
                        "eyeOpennessRight",
                        "pupilDiameterMillimetersLeft",
                        "pupilDiameterMillimetersRight",
                        "eyePositionCombinedWorld_x",
                        "eyePositionCombinedWorld_y",
                        "eyePositionCombinedWorld_z",
                        "eyeDirectionCombinedWorld_x",
                        "eyeDirectionCombinedWorld_y",
                        "eyeDirectionCombinedWorld_z",
                        "eyeDirectionCombinedLocal_x",
                        "eyeDirectionCombinedLocal_y",
                        "eyeDirectionCombinedLocal_z",
                        "testRow",
                        "Session",
                        "ETSession"
                        ]).to_csv(output_path)
     """