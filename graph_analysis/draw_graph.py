"""
This script loads graph data and node positions, and then draws the graphs on a background image representing a map. 
It saves the resulting images at specified time steps for a given participant.

Adjustable Variables:
- savepath (str): Directory to save generated plots.
- pos_path (str): Path to the node positions CSV file.
- part_id (int): Participant ID to process.
- save_dpi (int): Resolution of saved figures.

Outputs:
- Plots of graphs superimposed on the map background saved in the specified directory.

"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



################################### 0. Adjustable variables ###################################

savepath = 'D:/WestbrueckData/Analysis/Plots/'
pos_path = 'D:/WestbrueckData/Pre-processing/'
os.chdir('D:/WestbrueckData/Analysis/')

# 26 participants with 5x30min VR training less than 30% data loss
# List of possible ids [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
part_id = 1004 # particiapntID

# resolution of the saved figure 
save_dpi = 300 #600 


################################### 1. Load and transform data ###################################

# file in os.chdir folder
pos_data = pd.read_csv(pos_path + "node_positions.csv")
node_data = dict(zip(pos_data['hitObjectColliderName'], zip(pos_data['pos_x'], pos_data['pos_z'])))

wb_image = plt.imread('map_natural_500mMarker.png') 

################################### 2. Load Graph data and Draw graph on the loaded image ###################################


for i in [10,70,149]: # [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,149]:

    graph = nx.read_graphml(f'{part_id}_{i}_step_graph.graphml')

    fig,ax = plt.subplots(figsize=(8.8, 6.2))
    
    ax.imshow(wb_image, extent=[pos_data['pos_x'].min()-17,pos_data['pos_x'].min() + 969.988,
                         pos_data['pos_z'].min()-65, pos_data['pos_z'].min()+693.07], alpha=0.9)


    ax.set_xlim(pos_data['pos_x'].min()-17,pos_data['pos_x'].min() + 969.988)
    ax.set_ylim(pos_data['pos_z'].min()-65, pos_data['pos_z'].min()+693.07)

    nx.draw_networkx(graph, 
                     pos = node_data, 
                     with_labels = False, 
                     node_size = 9,
                     width = 0.9,
                     node_color = '#103F71',
                     edge_color = '#6555CB')


    #plt.title(f'Map-Plot of Gaze-Graph of participant {part} at t = {i+1}')
    plt.savefig(f'{savepath}graph_snapshot_{part_id}_{i}', dpi=save_dpi,  bbox_inches='tight')
    plt.show()

print('Finished')
