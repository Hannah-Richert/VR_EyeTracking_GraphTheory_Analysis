import os
import pandas as pd
import networkx as nx
import matgrab
import matplotlib.pyplot as plt
import graph_measure_functions as myGMF

"""
The scripts generates a gaze-graph from participant gazes (of the 150min) and takes some global graph measures on it, which are printed.

Functions of used graph measures are stored in graph_measure_unctions.py and importet as myGMF.
Conversion of mat file is done by the matgrab-library.

Parameters:
- data_dir (str): Directory with the X_gaze_data_WB.mat files (X = participant id)
- part_list (list<int>): IDs of the participants 

Results:
None

"""

# Adjustable variables
data_dir = 'D:/WestbrueckData/Pre-processing/gazes_vs_noise/'
os.chdir(data_dir)

#savepath= 'E:/Westbrueck Data/SpaRe_Data/1_Exploration/Analysis/tempDevelopment/1minSections/'
#os.chdir('E:/Westbrueck Data/SpaRe_Data/1_Exploration/Pre-processsing_pipeline/gazes_vs_noise/')

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]


# Main participant loop
for part in part_list:

    # file in os.chdir folder
    file_path = f"{part}_gazes_data_WB.mat"

    print('Participant:',str(part))

    # Load data from .mat file and convert to df
    data_df = matgrab.mat2df(file_path)

    # Extract relevant fields for graph generation from the csv
    gazes_data_df = pd.DataFrame({'colliderName': data_df['hitObjectColliderName']})

    # Remove rows with 'NH'(no house) in the 'hitObjectColliderName' column
    gaze_data = gazes_data_df[gazes_data_df['colliderName'] != 'NH']

    # list of unique colliders
    node_table = pd.DataFrame({'building': gaze_data['colliderName'].unique()})

    # list of edges (can contain identical edges)
    edge_table = pd.DataFrame({'orig': gaze_data['colliderName']})
    edge_table['dest'] = edge_table['orig'].shift(1)
    # drop edges containing na values (result from the shift)
    edge_table.dropna(inplace=True)

    # drop self-looping edges
    edge_table = edge_table[edge_table['orig'] != edge_table['dest']]

    # drop duplicated edges (both directions a-b == b-a) 
    # nx-undirected-graph does this implicitly automatically
    # edge_table = edge_table[~edge_table[['orig', 'dest']].apply(lambda row: frozenset(row), axis=1).duplicated(keep='first')]    

    # Create a undirected empty graph
    graph = nx.Graph()

    # adding nodes and edges
    graph.add_nodes_from(node_table['building'])
    graph.add_edges_from(edge_table.values)

    # Remove nodes 'noData' and 'newSession' node, if they exist
    # all corresponding edges will be removed as well - graph is cut
    nodes_to_remove = ['noData', 'newSession']
    nodes_to_remove_existing = [node for node in nodes_to_remove if node in graph.nodes]
    graph.remove_nodes_from(nodes_to_remove_existing)

    #print graph
    #nx.draw(graph)
    #plt.show()
    
    # Calculate the graph measures from myGMF

    num_nodes = myGMF.num_of_nodes(graph)
    num_edges = myGMF.num_of_edges(graph)
    
    density = myGMF.density(graph)

    diameter = myGMF.diameter(graph)
    avg_short_path = myGMF.average_shortest_path_length(graph)

    hierarchy_idx = myGMF.hierarchy_index(graph)

    print('Measurements:','NumNodes:',num_nodes,'NumEdges:',num_edges,'Density:',density,'Diameter:',diameter,'Avg. Shortest Path:',avg_short_path,'Hierarchy Index:',hierarchy_idx)

print('Finished')

