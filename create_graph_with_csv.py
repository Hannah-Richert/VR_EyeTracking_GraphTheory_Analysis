import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Adjustable variables

savepath = 'D:/WestbrueckData/Analysis/'
#save_graph_path = 'D:/WestbrueckData/Analysis/Graphs/Temp'
os.chdir('D:/WestbrueckData/Pre-processing/')

#savepath= 'E:/Westbrueck Data/SpaRe_Data/1_Exploration/Analysis/tempDevelopment/1minSections/'
#os.chdir('E:/Westbrueck Data/SpaRe_Data/1_Exploration/Pre-processsing_pipeline/gazes_vs_noise/')


# 26 participants with 5x30min VR training less than 30% data loss
#part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]

part_list = [1004]



for part in part_list:

    #file_path = file in os.chdir folder
    file_path = f"{part}_gazes_data_WB.csv"

    print('Participant:',str(part))
    # Load data from .csv file
    data_df = pd.read_csv(file_path)

    # Extract relevant fields for graph generation from the csv
    gazes_data_df = pd.DataFrame({'ColliderName': data_df['hitObjectColliderName']})
    #print(gazes_data_df)
    # Remove rows with 'NH'(no house) in the 'hitObjectColliderName' column
    gaze_data = gazes_data_df[gazes_data_df['ColliderName'] != 'NH']

    # list of unique colliders
    node_table = pd.DataFrame({'Name': gaze_data['ColliderName'].unique()})

    # all edges (can contain identical edges)
    edge_table = pd.DataFrame({'origin': gaze_data['ColliderName']})
    edge_table['destination'] = edge_table['origin'].shift(1)
    
    # drop edges containing na values (result from the shift)
    edge_table.dropna(inplace=True)

    # drop self-looping edges
    edge_table = edge_table[edge_table['origin'] != edge_table['destination']]

    # Remove all repetitions 
    # not necessary as nx.graph will only select different edges 
    # edge_table = edge_table.drop_duplicates()


    # Create a undirected empty graph
    graph = nx.Graph()

    # adding nodes and edges
    graph.add_nodes_from(node_table['Name'])
    graph.add_edges_from(edge_table.values)

    # Remove nodes 'noData' and 'newSession' node, if they exist
    # add corresponding edges will be removed as well
    nodes_to_remove = ['noData', 'newSession']
    nodes_to_remove_existing = [node for node in nodes_to_remove if node in graph.nodes]
    graph.remove_nodes_from(nodes_to_remove_existing)

    #print graph
    #nx.draw(graph)
    #plt.show()
    
    # Calculate and save some graph measures
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    
    max_edges = (num_nodes * (num_nodes - 1)) / 2  if num_nodes > 1 else 0
    density = num_edges / max_edges if max_edges > 0 else 0
    
    # Calculate diameter (if the graph is connected)
    diameter = nx.diameter(graph) if nx.is_connected(graph) else 0

    print('Measurements:','NumNodes:',num_nodes,'NumEdges:',num_edges,'Density:',density,'Diameter:',diameter)


print('Finished')





