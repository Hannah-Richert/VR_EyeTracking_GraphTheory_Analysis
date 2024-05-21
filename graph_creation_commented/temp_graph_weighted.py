import os
import ast
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import graph_measure_functions as myGMF

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")

"""
The scripts generates a gaze-graphs from participant gazes at regular intervall steps 
and takes some global graph measures on it, which are saved.

Functions of used graph measures are stored in graph_measure_unctions.py and importet as myGMF.

Parameters:
- data_dir (str): Directory with the X_gaze_data_WB.mat files (X = participant id)
- save_dir (str): Directory for the result files
- part_list (list<int>): IDs of the participants 
- num_time_segments (int): Number of time segments to interate through, depends on segmentation intervall

Saved Results:
- overview_graph_measures_new.csv (csv-file): containing all saved graph measures of all timesteps and participants
- X_node_measurements.csv: node degrees of participants (X = ID) at each timestep

"""

# Adjustable variables

# dir of the input gaze-files
data_dir = 'D:/WestbrueckData/Pre-processing/'

# dir for saving graph measures
save_dir = 'D:/WestbrueckData/Analysis/'

os.chdir(data_dir) # dir for input data

# 26 participants with 5x30min VR training less than 30% data loss
part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]

#part_list = [1005]
num_time_segments = 150



# main part starts




# Initialize data structures to store results with data from all participants
overview_num_nodes = []
overview_num_edges = []
overview_density = []
overview_diameter = []
overview_avg_short_path = []
overview_hirarchy_index = []
overview_session = []

for part in part_list:

    print(f"Processing participant {part}")

    # Load data from csv
    gazes_data_df = pd.read_csv( f'{part}_segmented_gaze_data_WB.csv')


    # Data structures to save data on graph-measurements 
    graph_measurements = pd.DataFrame(columns=['TimeStep',
                                      'NumNodes', 
                                      'NumEdges', 
                                      'Density',
                                      'Diameter',
                                      'HierarchyIndex',
                                      'Session',
                                      'AvgShortPath'])
    
    # Datat structure to save node degrees
    all_nodes_list = gazes_data_df['hitObjectColliderName'].unique().tolist()
    node_df = pd.DataFrame(index = np.arange(150),columns=all_nodes_list)
    node_df.drop(columns=['noData', 'newSession'], inplace=True)
    node_df['Measure'] = ''
    
    node_degree_df = node_df.copy()
    #between_centality_results_df = node_df.copy()



    # iterating through the time segments, taking snapchots of the graph
    for i in range(num_time_segments):

        # Each detected index is signaling the last entry for a sub-graph    
        current_data = gazes_data_df[gazes_data_df['segment'] <= i]
        
        
        # Remove rows with 'NH'(no house) in the 'hitObjectColliderName' column
        gaze_data = current_data[current_data['hitObjectColliderName'] != 'NH']

        

        # Create node end edge table
        node_table = pd.DataFrame({'building': gaze_data['hitObjectColliderName'].unique()})

        edge_table = pd.DataFrame({'dest': gaze_data['hitObjectColliderName']})
        edge_table['orig'] = edge_table['dest'].shift(1)
        edge_table.dropna(inplace=True) # because of the shift

        # drop self references
        edge_table = edge_table[edge_table['orig'] != edge_table['dest']]

        # drop duplicates (both directions a-b = b-a) - nx-graph does this automatically
        edge_table_unique = edge_table[~edge_table[['orig', 'dest']].apply(lambda row: frozenset(row), axis=1).duplicated(keep='first')]



        # Create a undirected graph
        graph = nx.Graph()

        graph.add_nodes_from(node_table['building'])

        #graph.add_edges_from(edge_table[['orig', 'dest']].values)


        # Add edges to the graph with custom weights

        for _,edge in edge_table_unique.iterrows():
            
            if edge in graph.edges:
                print(edge)

            weight = myGMF.custom_weight((edge['orig'], edge['dest']), edge_table)

            graph.add_edge(edge['orig'], edge['dest'], weight=weight)








        # Remove nodes with attached edges; nodes: 'noData' & 'newSession'
        nodes_to_remove = ['noData', 'newSession']
        nodes_to_remove_existing = [node for node in nodes_to_remove if node in graph.nodes]
        graph.remove_nodes_from(nodes_to_remove_existing)

        #save graph
        #nx.write_graphml(graph, f'{save_graph_path}/{current_part}/{i}_step_graph.graphml')

  
        # Calculate the graph measures from myGMF

        num_nodes = myGMF.num_of_nodes(graph)
        num_edges = myGMF.num_of_edges(graph)
        
        density = myGMF.density(graph)

        diameter = myGMF.diameter_weighted(graph)
        avg_short_path = myGMF.average_shortest_path_length_weighted(graph)
        try:
            hierarchy_idx = myGMF.hierarchy_index(graph)
        except Exception as e:
            print(f"An error occurred: {e}")
            hierarchy_idx = 0


        # variale indicating which of the 5x30min experiment session the data belongs
        current_session = i//30 + 1

        row_data = [i, 
                    num_nodes, 
                    num_edges, 
                    density, 
                    diameter, 
                    hierarchy_idx, 
                    current_session,
                    avg_short_path
                    ]
        
        graph_measurements.loc[len(graph_measurements)] = row_data

        # add node degrees to df
        node_degree = nx.degree(graph)
        for key, value in node_degree:
            if key in node_degree_df.columns:
                node_degree_df.loc[i, key] = value

        node_degree_df.loc[i, 'Measure'] = 'NodeDegree'
        
        '''between_centality = nx.betweenness_centrality(graph)
        for key, value in between_centality.items():
            if key in between_centality_results_df.columns:
                between_centality_results_df.loc[i, key] = value

        between_centality_results_df['Measure'] = 'BetweenCentr'

        all_measurements_df = pd.concat([node_degree_df,
                                         between_centality_results_df])'''
        
        print(row_data) #############################
        
    node_measurements_df = node_degree_df

    
    #node_measurements_df.to_csv(save_dir + f'{part}_node_measurements.csv', index=True)

        
    # Append the measures for this participant to the overview data
    overview_num_nodes.append(graph_measurements['NumNodes'].tolist())
    overview_num_edges.append(graph_measurements['NumEdges'].tolist())
    overview_density.append(graph_measurements['Density'].tolist())
    overview_diameter.append(graph_measurements['Diameter'].tolist())
    overview_hirarchy_index.append(graph_measurements['HierarchyIndex'].tolist())
    overview_session.append(graph_measurements['Session'].tolist())
    overview_avg_short_path.append(graph_measurements['AvgShortPath'].tolist())

    print(row_data) #########################################s

# Save the overview data to separate CSV files
overview_df = pd.DataFrame({
    'Participant': part_list,
    'Session': overview_session,
    'NumNodes': overview_num_nodes,
    'NumEdges': overview_num_edges,
    'Density': overview_density,
    'Diameter': overview_diameter,
    'HierarchyIndex': overview_hirarchy_index,
    'AvgShortestPath': overview_avg_short_path
    })


overview_df.to_csv(save_dir + 'overview_graph_measures_weighted.csv', index=False)

print('Finished')