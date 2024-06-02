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

import os
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import graph_measure_functions as myGMF

#warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize')


# Start of the main script
if __name__ == "__main__": # savety measure to only be executet if the file is called directly and not as a module



    ##### 0. Adjustable variables

    data_dir = 'D:/WestbrueckData/Pre-processing/' # dir of the input gaze-files
    save_dir = 'D:/WestbrueckData/Analysis/' # dir for saving graph measures
    os.chdir(data_dir) # dir for input data

    # 26 participants with 5x30min VR training less than 30% data loss
    part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 
                1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 
                1073, 1074, 1075, 1077, 1079, 1080
                ]

    num_time_segments = 150 # each timestep 1 minute



    ##### 1. Initialize data structures to store results with data from all participants

    # gaze properties, extracted from raw tables (where duplicates are allowed)
    overview_session = []
    overview_node_hits = []
    overview_diff_nodes = []
    overview_diff_edges = []
    overview_edge_hits = []
    overview_new_nodes = []
    overview_new_edges = []
    # graph properties/measures
    overview_num_nodes = []
    overview_num_edges = []
    overview_density = []
    overview_diameter = []
    overview_avg_short_path = []
    overview_hirarchy_index = []



    ##### 2. iterate through participants, generate graph and measures

    for part in part_list:

        ##### 2.1 Load data and make structures for saving measures

        print(f'Processing participant {part}')

        # Load data from csv
        gazes_data_df = pd.read_csv( f'{part}_segmented_gaze_data_WB.csv')

        # Data structures to save data on graph-measurements 
        graph_measurements = pd.DataFrame(columns=['Session',
                                                'NumNodehitsInLastSeg',
                                                'NumNodesInLastSeg',
                                                'NumNewNodesInLastSeg',
                                                'NumEdgehitsInLastSeg',
                                                'NumEdgesInLastSeg',
                                                'NumNewEdgesInLastSeg',
                                                'NumNodes', 
                                                'NumEdges', 
                                                'Density',
                                                'Diameter',
                                                'AvgShortPath',
                                                'HierarchyIndex'
                                                ])
        
        # Datat structure to save node degrees
        all_nodes_list = gazes_data_df['hitObjectColliderName'].unique().tolist()
        node_df = pd.DataFrame(index = np.arange(150),columns=all_nodes_list)
        node_df.drop(columns=['noData', 'newSession'], inplace=True)
        node_df['Measure'] = ''
        
        node_degree_df = node_df.copy()



        ##### 2.2 iterating through the time segments, generate graph and meausre on the subset of the gazes

        for i in range(num_time_segments):

            ##### 2.2.1 Preperations: node and edge table for the graph and some gaze-quatities for Discovery and Exploration Rate

            # Each detected index is signaling the last entry for a sub-graph    
            current_data = gazes_data_df[gazes_data_df['segment'] <= i]
            
            # Remove rows with 'NH'(no house) in the 'hitObjectColliderName' column
            gaze_data = current_data[current_data['hitObjectColliderName'] != 'NH']


            # Make node table and select nodes/gazes from the last segment with certain properties(all nodes,unique nodes,new nodes) 
            node_table = gaze_data.loc[:, ['hitObjectColliderName','segment']].copy()
            node_table.rename(columns={'hitObjectColliderName': 'name'}, inplace=True)

            # Measures: all node hits and unique node hits dfs
            all_node_hits = node_table[node_table['segment'] == i]
            unique_nodes = node_table[node_table['segment'] == i].drop_duplicates()

            # FINAL node table
            node_table = node_table.drop_duplicates(subset=['name'], keep='first')

            # For Discovery and Exploration Rate: new nodes in that segment  
            new_nodes = node_table[node_table['segment'] == i].drop_duplicates()


            # Make edge table and select gaze transitions 
            # Also gather properties from the last segment (number of all transitions,unique transitions,new transitions) 

            edge_table = gaze_data.loc[:, ['hitObjectColliderName','segment']].copy()
            edge_table['orig'] = edge_table['hitObjectColliderName'].shift(1)

            edge_table.rename(columns={'hitObjectColliderName': 'dest'}, inplace=True)
            edge_table.dropna(inplace=True)
            # drop self references
            edge_table = edge_table[edge_table['orig'] != edge_table['dest']]

            # For Discovery and Exploration Rate: all transition hits and unique transitions   
            all_edge_hits = edge_table[edge_table['segment'] == i]
            unique_edges = edge_table[edge_table['segment'] == i][['orig', 'dest']].drop_duplicates()

            # drop duplicates (both directions a-b = b-a) - nx-graph does this automatically, but we do it anywas for gaze-transition measures
            # FINAL edge table
            edge_table = edge_table[~edge_table[['orig', 'dest']].apply(lambda row: frozenset(row), axis=1).duplicated(keep='first')]

            # For Discovery and Exploration Rate:: amount of new edges as a df  
            new_edges = edge_table[edge_table['segment'] == i][['orig', 'dest']].drop_duplicates()

            # drop certain hit objects from the dfs for the Exploration & Discovery Rate measure
            for node in ['noData', 'newSession']:
                unique_nodes = unique_nodes[unique_nodes.ne(node).all(axis=1)]
                new_nodes = new_nodes[new_nodes.ne(node).all(axis=1)]
                all_node_hits = all_node_hits[all_node_hits.ne(node).all(axis=1)]

                unique_edges = unique_edges[unique_edges.ne(node).all(axis=1)]
                new_edges = new_edges[new_edges.ne(node).all(axis=1)]
                all_edge_hits = all_edge_hits[all_edge_hits.ne(node).all(axis=1)]



            ##### 2.2.2 Create a undirected, unweighted graph

            graph = nx.Graph()

            graph.add_nodes_from(node_table['name'])

            graph.add_edges_from(edge_table[['orig', 'dest']].values)

            # Remove nodes with attached edges; nodes: 'noData' & 'newSession'
            nodes_to_remove = ['noData', 'newSession']
            nodes_to_remove_existing = [node for node in nodes_to_remove if node in graph.nodes]
            graph.remove_nodes_from(nodes_to_remove_existing)

            #save graph of the first paticipants at certain instances for later plotting of example graph snapchots
            if part == 1004 and i & 10 == 0:
                print(i)
                nx.write_graphml(graph, f'{save_dir}{part}_{i}_step_graph.graphml')



            ##### 2.2.3 Calculate the graph measures from myGMF & node degrees

            num_nodes = myGMF.num_of_nodes(graph)
            num_edges = myGMF.num_of_edges(graph)
            
            density = myGMF.density(graph)

            diameter = myGMF.diameter(graph)
            avg_short_path = myGMF.average_shortest_path_length(graph)
            try:
                hierarchy_idx = myGMF.hierarchy_index(graph)
            except Exception as e:
                print(f'An error occurred: {e}')
                hierarchy_idx = 0


            # variale indicating which of the 5x30min experiment session the data belongs to
            current_session = i//30 + 1

            # gathering node degree data at timestep i
            node_degree = nx.degree(graph)
            for key, value in node_degree:
                if key in node_degree_df.columns:
                    node_degree_df.loc[i, key] = value

            node_degree_df.loc[i, 'Measure'] = 'NodeDegree'



            ##### 2.2.4 Collect and save the global graph and gaze properties for the current time step i for the current participant

            row_data = [current_session, 
                        len(all_node_hits),
                        len(unique_nodes),
                        len(new_nodes),
                        len(all_edge_hits),
                        len(unique_edges),
                        len(new_edges),
                        num_nodes, 
                        num_edges, 
                        density, 
                        diameter, 
                        avg_short_path,
                        hierarchy_idx
                        ]
            
            graph_measurements.loc[len(graph_measurements)] = row_data

        
        
        ##### 2.3 Save local graph measures of a participant and add global graph measures to overview lists (contain data from al participants)
        
        # save node degree data
        node_measurements_df = node_degree_df
        node_measurements_df.to_csv(save_dir + f'{part}_node_measurements.csv', index=True)

    
        # Append the measures for this participant to the overview data of all participants
        overview_session.append(graph_measurements['Session'].tolist())

        overview_node_hits.append(graph_measurements['NumNodehitsInLastSeg'].tolist())
        overview_diff_nodes.append(graph_measurements['NumNodesInLastSeg'].tolist())
        overview_edge_hits.append(graph_measurements['NumEdgehitsInLastSeg'].tolist())
        overview_diff_edges.append(graph_measurements['NumEdgesInLastSeg'].tolist())
        overview_new_nodes.append(graph_measurements['NumNewNodesInLastSeg'].tolist())
        overview_new_edges.append(graph_measurements['NumNewEdgesInLastSeg'].tolist())

        overview_num_nodes.append(graph_measurements['NumNodes'].tolist())
        overview_num_edges.append(graph_measurements['NumEdges'].tolist())
        overview_density.append(graph_measurements['Density'].tolist())
        overview_diameter.append(graph_measurements['Diameter'].tolist())
        overview_avg_short_path.append(graph_measurements['AvgShortPath'].tolist())
        overview_hirarchy_index.append(graph_measurements['HierarchyIndex'].tolist())



    ##### 3. Combline all lists with graph measures into one df and save it.

    # Save the overview data to separate CSV files
    overview_df = pd.DataFrame({
        'Participant': part_list,
        'Session': overview_session,
        'NumNodehitsInLastSeg': overview_node_hits,
        'NumNodesInLastSeg': overview_diff_nodes,
        'NumNewNodesInLastSeg': overview_new_nodes,
        'NumEdgehitsInLastSeg': overview_edge_hits,
        'NumEdgesInLastSeg': overview_diff_edges,
        'NumNewEdgesInLastSeg': overview_new_edges,
        'NumNodes': overview_num_nodes,
        'NumEdges': overview_num_edges,
        'Density': overview_density,
        'Diameter': overview_diameter,
        'HierarchyIndex': overview_hirarchy_index,
        'AvgShortestPath': overview_avg_short_path
        })

    overview_df.to_csv(save_dir + 'overview_graph_measures_final.csv', index=False)

    print('Finished')