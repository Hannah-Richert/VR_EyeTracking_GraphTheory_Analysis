import os
import ast
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calculate_hierarchy_index(graph, plotting_wanted=False, saving_wanted=False, savepath='D:/WestbrueckData/Analysis/'):
    # Calculate Node Degree
    node_degree = dict(graph.degree())

    #max_degree = max(node_degree.values())
    # Finde alle Knoten mit dem maximalen Grad
    #print([node for node, degree in node_degree.items() if degree == max_degree])

    # Delete 0 degrees (non-connected nodes)
    node_degree = {k: v for k, v in node_degree.items() if v != 0}
    
    # Get unique degrees and sort
    
    #unique_degree = sorted(set(node_degree.values()))
    
    # Calculate Median of Node Degree
    median_degree = np.median(list(node_degree.values()))
    # Filter degrees above the median
    node_degree_upper = {k: v for k, v in node_degree.items() if v > median_degree}
    # Get unique degrees above the median and sort
    unique_degree_upper = sorted(set(node_degree_upper.values()))
    # Sortiere die Knoten nach ihrem Grad in absteigender Reihenfolge
    sorted_nodes = sorted(node_degree_upper.items(), key=lambda x: x[1], reverse=True)
    # Berechne das 10%-Quantil basierend auf der Anzahl der Knoten
    num_nodes = len(sorted_nodes)
    top_percentile_threshold = int((5 / 100) * num_nodes)
    # Wähle die Knoten im oberen 10%-Bereich aus
    top_percentile_nodes = sorted_nodes[:top_percentile_threshold]
    #print(top_percentile_nodes)

    # Calculate Degree Frequency for degrees above the median
    degree_frequency_upper = [list(node_degree_upper.values()).count(deg) for deg in unique_degree_upper]

    # Fit a curve to the data with a Power Fit
    def power_law(x, a, b):
        return a * np.power(x, b)

    try:
        params, covariance = curve_fit(power_law, unique_degree_upper, degree_frequency_upper)
    except RuntimeError:
        return None

    hierarchy_index = -params[1]
    #print(hierarchy_index)

    if plotting_wanted:
        plt.scatter(np.log(unique_degree_upper), np.log(degree_frequency_upper), c=[0.24, 0.15, 0.66], s=300, marker='o')
        plt.plot(np.log(unique_degree_upper), params[0] * np.log(unique_degree_upper) ** params[1], linewidth=3, color=[0.96, 0.73, 0.23])
        plt.xlim([0, 4])
        plt.ylim([0, 4])
        plt.xticks([0, 1, 2, 3, 4])
        plt.yticks([0, 1, 2, 3, 4])
        plt.xlabel('log(Degree)')
        plt.ylabel('log(Frequency)')
        plt.grid(True)
        plt.show()

        if saving_wanted:
            plt.savefig(f'{savepath}Hierarchy_Index_Plot.png', format='png')
            plt.close()

    return hierarchy_index


# Adjustable variables
savepath = 'D:/WestbrueckData/Analysis/'
os.chdir('D:/WestbrueckData/Pre-processing/') # dir for input data

# 26 participants with 5x30min VR training less than 30% data loss
#part_list = [1004, 1005, 1008, 1010, 1011, 1013, 1017, 1018, 1019, 1021, 1022, 1023, 1054, 1055, 1056, 1057, 1058, 1068, 1069, 1072, 1073, 1074, 1075, 1077, 1079, 1080]
part_list = [1004]

# Initialize data structures to store results with data from all participants
overview_num_nodes = []
overview_num_edges = []
overview_density = []
overview_diameter = []
overview_hirarchy_index = []
overview_session = []

for part in part_list:

    # Load data from csv
    gazes_data_df = pd.read_csv( f'{part}_segmented_gaze_data_WB.csv')


    # Data structures to save data on graph-measurements and node characteristics
    graph_measurements = pd.DataFrame(columns=['TimeStep',
                                      'NumNodes', 
                                      'NumEdges', 
                                      'Density',
                                      'Diameter',
                                      'HierarchyIndex',
                                      'Session'])

    all_nodes_list = gazes_data_df['hitObjectColliderName'].unique().tolist()
    node_df = pd.DataFrame(columns=all_nodes_list)
    node_df.drop(columns=['noData', 'newSession'], inplace=True)
    node_degree_df = node_df.copy()
    between_centality_results_df = node_df.copy()



    # iterating through the segments, making 150 snapchots of the graph
    for i in range(150):

        # Each detected index is signaling the last entry for a sub-graph    
        current_data = gazes_data_df[gazes_data_df['segment'] <= i]
        
        
        # Remove rows with 'NH'(no house) in the 'hitObjectColliderName' column
        gaze_data = current_data[current_data['hitObjectColliderName'] != 'NH']
        

        # Create a graph for the current data subset
        node_table = gaze_data.loc[:, ['hitObjectColliderName','segment']].copy()
        node_table.rename(columns={"hitObjectColliderName": "name"}, inplace=True)
        node_table = node_table.drop_duplicates(subset=['name'], keep='first')

        edge_table = gaze_data.loc[:, ['hitObjectColliderName','segment']].copy()
        edge_table['orig'] = edge_table['hitObjectColliderName'].shift(1)
        edge_table.rename(columns={"hitObjectColliderName": "dest"}, inplace=True)
        edge_table.dropna(inplace=True)
        # drop self references
        edge_table = edge_table[edge_table['orig'] != edge_table['dest']]
        # drop duplicates (both directions a-b = b-a) - nx-graph does this automatically
        #edge_table = edge_table[~edge_table[['orig', 'dest']].apply(lambda row: frozenset(row), axis=1).duplicated(keep='first')]
        #edge_table = edge_table.drop_duplicates(subset=['orig', 'dest'], keep='first')

        # Create a undirected, unweighted graph
        graph = nx.Graph()

        node_table = node_table.sort_index()
        edge_table = edge_table.sort_index()

        for idx, node in node_table.iterrows():
            graph.add_node(node['name'], segment=node['segment'], index=idx)

        for idx, edge in edge_table.iterrows():
             graph.add_edge(edge['orig'],edge['dest'], segment=edge['segment'], index=idx)


        # Remove nodes with attached edges nodes: 'noData' and 'newSession'
        nodes_to_remove = ['noData', 'newSession']
        nodes_to_remove_existing = [node for node in nodes_to_remove if node in graph.nodes]
        graph.remove_nodes_from(nodes_to_remove_existing)

        #save graph
        #nx.write_graphml(graph, f'{save_graph_path}/{current_part}/{i}_step_graph.graphml')

        # Calculate and save graph measures
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)

        
        max_edges = (num_nodes * (num_nodes - 1)) / 2  if num_nodes > 1 else 0
        density = num_edges / max_edges if max_edges > 0 else 0
       
        # Calculate diameter (if the graph is connected)
        diameter = nx.diameter(graph) if nx.is_connected(graph) else 0

        # variable indicating which of the 5x30min experiment session the data belongs
        current_session = i//30 +1

        hierarchy_idx = calculate_hierarchy_index(graph)


        node_degree = nx.degree(graph)
        for key, value in node_degree:
            if key in node_degree_df.columns:
                node_degree_df.loc[i, key] = value
        
        between_centality = nx.betweenness_centrality(graph)
        for key, value in between_centality.items():
            if key in between_centality_results_df.columns:
                between_centality_results_df.loc[i, key] = value

        node_degree_df['Measure'] = 'NodeDegree'
        between_centality_results_df['Measure'] = 'BetweenCentr'

        all_measurements_df = pd.concat([node_degree_df,
                                         between_centality_results_df
                                     ])
        all_measurements_df.to_csv(savepath + f'{part}_node_measurements.csv', index=True)

        row_data = [i, 
                    num_nodes, 
                    num_edges, 
                    density, 
                    diameter, 
                    hierarchy_idx, 
                    current_session]
        
        graph_measurements.loc[len(graph_measurements)] = row_data
        
    # Append the measures for this participant to the overview data
    overview_num_nodes.append(graph_measurements['NumNodes'].tolist())
    overview_num_edges.append(graph_measurements['NumEdges'].tolist())
    overview_density.append(graph_measurements['Density'].tolist())
    overview_diameter.append(graph_measurements['Diameter'].tolist())
    overview_hirarchy_index.append(graph_measurements['HierarchyIndex'].tolist())
    overview_session.append(graph_measurements['Session'].tolist())

    print(row_data) #########################################

# Save the overview data to separate CSV files
overview_df = pd.DataFrame({
    'Participant': part_list,
    'Session': overview_session,
    'NumNodes': overview_num_nodes,
    'NumEdges': overview_num_edges,
    'Density': overview_density,
    'Diameter': overview_diameter,
    'HierarchyIndex': overview_hirarchy_index
    })


overview_df.to_csv(savepath + 'overview_graph_measures.csv', index=False)

print('Finished')