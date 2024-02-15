import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

######## Graph measures #########



def num_of_nodes(graph):
    """
    Calculating the number of nodes in my graph.

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - _ (int): Number of nodes

    """
    return len(graph.nodes)



def num_of_edges(graph):
    """
    Calculating the number of edges in my graph.

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - _ (int): Number of edges

    """
    return len(graph.edges)



def density(graph):
    """
    Calculating the density (=num_edges/possible_edges) of my graph

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - density (float): Density of my graph

    """

    num_nodes = num_of_nodes(graph)
    num_edges = num_of_edges(graph)

    max_edges = (num_nodes * (num_nodes - 1)) / 2  if num_nodes > 1 else 0
    density = num_edges / max_edges if max_edges > 0 else 0

    return density



def diameter(graph):

    """
    Calculating the diameter of my graph (or its largest component). 

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - diameter (int): Diameter of my graph (or the largest component)

    """

    #if the graph is connected with the interenal nx-function)
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
    # if there are isolated nodes or multiple disconnected components, then set any infinitive distances to 0 
    else:
        distanceM = dict(nx.all_pairs_shortest_path_length(graph))
        for source in distanceM:
            distanceM[source] = {target: 0 if distance == float('inf') else distance for target, distance in distanceM[source].items()}
        
        # Find the maximum distance (diameter) from the disconnected components
        diameter = int(max(max(distanceM[source].values()) for source in graph.nodes))
    
    return diameter



def average_shortest_path_length(graph):
    """
    Calculating the average shortest path length of my graph. 

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - avg_shortest_path (float): Average shortest path length of my graph

    """

    if nx.is_connected(graph):
        avg_shortest_path = nx.average_shortest_path_length(graph)

    else:
        distanceM = dict(nx.all_pairs_shortest_path_length(graph))

        # Modify the distance matrix to exclude infinitive distances (between not connected nodes) and self-references
        for source in distanceM:
            distanceM[source] = {target: distance for target, distance in distanceM[source].items() if target != source and distance != float('inf')}

        # Calculate the average shortest path
        num_paths = sum(len(distanceM[source]) for source in graph.nodes)
        total_distance = sum(sum(distanceM[source].values()) for source in graph.nodes)

        avg_shortest_path = total_distance / num_paths if num_paths > 0 else float('inf')

    return avg_shortest_path



def hierarchy_index(graph):
    """
    Calculating the hierarchy index (= exponent of exponential decay slope of the degree frequencies) of my graph.
    
    Based on a script from: Github: JasminLWalter/VR-EyeTracking-GraphTheory/GraphTheory_ET_VR_Westbrueck/Analysis/Hierarchy_Index.m
    Which has been written by Lucas Essmann (2020, lessmann@uni-osnabrueck.de) and adjusted & extended by Jasmin L. Walter (2022/23, jawalter@uos.de)

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - hierarchy_index (float): Hierarchy Index of my graph

    """

    # get Node Degrees
    node_degrees = dict(graph.degree())

    # Delete 0 degrees (non-connected nodes)
    node_degrees = {k: v for k, v in node_degrees.items() if v != 0}
    
    # Calculate Median of Node Degree
    median_degree = np.median(list(node_degrees.values()))
    
    # Filter degrees above the median
    node_degrees_upper = {k: v for k, v in node_degrees.items() if v > median_degree}

    # Get unique degrees above the median
    unique_degrees_upper = sorted(set(node_degrees_upper.values()))

    # Calculate Degree Frequency for degrees above the median
    degree_frequency_upper = [list(node_degrees_upper.values()).count(deg) for deg in unique_degrees_upper]


    # Fit a linear curve to the logarithmic data = exponential relationship
    unique_degrees_upper_log = np.log(unique_degrees_upper)
    degree_frequency_upper_log = np.log(degree_frequency_upper)

    def linear_function(x, a, b):
        return a * x + b

    try:
        popt_exp, _ = curve_fit(linear_function, unique_degrees_upper_log, degree_frequency_upper_log)
    except RuntimeError:
        return None

    # hierarchy_index = negative slope of the linear regression, 
    # corresponding to the exponent in the exponential relationship in my degree_frequency data 
    hierarchy_index = - popt_exp[0]

    return hierarchy_index






def custom_weight(edge, edge_table):
    """
    Calculates the weight of an edge from the edges of my undirected graph.

    The more often an edge appears in the edge-table, the smaller the weight (inverse frequency)
    Parameters:

    Returns:
    weight <float>: the weight of my undirected edge.
    
    """
    # Count the number of times the edge appears in the edge table

    frequency_a = edge_table[(edge_table['orig'] == edge[0]) & (edge_table['dest'] == edge[1])].shape[0]
    frequency_b = edge_table[(edge_table['orig'] == edge[1]) & (edge_table['dest'] == edge[0])].shape[0]
    
    frequency = frequency_a + frequency_b

    # Assign a weight based on the inverse of the frequency
    weight = 1.0 / (frequency + 1)
    #weight = frequency
    #weight = 1
    return weight


def diameter_weighted(graph):
    """
    Calculating the diameter of a weighted graph (or its largest component). 

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - diameter (int): Diameter of the weighted graph (or the largest component)

    """

    #if the graph is connected with the interenal nx-function)
    if nx.is_connected(graph):

        diameter = nx.diameter(graph,weight='weight')

    # if there are isolated nodes or multiple disconnected components, then set any infinitive distances to 0 
    else:
        distanceM = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
        for source in distanceM:
            distanceM[source] = {target: 0 if distance == float('inf') else distance for target, distance in distanceM[source].items()}
        
        # Find the maximum distance (diameter) from the disconnected components
        diameter = int(max(max(distanceM[source].values()) for source in graph.nodes))
    
    return diameter


def average_shortest_path_length_weighted(graph):
    """
    Calculating the average shortest path length of a weighted graph. 

    Parameters:
    - graph (networkx.classes.graph.Graph): NetworkX graph object

    Returns: 
    - avg_shortest_path (float): Average shortest path length of the weighted graph

    """

    if nx.is_connected(graph):

        avg_shortest_path = nx.average_shortest_path_length(graph, weight = 'weight')

    else:
        distanceM = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))

        # Modify the distance matrix to exclude infinitive distances (between not connected nodes) and self-references
        for source in distanceM:
            distanceM[source] = {target: distance for target, distance in distanceM[source].items()  if target != source and distance != float('inf')}

        # Calculate the average shortest path
        num_paths = sum(len(distanceM[source]) for source in graph.nodes)
        total_distance = sum(sum(distanceM[source].values()) for source in graph.nodes)

        avg_shortest_path = total_distance / num_paths if num_paths > 0 else float('inf')


    return avg_shortest_path

