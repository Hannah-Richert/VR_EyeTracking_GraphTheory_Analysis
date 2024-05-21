# Graph Analysis on Eye-Tracking Data of VR City Exploration
## Bsc-Thesis Repository

This is a repository belonging to a Bsc-Thesis at the Neurobiopsychology Group at the University of Osnabrück with the title:
"Exploratory Analysis of Visual Behavior During Free Exploration of a Large-Scale VR Environment Using a Graph-Theoretical Framework to Identify Characteristics and Differences in Spatial Exploration"

Contact for further details about the thesis or any script: Hannah Richert hrichert@uos.de

This readme will provide rough documentation about the files' structure and purpose, with further documentation at the beginning of each file itself.


### 0. Data Origin

The raw Eye-tracking data stems from an experiment by Schmidt et al. (2022):
It was further processed by a pipeline by Jasmin Walter: Github:

The resulting data are "gazes" of participants, saved in Matlab format.


### 1. Data Processing & Graph Creation Pipeline [.../graph_creation/...]

#### 1.1 .mat to .csv conversion [Step1_mat_to_csv.py]
In the first instance, '.mat' files are converted into '.csv' files to make working in Python easier.

#### 1.2 Saving Node Positions [Step1.5_save_abs_node_positions]
As the graph nodes are buildings in a VR environment, we save all buildings' coordinates for later graph plotting.

#### 1.3 Gaze Segmentation [Step2_gata_segmentation.py]
In this step, the data samples are divided into time segments; an added column indicates the time segment of each gaze/sample.

#### 1.4 Graph Creation [Step3_temp_graph_development.py; graph_measure_functions.py]
In the third step, multiple network graphs are created from the gazes and their properties are saved.
The function for the graph measures is saved separately and called.

The saved values are the basis for the analysis.


### 2. Data Analysis
