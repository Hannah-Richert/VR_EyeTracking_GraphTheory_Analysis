# Graph Analysis on Eye-Tracking Data of VR City Exploration <br> Bsc-Thesis Repository

This is a repository belonging to a BSc-Thesis at the Neurobiopsychology Group at the University of Osnabrück with the title:
"Exploratory Analysis of Visual Behavior During Free Exploration of a Large-Scale VR Environment Using a Graph-Theoretical Framework to Identify Characteristics and Differences in Spatial Exploration"

Contact for further details about the thesis or any script: Hannah Richert hrichert@uos.de

This readme will provide rough documentation about the files' structure and purpose, with further documentation at the beginning of each file itself.


## 0. Data Origin

The raw Eye-tracking data stems from an experiment by Schmidt et al. (2022):
It was further processed by a pipeline by Jasmin Walter: Github:

The resulting data are "gazes" of participants, saved in Matlab format.


## 1. Data Processing & Graph Creation Pipeline [.../graph_creation/...]

#### 1.1 .mat to .csv conversion [Step1_mat_to_csv.py]
In the first instance, '.mat' files are converted into '.csv' files to make working in Python easier.

#### 1.2 Saving Node Positions [Step1.5_save_abs_node_positions]
As the graph nodes are buildings in a VR environment, we save all buildings' coordinates for later graph plotting.

#### 1.3 Gaze Segmentation [Step2_gata_segmentation.py]
In this step, the data samples are divided into time segments; an added column indicates the time segment of each gaze/sample.

#### 1.4 Graph Creation [Step3_temp_graph_development.py; graph_measure_functions.py]
In the third step, multiple network graphs are created from the gazes and their properties are saved.
<br> The functions of the graph measures are stored in a separate file.
<br> The saved values are the basis for the analysis.

#### 1.5 Additional Files [.../graph_creation/additional/...]
This folder contains 2 short files, with which one can create and measure a single graph snapshot from '.mat' (before 1.1) or '.csv' data files.
<br> They are from an earlier stage in the thesis-process and are simpler than the script of 1.4 (logic and measures might differ).


## 2. Data Analysis & Plotting [.../graph_analysis/...]

#### 2.0 Utils-Files [analysis_plotting_functions.py]

#### 2.1 Graph Growth Analysis [growth_analysis.py]

#### 2.2 Graph Structure Analysis [diameter_analysis.py;avgpath_analysis.py; growth_analysis.py]

#### 2.3 Spatial Knowledge Performance Analysis [performance_global_features.py; performance_lm.py]

#### 2.4 Graph Feature Complexity Analysis [pca.py]

#### 2.5 Graph Hierarchy/Landmark Analysis [hierarchy_analysis.py; lm_analysis.py]

#### 2.6 Drawing Graph [draw_graph.py]


## 3. Additional Files

#### Requirements [requirements.txt]
Entails the Python packages, which need to be installed to run all scripts

#### Color Codes [color_codes.txt]
Entails the main colors used in the figures.


