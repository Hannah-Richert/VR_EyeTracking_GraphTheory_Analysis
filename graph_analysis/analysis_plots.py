import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from scipy.stats import pearsonr

warnings.simplefilter(action='ignore', category=FutureWarning)

def continuous_distribution_hist(data,x_label,y_label,fig_title, savepath,bins=10,color='skyblue', show = True):
    '''

    # Plot distribution of max_percentage, means, and  max_values 
    # Create a histogram with KDE (Kernel Density Estimate)
    '''
    #data = data.astype(int)
    #sns.histplot(data, kde=True, bins=range(min(data), max(data) + 2), color=color)
    sns.histplot(data, kde=True, bins=bins, color=color)

    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

def data_and_avg_line_plot(data,x_label,y_label,fig_title, savepath, color='red', show = True):
    '''
    #--------------------------------------------------#
    # Plot Exploration Efficiency / Graph Growth over time
    # Plot individual and avg curves

    
    '''

    plt.figure(figsize=(12, 6))

    for _, row in data.iterrows():
        plt.plot(row.index, row.values, alpha=0.5)

    # means per ts
    mean_values_per_ts = data.mean(axis=0)
    # Plot the mean line
    plt.plot(mean_values_per_ts.index, mean_values_per_ts.values, color='red', linestyle='-', linewidth=2, label='Mean')

    # Customize labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    xticks = np.arange(10,len(data.columns)+1, 10)  # Show every 10th time step
    xticks = np.array([1] + xticks.tolist()) # include the first one
    plt.xticks(xticks)
    plt.legend()

    # Show the plot
    plt.savefig(savepath)
    
    if show == True:
        plt.show()
    plt.close()



def boxplot(data,x_label,y_label,fig_title, savepath, color='red', show = True):
# Create boxplots 
    plt.figure(figsize=(12, 6))

    sns.boxplot(data=data, orient='v', showmeans=True, meanline=True, meanprops={"linestyle": "-", "linewidth": 2, "color": "blue"})  # 'v' for vertical boxplot
    violinplot
    # Customize labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    xticks = np.arange(0, len(data.columns), 10)  # Show every 10th time step
    plt.xticks(xticks)

    # Show the plot
    plt.savefig(savepath)
    
    if show == True:
        plt.show()
    plt.close()

def violinplot(data,x_label,y_label,fig_title, savepath, color='red', show = True):
# Create boxplots 
    plt.figure(figsize=(12, 6))

    sns.violinplot(data=data, orient='v', showmeans=True, meanline=True, meanprops={"linestyle": "-", "linewidth": 2, "color": "blue"})  # 'v' for vertical boxplot
    
    # Customize labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    xticks = np.arange(0, len(data.columns), 10)  # Show every 10th time step
    plt.xticks(xticks)

    # Show the plot
    plt.savefig(savepath)
    
    if show == True:
        plt.show()
    plt.close()


def identify_peaks(data, prominence, distance, width, x_label, y_label, fig_title, savepath, show=True):

    # Identify peaks in the data
    peaks = []
    for row_idx in data.index:
        row_data = data.iloc[row_idx]
        peak_indices, peak_info = find_peaks(row_data, prominence = prominence, distance=distance, width=width)
        peaks.extend(list(zip([row_idx] * len(peak_indices), peak_indices, row_data.iloc[peak_indices])))
        plt.plot(data.columns, row_data, alpha= 0.5)
        #plt.scatter(peak_indices, row_data.iloc[peak_indices], c='red', marker='x', label='Peaks')
        #plt.show()

    # Convert the peaks list to a DataFrame for easier analysis
    peaks_df = pd.DataFrame(peaks, columns=['Participant', 'Peak Index', 'Peak Value'])

    plt.scatter(peaks_df['Peak Index'], peaks_df['Peak Value'], c='red', marker='x', label='Peaks')

    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

def scatter_corr(x_data,y_data,x_label,y_label,fig_title, savepath,  show = True,regression=False):


    plt.scatter(x_data, y_data)

    if regression:
    # Extracting x and y values
        x = x_data.values.reshape(-1, 1)
        y = y_data.values

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Get the slope and intercept of the regression line
        slope = model.coef_[0]
        intercept = model.intercept_

        
        plt.plot(x_data, slope * x_data + intercept, color='red', label='Regression Line')

    # Fit a linear regression model
    #X = x_data.reshape(-1, 1)  # Reshape to a column vector
    #model = LinearRegression()
    #model.fit(X, y_data)

    # Plot the regression line
    #x_line = np.linspace(min(x_data), max(x_data), 100).reshape(-1, 1)
    #y_pred = model.predict(x_line)
    #plt.plot(x_line, y_pred, color='red', label='Regression Line')
    #correlation_coefficient, p_value = pearsonr(x_data, y_data)
    # Display results
    #print("Correlation coefficient:", correlation_coefficient)
    #print("P-value:", p_value)
    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.legend()
    
    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

def scatter_corr_cat(x_data,y_data,x_label,y_label,fig_title, savepath, category= None,  show = True,regression=False):


    for cat in [7,8,9]:
        mask = category == cat
        plt.scatter(x_data[mask], y_data[mask], label=cat)

    #plt.scatter(x_data, y_data, c= category)
    if regression:
    # Extracting x and y values
        x = x_data.values.reshape(-1, 1)
        y = y_data.values

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Get the slope and intercept of the regression line
        slope = model.coef_[0]
        intercept = model.intercept_

        
        plt.plot(x_data, slope * x_data + intercept, color='red', label='Regression Line')

    # Fit a linear regression model
    #X = x_data.reshape(-1, 1)  # Reshape to a column vector
    #model = LinearRegression()
    #model.fit(X, y_data)

    # Plot the regression line
    #x_line = np.linspace(min(x_data), max(x_data), 100).reshape(-1, 1)
    #y_pred = model.predict(x_line)
    #plt.plot(x_line, y_pred, color='red', label='Regression Line')
    correlation_coefficient, p_value = pearsonr(x_data, y_data)
    # Display results
    print("Correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)
    plt.legend(title='End Diameter Value')
    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.legend()
    
    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

    
def heatmap(data,x_label,y_label,fig_title, savepath, cmap='viridis',cbar=True, show = True):

    plt.figure(figsize=(12, 6))
    '''
    #--------------------------------------------------#
    # Create a  heatmap
    '''
    sns.heatmap(data, cmap=cmap, cbar=cbar)
    # Customize labels and title
    plt.xlabel(x_label)
    #xticks = np.arange(0, len(data.columns), 10)  # Show every 10th time step
    #plt.xticks(xticks)
    plt.ylabel(y_label)
    plt.title(fig_title)

    # Show the plot
    plt.savefig(savepath)
    
    if show == True:
        plt.show()
    plt.close()


def categorical_stacked_barplot(data,x_label,y_label,fig_title, savepath, show = True):

    # Stacked bar chart
    bottom = np.zeros(len(data))

    data.index = data.index.astype(str)

    for category in data.columns[:]:
        plt.bar(data.index, data[category], label=category, bottom=bottom, alpha= 1)
        bottom += data[category]

    # Customize labels, legend and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Category',  loc='upper left')
    if len(data.index) > 100:
        xticks = np.arange(0, len(data.index), 10)  # Show every 10th time step
        plt.xticks(xticks)
    plt.title(fig_title)

    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

def categorical_barplot(data,x_label,y_label,fig_title, savepath, show = True):

    for category in data.columns[:]:
        plt.bar(data.index, data[category], label=category, alpha= 0.7)

    # Customize labels, legend and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Category',  loc='upper left')
    if len(data.index) > 100:
        xticks = np.arange(0, len(data.index), 10)  # Show every 10th time step
        plt.xticks(xticks)
    plt.title(fig_title)

    plt.savefig(savepath)

    if show == True:
        plt.show()
    plt.close()

def grouped_barplot(data,x_label,y_label,fig_title, savepath, show = True):

    # Grouped bar chart
    bar_width = 0.35  # Width of the bars
    bar_positions = np.arange(len(data.index))  # Positions of bars on X-axis

    for i, category in enumerate(data.columns[:]):
        plt.bar(bar_positions + i * bar_width, data[category], label=category, width=bar_width, alpha=0.7)

    # Customize labels, legend, and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Category', loc='upper left')
    
    if len(data.index) > 100:
        xticks = np.arange(0, len(data.index), 10)  # Show every 10th time step
        plt.xticks(xticks, data.index[xticks])

    plt.title(fig_title)

    plt.savefig(savepath)

    if show:
        plt.show()
    plt.close()





def smooth_groups(num_groups, group_idx_list,data_df, subtitle_list, x_label, y_label, fig_title):

    fig, ax = plt.subplots(num_groups, 1, figsize=(8, 12), sharex = True, sharey = True)

    for idx, set in enumerate(group_idx_list):
        #peaks = []

        set_data = pd.DataFrame(index=range(len(set)),columns=data_df.columns)
        
        for idx_2, i in enumerate(set):

            #max_diameter_idx = max_diameters_index.iloc[i]
            y_data = data_df.iloc[i]

            y_data_smooth = data_df.iloc[i].rolling(window=10, center=True).mean()
            #y_data_smooth = y_data

            # fill nans with orig. values
            #y_data_smooth.iloc[:5] = data_df.iloc[i, :5].values  # Assuming a window of 10, fill first 5 values
            #y_data_smooth.iloc[-5:] = data_df.iloc[i, -5:].values 

            x_data = data_df.columns
            set_data.iloc[idx_2] = y_data

            ax[idx].plot(x_data,y_data_smooth,alpha = 1)


        mean_values_per_ts = set_data.mean(axis=0)
        glob_mean_values_per_ts = data_df.mean(axis=0)


        ax[idx].plot(mean_values_per_ts.index, mean_values_per_ts.values, color='blue', linestyle='-', linewidth=2, label='Group Mean (unsmoothed data)')
        
        ax[idx].plot(glob_mean_values_per_ts.index, glob_mean_values_per_ts.values, color='red', linestyle='-', linewidth=2, label='Global Mean (unsmoothed data)')
    
    ax[0].legend()
    ax[0].set_title(subtitle_list[0])
    ax[1].set_title(subtitle_list[1])
    ax[2].set_title(subtitle_list[2])
    #fig.legend()
    fig.suptitle(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #fig.tight_layout()
    plt.show()


def smooth_line_plot(data,x_label, y_label,fig_title,savepath,color = 'blue', show= True):
    
        #max_diameter_idx = max_diameters_index.iloc[i]

        data_smooth = data.copy()

        for i in range(len(data)):

            y_data = data.iloc[i]

            y_data_smooth = data.iloc[i].rolling(window=10, center=True).mean()

            # fill nans with orig. values
            #y_data_smooth.iloc[:5] = data.iloc[i, :5].values  # Assuming a window of 10, fill first 5 values
            #y_data_smooth.iloc[-5:] = data.iloc[i, -5:].values 

            data_smooth.iloc[i] = y_data_smooth

            x_data = data.columns

            plt.plot(x_data,y_data_smooth,alpha = 1)


        mean_values_per_ts = data.mean(axis=0)

        plt.plot(mean_values_per_ts.index, mean_values_per_ts.values, color=color, linestyle='-', linewidth=2, label='Mean (smoothed data)')
    
        # Customize labels, legend and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        if len(data.index) > 100:
            xticks = np.arange(0, len(data.index), 10)  # Show every 10th time step
            plt.xticks(xticks)
        plt.title(fig_title)

        plt.savefig(savepath)

        if show == True:
            plt.show()
        plt.close()


































'''# Sum the values for each category for each participant
sums_df = pd.DataFrame(index=growth_df.index)

for i in range(1,7):
    sums_df[f'{i}'] = categorized_df[categorized_df == i].sum(axis=1)/i
    #print(i, sums_df[f'{i}'])


# Set up the plot
#plt.figure(figsize=(12, 8))

print(sums_df.columns)

# Loop through each category and create a histplot
for i, col in enumerate(sums_df.columns):

    sns.histplot(data=sums_df.loc[:,col], legend = col)

#sns.histplot(data=sums_df)

# Add legend
plt.legend()

# Show the plot
plt.savefig(savepath + f'Distri_244_{i}')
plt.show()



# Create a histogram with KDE (Kernel Density Estimate)
sns.histplot(participant_mean, kde=True, bins=26, color='skyblue')

# Set labels and title
plt.xlabel('Mean Growth Values')
plt.ylabel('Frequency')
plt.title('Distribution of Participant Means')
plt.show()

# Create boxplots 



plt.figure(figsize=(12, 6))

#flat_data = [item for sublist in g_measures_df['GrowthPercentage'] for item in sublist]

# Create a new DataFrame with the flattened data
#flat_df = pd.DataFrame({'GrowthPercentage': flat_data})

# Set up the plot
#plt.figure(figsize=(15, 6))
sns.boxplot(data=growth_df, orient='v',  showmeans=True, meanline=True, meanprops={"linestyle": "-", "linewidth": 2, "color": "blue"})  # 'v' for vertical boxplot
#sns.swarmplot(data=boxplot_df, orient='v')  # 'v' for vertical boxplot

# Create a boxplot
#sns.boxplot(x=flat_df['GrowthPercentage'])
#sns.swarmplot(x=flat_df['GrowthPercentage'])
#sns.boxplot(x='TimeStep', y=col, data=df_long_first_30, showfliers=False)
#sns.swarmplot(x='TimeStep', y=col,hue='Participant', data=df_long_first_30, marker='.', alpha=0.7)

plt.title('Boxplot of NumNodes Variance for All Participants')
plt.xlabel('Timestep')
# Customize x-axis ticks
xticks = np.arange(1, 150, 10)  # Show every 10th time step
plt.xticks(xticks)
plt.ylabel('NodeDiscovery Percentage')

# Show the plot
plt.savefig(savepath + f'Growth_Distributions_normalized')
plt.show()


# Calculate the mean for each time step
mean_values = np.mean(growth_array, axis=0)

# Calculate the differences from the mean for each participant
difference_array = growth_array - mean_values

# Create a DataFrame for boxplotting
boxplot_df = pd.DataFrame(difference_array, columns=[f'{i+1}' for i in range(growth_array.shape[1])])

# Set up the plot
plt.figure(figsize=(15, 6))
sns.boxplot(data=boxplot_df, showmeans=True, meanline=True, meanprops={"linestyle": "-", "linewidth": 2, "color": "blue"})

plt.title('Boxplot of NumNodes Variance for All Participants')
plt.xlabel('Timestep')
# Customize x-axis ticks
xticks = np.arange(1, 150, 10)  # Show every 10th time step
plt.xticks(xticks)
plt.ylabel('NodeDiscovery Percentage')

# Show the plot
#plt.savefig(savepath + f'Growth_Mean_Variance_244')
#plt.show()'''


'''plt.figure(figsize=(25,10)) 

m_df = g_measures_df['GrowthPercentage']#.reset_index(drop=True)
print(m_df)

# Convert the 'GrowthPercentage' column to a list of lists
data_list = g_measures_df['GrowthPercentage'].tolist()

# Create a NumPy array from the list of lists
data_array = [np.array(row) for row in data_list]

print(m_df.shape)
sns.heatmap(data_array, cmap='viridis', cbar_kws={'label': 'NodePercentage'})
plt.title(f'Graph Growth Over Time')
plt.xlabel('Time Step')
plt.ylabel('Percentage of Discoveries')

plt.savefig(savepath + f'Discoverys_over_time')
plt.show()'''
