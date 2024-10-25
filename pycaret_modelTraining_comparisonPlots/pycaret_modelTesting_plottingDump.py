import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('lightgbm_soloVSlong_featRed.csv')

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    pivot_table.plot(kind='bar', figsize=(10, 7))
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('lightgbm_soloVSlong_featRed.csv')

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    pivot_table.plot(kind='bar', figsize=(10, 7))
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('lightgbm_soloVSlong_featRed.csv')

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    pivot_table.plot(kind='bar', figsize=(10, 7))
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"lightgbm_soloVSlong_{metric}.png")

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    pivot_table.plot(kind='bar', figsize=(10, 7))
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    pivot_table.plot(kind='bar', figsize=(10, 7))
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

dataset_order = [
    "noPMA_1176Features", "PMA_1176Features", "PMA+noPMA_2352Features", 
    "UMAP01_noPMA_100Features", "UMAP01_PMA_100Features", "UMAP01_Horiz_PMA+noPMA_100Features", 
    "FeatRed08_noPMA_482Features", "FeatRed08_PMA_482Features", "FeatRed08_PMA+noPMA_964Features"
]
# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    ax = pivot_table.plot(kind='bar', figsize=(10, 7))
    
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

dataset_order = [
    "noPMA_1176Features", "PMA_1176Features", "PMA+noPMA_2352Features", 
    "UMAP01_noPMA_100Features", "UMAP01_PMA_100Features", "UMAP01_Horiz_PMA+noPMA_100Features", 
    "FeatRed08_noPMA_482Features", "FeatRed08_PMA_482Features", "FeatRed08_PMA+noPMA_964Features"
]

# Ensure the DataFrame is ordered by the specified datasets
df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    ax = pivot_table.plot(kind='bar', figsize=(10, 7))
    
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

dataset_order = [
    "noPMA_1176Features", "PMA_1176Features", "PMA+noPMA_2352Features", 
    "UMAP01_noPMA_100Features", "UMAP01_PMA_100Features", "UMAP01_Horiz_PMA+noPMA_100Features", 
    "FeatRed08_noPMA_482Features", "FeatRed08_PMA_482Features", "FeatRed08_PMA+noPMA_964Features"
]

# Ensure the DataFrame is ordered by the specified datasets
df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    ax = pivot_table.plot(kind='bar', figsize=(10, 7))
    
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1))#, loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Replace the following with your data dictionary or DataFrame creation code

# Create DataFrame
df = pd.read_csv('pycaret_featRedMethod_modelTesting.csv')

dataset_order = [
    "noPMA_1176Features", "PMA_1176Features", "PMA+noPMA_2352Features", 
    "UMAP01_noPMA_100Features", "UMAP01_PMA_100Features", "UMAP01_Horiz_PMA+noPMA_100Features", 
    "FeatRed08_noPMA_482Features", "FeatRed08_PMA_482Features", "FeatRed08_PMA+noPMA_964Features"
]

# Ensure the DataFrame is ordered by the specified datasets
df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)

# Define the metrics columns (excluding 'Top 5 Accuracy')
metrics_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)','Balanced Accuracy']

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    ax = pivot_table.plot(kind='bar', figsize=(12, 7))
    
    ax.legend(title='Dataset', bbox_to_anchor=(1.2, 1))#, loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

np.mean
import numpy as np, pandas as pd, os,sys,glob
# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    
    # Increase bar width by reducing the width of all bars slightly
    bar_width = 0.8 / len(dataset_order)  # Slightly reduce the total width
    
    ax = pivot_table.plot(kind='bar', width=bar_width, figsize=(14, 7))  # Increase the width of the plot

    # Adjust legend placement completely outside the plot
    ax.legend(title='Dataset', bbox_to_anchor=(1.25, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    
    # Increase bar width by reducing the width of all bars slightly
    bar_width = 0.8 / len(dataset_order)  # Slightly reduce the total width
    
    ax = pivot_table.plot(kind='bar', width=1.01, figsize=(14, 7))  # Increase the width of the plot

    # Adjust legend placement completely outside the plot
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    
    # Increase bar width by reducing the width of all bars slightly
    bar_width = 1.01 / len(dataset_order)  # Slightly reduce the total width
    
    ax = pivot_table.plot(kind='bar', width=bar_width, figsize=(14, 7))  # Increase the width of the plot

    # Adjust legend placement completely outside the plot
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

# Generate grouped bar plots for each metric
for metric in metrics_columns:
    pivot_table = df.pivot_table(index='Model', columns='Dataset', values=metric, aggfunc='mean')
    
    # Increase bar width by reducing the width of all bars slightly
    bar_width = 1.01 #/ len(dataset_order)  # Slightly reduce the total width
    
    ax = pivot_table.plot(kind='bar', width=bar_width, figsize=(20, 7))  # Increase the width of the plot

    # Adjust legend placement completely outside the plot
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Grouped Bar Plot for {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"FeatRed_modelComparisons_{metric}.png")
    plt.close()

import plotly.graph_objs as go
from plotly.subplots import make_subplots
# Generate grouped bar plots for each metric
for metric in metrics_columns:
    fig = make_subplots(rows=1, cols=1)

    # Add bars for each dataset
    for dataset in dataset_order:
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df[df['Dataset'] == dataset][metric],
            name=dataset
        ))

    # Update layout for spacing and legend
    fig.update_layout(
        title=f'Grouped Bar Plot for {metric}',
        xaxis=dict(title='Model', tickangle=-45),
        yaxis=dict(title=metric),
        barmode='group',  # Group bars together
        bargap=0.15,      # Gap between bars within a group
        bargroupgap=0.2,  # Gap between bar groups
        legend=dict(
            title='Dataset',
            x=1.05,
            y=1,
            traceorder="normal",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=50, b=150)  # Adjust margins to ensure legend is outside the plot
    )

    # Save the plot as PNG
    png_filename = f'FeatRed_modelComparisons_{metric}.png'
    fig.write_image(png_filename)

    # Save the plot as HTML
    html_filename = f'FeatRed_modelComparisons_{metric}.html'
    fig.write_html(html_filename)

get_ipython().system('pip install kaleido')
get_ipython().run_line_magic('pip', 'install kaleido')
# Generate grouped bar plots for each metric
for metric in metrics_columns:
    fig = go.Figure()

    # Add bars for each dataset
    for dataset in dataset_order:
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df[df['Dataset'] == dataset][metric],
            name=dataset
        ))

    # Update layout for spacing and legend
    fig.update_layout(
        title=f'Grouped Bar Plot for {metric}',
        xaxis=dict(title='Model', tickangle=-45),
        yaxis=dict(title=metric),
        barmode='group',  # Group bars together
        bargap=0.15,      # Gap between bars within a group
        bargroupgap=0.2,  # Gap between bar groups
        legend=dict(
            title='Dataset',
            x=1.05,
            y=1,
            traceorder="normal",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=50, b=150)  # Adjust margins to ensure legend is outside the plot
    )

    # Save the plot as PNG
    png_filename = f'FeatRed_modelComparisons_{metric}.png'
    fig.write_image(png_filename)

    # Save the plot as HTML
    html_filename = f'FeatRed_modelComparisons_{metric}.html'
    fig.write_html(html_filename)

    print(f'Saved {metric} plot as {png_filename} and {html_filename}')

##########

import pandas as pd
import plotly.graph_objs as go

# Load your dataset
df = pd.read_csv('pycaret_catboost_3compClass_modelTesting.csv') 
# Step 2: Identify Dataset Types
# Replace 'DatasetType' with the actual column name that distinguishes different dataset types
dataset_type_column = 'Dataset'  # Update this to match your dataset
if dataset_type_column not in df.columns:
    raise ValueError(f"Column '{dataset_type_column}' not found in the dataset.")

dataset_order = [
    "noPMA_1176Features",
    "PMA_1176Features",
    "PMA+noPMA_2352Features",
    "UMAP01_noPMA_100Features",
    "UMAP01_PMA_100Features",
    "UMAP01_Horiz_PMA+noPMA_100Features",
    "FeatRed08_noPMA_482Features",
    "FeatRed08_PMA_482Features",
    "FeatRed08_PMA+noPMA_964Features",
]

# Ensure the DataFrame is ordered by the specified datasets
df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)

dataset_types = df[dataset_type_column].unique()

assert len(dataset_types) == len(dataset_order)
print(np.array(dataset_types) == np.array(dataset_order))

# Step 3: Define Metrics
# Define the list of metrics that you want to plot
metrics_columns = ['Accuracy', 'F1','Balanced Accuracy']
#'AUC', 'Recall', 'Prec.', 'Kappa', 'MCC',

# Verify that all specified metrics exist in the DataFrame
missing_metrics = [metric for metric in metrics_columns if metric not in df.columns]
if missing_metrics:
    raise ValueError(f"The following metric columns are missing from the dataset: {missing_metrics}")

# Step 4: Create the Plot
fig = go.Figure()

# Define a list of colors for different dataset types (optional customization)
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

for i, dataset in enumerate(dataset_types):
    # Filter the DataFrame for the current dataset type
    dataset_df = df[df[dataset_type_column] == dataset]
    
    # Handle cases where there might be multiple entries for a dataset type
    # For simplicity, we'll assume there's only one entry per dataset type
    if dataset_df.empty:
        print(f"No data found for dataset type '{dataset}'. Skipping.")
        continue
    elif len(dataset_df) > 1:
        print(f"Multiple entries found for dataset type '{dataset}'. Using the first entry.")
        dataset_df = dataset_df.iloc[[0]]
    else:
        dataset_df = dataset_df.iloc[0]

    # Extract metric values
    metric_values = [dataset_df[metric] for metric in metrics_columns]

    # Add a bar trace for the current dataset type
    fig.add_trace(go.Bar(
        x=metrics_columns,    # Metrics on the x-axis
        y=metric_values,      # Corresponding metric values
        name=str(dataset),    # Name for the legend
        marker_color=custom_colors[i % len(custom_colors)]  # Assign a color
    ))

# Step 5: Customize the Layout
fig.update_layout(
    title='Catboost Classifier Model Metrics for 428 MOA Class Annotations',
    xaxis=dict(
        title='Metrics',
        tickangle=-45,
        title_font=dict(size=20),
        tickfont=dict(size=18),
    ),
    yaxis=dict(
        title='Value',
        title_font=dict(size=20),
        tickfont=dict(size=18),
    ),
    barmode='group',       # Group bars together by dataset type
    bargap=0.2,            # Gap between bars within a group
    bargroupgap=0.1,       # Gap between different metric groups
    legend=dict(
        title='Dataset Types',
        x=1.02,            # Position the legend closer to the plot area
        y=1,
        traceorder='normal',
        bordercolor='Black',
        borderwidth=1,
        font=dict(size=18)
    ),
    margin=dict(l=50, r=150, t=50, b=150),  # Adjust margins to accommodate the legend
    width=1200,            # Optional: Increase figure width for better spacing
    height=600             # Optional: Adjust figure height as needed
)

# Optional: Customize the hover template for better interactivity
# fig.update_traces(hovertemplate='Metric: %{x}<br>Value: %{y}<extra></extra>')

# Save plot as PNG
png_filename = f'../CatBoost_3compClass_Model_Metrics_Plot.png'
# fig.write_image(png_filename)

# Save the plot as HTML
html_filename = '../CatBoost_3compClass_Model_Metrics_Plot.html'
fig.write_html(html_filename)

print(f'Saved plot as {html_filename}')


#####################
df_85class = pd.read_csv("pycaret_featRedMethod_modelTesting.csv")
df_85class = df_85class[df_85class['Model'] =='CatBoost Classifier']

# Step 2: Identify Dataset Types
# Replace 'DatasetType' with the actual column name that distinguishes different dataset types
dataset_type_column = 'Dataset'  # Update this to match your dataset
if dataset_type_column not in df_85class.columns:
    raise ValueError(f"Column '{dataset_type_column}' not found in the dataset.")

dataset_order = [
    "noPMA_1176Features",
    "PMA_1176Features",
    "PMA+noPMA_horiz_2352Features",
    "UMAP01_noPMA_100Features",
    "UMAP01_PMA_100Features",
    "UMAP01_Horiz_PMA+noPMA_100Features",
    "FeatRed08_noPMA_482Features",
    "FeatRed08_PMA_482Features",
    "FeatRed08_PMA+noPMA_964Features",
]

dataset_types = df_85class.loc[df_85class['Model']=='CatBoost Classifier'][dataset_type_column].unique()

# Ensure the DataFrame is ordered by the specified datasets
df_85class["Dataset"] = pd.Categorical(df_85class["Dataset"], categories=dataset_order, ordered=True)

assert len(dataset_types) == len(dataset_order)
print( np.array(dataset_types) == np.array(dataset_order))

# Step 3: Define Metrics
# Define the list of metrics that you want to plot
metrics_columns = ['Accuracy', 'F1','Balanced Accuracy']
#'AUC', 'Recall', 'Prec.', 'Kappa', 'MCC',

# Verify that all specified metrics exist in the DataFrame
# missing_metrics = [metric for metric in metrics_columns if metric not in df_85class.columns]
if missing_metrics:
    raise ValueError(f"The following metric columns are missing from the dataset: {missing_metrics}")

# Step 4: Create the Plot
fig = go.Figure()

# Define a list of colors for different dataset types (optional customization)
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

for i, dataset in enumerate(dataset_order):
    # Filter the DataFrame for the current dataset type
    dataset_df = df_85class.loc[df_85class['Model']=='CatBoost Classifier'][df_85class[dataset_type_column] == dataset]
    
    # Handle cases where there might be multiple entries for a dataset type
    # For simplicity, we'll assume there's only one entry per dataset type
    if dataset_df.empty:
        print(f"No data found for dataset type '{dataset}'. Skipping.")
        continue
    elif len(dataset_df) > 1:
        print(f"Multiple entries found for dataset type '{dataset}'. Using the first entry.")
        dataset_df = dataset_df.iloc[[0]]
    else:
        dataset_df = dataset_df.iloc[0]

    # Extract metric values
    metric_values = [dataset_df[metric] for metric in metrics_columns]

    # Add a bar trace for the current dataset type
    fig.add_trace(go.Bar(
        x=metrics_columns,    # Metrics on the x-axis
        y=metric_values,      # Corresponding metric values
        name=str(dataset),    # Name for the legend
        marker_color=custom_colors[i % len(custom_colors)]  # Assign a color
    ))

# Step 5: Customize the Layout
fig.update_layout(
    title="Catboost Classifier Model Metrics for 85 MOA Class Annotations for Different Datasets",
    xaxis=dict(
        title="Metrics",
        tickangle=-45,
        title_font=dict(size=20),
        tickfont=dict(size=18),
    ),
    yaxis=dict(
        title="Value",
        title_font=dict(size=20),
        tickfont=dict(size=18),
    ),
    barmode="group",  # Group bars together by dataset type
    bargap=0.2,  # Gap between bars within a group
    bargroupgap=0.1,  # Gap between different metric groups
    legend=dict(
        title="Dataset Types",
        x=1.02,  # Position the legend closer to the plot area
        y=1,
        traceorder="normal",
        bordercolor="Black",
        borderwidth=1,
        font=dict(size=18),
    ),
    margin=dict(l=50, r=150, t=50, b=150),  # Adjust margins to accommodate the legend
    width=1200,  # Optional: Increase figure width for better spacing
    height=600,  # Optional: Adjust figure height as needed
)

# Optional: Customize the hover template for better interactivity
fig.update_traces(hovertemplate='Metric: %{x}<br>Value: %{y}<extra></extra>')

# Save plot as PNG
png_filename = f'FeatRed_CatBoost_ModelComparison_Metrics_Plot.png'
# fig.write_image(png_filename)

# Save the plot as HTML
html_filename = 'FeatRed_Catboost_ModelComparison_Metrics_Plot.html'
fig.write_html(html_filename)

print(f'Saved plot as {html_filename}')
