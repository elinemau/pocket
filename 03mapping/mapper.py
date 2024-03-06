import pandas as pd
from mycolorpy import colorlist as mcp
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import sys

#USAGE:
#python3 mapper.py path/descriptor_csv_files 

#read in all descriptor csv files, give descriptor directory 
desc_dir = sys.argv[1]

databases=[]
combo = pd.DataFrame()
for filename in os.listdir(desc_dir):
    f = os.path.join(desc_dir, filename)
    filename, file_extension = os.path.splitext(filename)
    databases.append(filename)

    csv = pd.read_csv(f)
    csv.set_index("protein_code", inplace=True)
    csv = csv.dropna()
    csv = csv.drop(csv.columns[:1], axis=1)
    csv.index = csv.index + "_" + filename
    csv_columns = list(csv.columns)

    if combo.empty:
        combo = csv
        combo_columns = csv_columns
    else: 
        #deal with variable columns by removing non- common columns
        if sorted(combo_columns) == sorted(csv_columns):
            combo = pd.concat([combo, csv], axis=0)
        elif len(combo_columns) <= len(csv_columns):
            csv = csv[combo_columns]
            combo = pd.concat([combo, csv], axis=0)
        else:
            combo = combo[csv_columns]
            combo = pd.concat([combo, csv], axis=0)

threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (combo == 0).sum() / len(combo)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
combo = combo.drop(columns=columns_to_drop)
#select relevant features for UMAP training
features = combo.columns

# Convert the DataFrame to a NumPy array
data_array = combo[features].values
data_array = pd.DataFrame(data_array).dropna().values
# Normalize the data (optional but often recommended)
data_array_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))

#create UMAP model
umap_model = umap.UMAP(n_components=2)
umap_result = umap_model.fit_transform(data_array_normalized)

#adding dataset column for color assignment of the points
for db in databases:
    combo.loc[combo.index.str.contains(db), 'database'] = db
    
#save the combo csv file
combo.to_csv('plotting_file.csv')

#setting a color list
colors = mcp.gen_color(cmap='winter', n=len(databases))
color_dict = {db: colors[i] for i, db in enumerate(databases)}
color_list = [color_dict[db] for db in combo['database']]

#plotting
plt.figure(figsize=(25, 25))
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=color_list, s=3)

legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[db], markersize=10, label=db) for db in databases]
plt.legend(handles=legend_labels, loc='upper right')

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP.svg', format='svg', transparent=True)
plt.show()





