import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import sys

os.makedirs('figures', exist_ok=True)

# Set the Qt platform plugin environment variable
os.environ['QT_QPA_PLATFORM'] = 'xcb'

#read in all descriptor csv files, give descriptor directory 
desc_dir = sys.argv[1]

d = dict()
databases=[]
combo = pd.DataFrame()
for filename in os.listdir(desc_dir):
    f = os.path.join(desc_dir, filename)
    filename, file_extension = os.path.splitext(filename)
    
    csv = pd.read_csv(f)
    databases.append(filename)
    csv['dataset'] = filename
    d.update({filename: csv.columns})
    csv.set_index("protein_code", inplace=True)
    csv.index = csv.index.astype(str) + filename
    print(filename)
    
    if combo.empty:
        combo = csv
    else: 
        combo = pd.concat([combo, csv], ignore_index=True)

selected_columns = d.get("1433")[1:]
combo = combo[selected_columns]
#df_sc.index = df_sc.index + "_sc"

# combine df
#df = pd.concat([df_sc, df_iri, df_kel], axis=0)
#df_volsite = combo.iloc[:, 1:89]
#df_algorithm = combo.iloc[:, 90:140]

threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (combo == 0).sum() / len(combo)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
combo = combo.drop(columns=columns_to_drop)
# Select the relevant features for training UMAP
dataset_att = combo['dataset']
color_map = {db: cm.get_cmap('viridis')(i) for i, db in enumerate(databases)}
combo['dataset_color'] = dataset_att.map(color_map)

print(combo['dataset_color'])

features = combo.drop(columns=['dataset']).columns


# Convert the DataFrame to a NumPy array
#data_array = combo[features].apply(pd.to_numeric, errors='coerce')
#data_array = data_array.dropna(axis=1)
#data_array = combo[features].apply(pd.to_numeric, errors='coerce').values
#data_array = pd.DataFrame(data_array).dropna().values
#data_array = data_array[:, :-1]
#data_array = data_array[:, 1:]
#print(data_array) 
combo[features] = combo[features].apply(lambda x: (x-x.mean())/ x.std(), axis=0)


# Normalize the data (optional but often recommended)
#data_array_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))

# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model = umap.UMAP(n_components=n_components)
umap_result = umap_model.fit_transform(combo[features])

# Plot the UMAP result
# Create a color map for all unique values in 'databases'

plt.figure(figsize=(25, 25))
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=combo['dataset_color'], s=3)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP.svg', format='svg', transparent=True)
plt.show()


""" # VOLSITE

zero_proportion_volsite = (df_volsite == 0).sum() / len(df_volsite)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop_volsite = zero_proportion_volsite[zero_proportion_volsite > threshold].index
df_volsite = df_volsite.drop(columns=columns_to_drop_volsite)
# Select the relevant features for training UMAP
features_volsite = df_volsite.columns

# Convert the DataFrame to a NumPy array
data_array_volsite = df_volsite[features_volsite].values
data_array_volsite = pd.DataFrame(data_array_volsite).dropna().values
# Normalize the data (optional but often recommended)
data_array_volsite_normalized = (data_array_volsite - data_array_volsite.min(axis=0)) / (data_array_volsite.max(axis=0) - data_array_volsite.min(axis=0))

df_volsite['dataset'] = df_volsite.index.map(lambda x: 'sc' if 'sc' in x else ('iri' if 'iri' in x else 'kel'))
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model_volsite = umap.UMAP(n_components=n_components)
umap_result_volsite = umap_model_volsite.fit_transform(data_array_volsite_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result_volsite[:, 0], umap_result_volsite[:, 1],
            c=df_volsite['dataset'].map({'sc': "orange", 'iri': 'blue', 'kel': 'red'}),
            s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP_volsite.svg', format='svg', transparent=True)
plt.show()

# ALGORITHM

zero_proportion_alg = (df_algorithm == 0).sum() / len(df_algorithm)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop_alg = zero_proportion_alg[zero_proportion_alg > threshold].index
df_algorithm = df_algorithm.drop(columns=columns_to_drop_alg)
# Select the relevant features for training UMAP
features_alg = df_algorithm.columns

# Convert the DataFrame to a NumPy array
data_array_alg = df_algorithm[features_alg].values
data_array_alg = pd.DataFrame(data_array_alg).dropna().values
# Normalize the data (optional but often recommended)
data_array_alg_normalized = (data_array_alg - data_array_alg.min(axis=0)) / (
            data_array_alg.max(axis=0) - data_array_alg.min(axis=0))

df_algorithm['dataset'] = df_algorithm.index.map(lambda x: 'sc' if 'sc' in x else ('iri' if 'iri' in x else 'kel'))
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model_alg = umap.UMAP(n_components=n_components)
umap_result_alg = umap_model_alg.fit_transform(data_array_alg_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result_alg[:, 0], umap_result_alg[:, 1],
            c=df_algorithm['dataset'].map({'sc': "orange", 'iri': 'blue', 'kel': 'red'}),
            s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP_algorithm.svg', format='svg', transparent=True)
plt.show() """
