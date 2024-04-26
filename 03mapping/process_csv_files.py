import pandas as pd
import os
import sys

# USAGE:
# desc_dir = "folder_path_containing_descriptor_csv_files"
# combo_df = process_csv_files(desc_dir)
# print(combo_df)

def process_csv_files(desc_dir):
    databases=[]
    combo = pd.DataFrame()
    for filename in os.listdir(desc_dir):
        f = os.path.join(desc_dir, filename)
        filename, file_extension = os.path.splitext(filename)
        databases.append(filename)
        csv = pd.read_csv(f)
        print(filename)
        print(csv.axes[1])
        if combo.empty:
            combo = csv
        else: 
            combo = combo.merge(csv, how='inner')
        combo.set_index("protein_code", inplace=True)
        csv.index = csv.index.astype(str) + filename

    combo = combo.dropna()
    combo = combo.drop(combo.columns[:1], axis=1)

    threshold = 0.9
    zero_proportion = (combo == 0).sum() / len(combo)
    columns_to_drop = zero_proportion[zero_proportion > threshold].index
    combo = combo.drop(columns=columns_to_drop)
    features = combo.columns

    data_array = combo[features].values
    data_array = pd.DataFrame(data_array).dropna().values

    if data_array.size > 0:
        data_array_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))
    else:
        print('Data array is empty. Possibly wrong csv merging.')
        sys.exit(1)

    for db in databases:
        combo['dataset'] = combo.index.map(lambda x: db if db in x else combo['dataset'])

    return combo

#print the dimensions of the dataframe
print(combo.shape)

#get idea of the variances of the dataframe
#print(combo.describe())