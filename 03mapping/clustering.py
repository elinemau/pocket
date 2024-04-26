import os
import sys

from process_csv_files import process_csv_files

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

desc_dir = sys.argv[1]
structures = process_csv_files(desc_dir)

###############################################################################################
# Exploratory visualization of the high dimensional data
###############################################################################################

#visualization with t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(structures)
