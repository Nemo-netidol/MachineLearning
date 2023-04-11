from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

x, y = make_blobs(n_features=10, n_samples=100) # Creates a dataset of 10 fewtures 100 samples

pca = PCA(n_components=4)
pca.fit_transform(x)

df = pd.DataFrame({
    'Varience' : pca.explained_variance_ratio_, 
    'Principle Component' : ['PC1', 'PC2', 'PC3', 'PC4']
    })

sns.barplot(x='Principle Component', y='Varience', data=df)
plt.show()