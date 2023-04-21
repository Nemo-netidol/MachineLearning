import seaborn as sns
import matplotlib.pyplot as plt

iris_dataset = sns.load_dataset('iris')

sns.set() # sns.set() will load seaborn's default theme and color palette to the session.
sns.pairplot(iris_dataset, hue='species', height=2) # plot pairwise relationships(ความสัมพันธ์แบบคู่) between variables within a dataset.
plt.show()