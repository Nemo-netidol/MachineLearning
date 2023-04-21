import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = sns.load_dataset('iris')
x = dataset.drop('species', axis=1)
y = dataset['species']

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)

x['PCA1'] = x_pca[:, 0] # length, index
x['PCA2'] = x_pca[:, 1]
x['PCA3'] = x_pca[:, 2]

x_train, x_test, y_train, y_test = train_test_split(x, y)

x_train=x_train.loc[:, ['PCA1','PCA2','PCA3']]
x_test=x_test.loc[:, ['PCA1','PCA2','PCA3']]

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%')