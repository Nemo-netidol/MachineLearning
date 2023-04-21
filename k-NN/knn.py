from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0, test_size=0.4)
knn = KNeighborsClassifier(n_neighbors=5)

# Training
knn.fit(x_train, y_train)

# Prediction
y_predict = knn.predict(x_test)
print(classification_report(y_test, y_predict, target_names=dataset['target_names']))
print(f'Accracy : {accuracy_score(y_test, y_predict)*100}%')