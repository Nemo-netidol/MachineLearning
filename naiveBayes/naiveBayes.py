from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
dataset = load_iris()
x = dataset['data']
y = dataset['target']
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train
model = GaussianNB()
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)

print(f'Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%')
print(" ")
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predict"], margins=True))
