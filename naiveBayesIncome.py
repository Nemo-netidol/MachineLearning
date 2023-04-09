import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            dataset[column] = LabelEncoder().fit_transform(dataset[column])
        # print(dataset[column].dtype)
    return dataset

def splitOutcome(dataset):
    features = dataset.drop('income', axis=1)
    outcome = dataset['income'].copy()
    return features, outcome

df = pd.read_csv('adult.csv')
dataset = cleanData(df)

# แยก portion ของ data ในการ train, test
training_set, test_set = train_test_split(dataset, test_size=0.2)

# แยก Attributes กับ outcome
x_train, y_train = splitOutcome(training_set)
x_test, y_test = splitOutcome(test_set)

#Model
model = GaussianNB()
model.fit(x_train, y_train)

predict = model.predict(x_test)

# Accuracy
print("Accuracy :", accuracy_score(y_test, predict))
print(f"Accuracy : {accuracy_score(y_test, predict) * 100:.2f}%")