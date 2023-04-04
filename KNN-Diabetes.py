import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

x = df.drop("Outcome", axis=1).values
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

k_neighbors = np.arange(1, 9)
train_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
# print(classification_report(y_test, y_predict))
print(pd.crosstab(y_test, y_predict, rownames=["Actual"], colnames=["Predict"], margins=True))


# หาค่า k ที่เหมาะที่สุด

# for i, K in enumerate(k_neighbors):
#     # ปั้นโมเดลใหม่ทุกๆค่า K
#     knn = KNeighborsClassifier(n_neighbors=K)
#     knn.fit(x_train, y_train)
#     train_score[i] = knn.score(x_train, y_train) # Return the mean accuracy on the given test data and labels.
#     test_score[i] = knn.score(x_test, y_test) # Return the mean accuracy on the given test data and labels.

# plt.title("Compare accuracy score of k value from every models")
# plt.plot(k_neighbors, test_score, label="Test score")
# plt.plot(k_neighbors, train_score, label="Train score")
# plt.xlabel("K value")
# plt.ylabel("Accuracy score")
# plt.show()
