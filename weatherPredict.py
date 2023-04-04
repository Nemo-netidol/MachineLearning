import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")

# Training & Testing dataset
x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
# 80% for training & 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the algorithm
model = LinearRegression()
model.fit(x_train, y_train)

# Test
y_predict = model.predict(x_test)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_predict, color="red", linewidth=2) # กราฟ linear regression ของโมเดลของเรา
# plt.show()

# Compare training data and predicted data
df = pd.DataFrame({'Actual_data':y_test.flatten(), 'Predicted_data':y_predict.flatten()})

df1 = df.head(20)

df1.plot(kind='bar', figsize=(16, 10))
plt.show()
# figsize : a tuple of the width and height of the figure in inches

# กราฟของข้อมูลทั้งหมด
# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title("Min & Max Temp")
# plt.xlabel("MinTemp")
# plt.ylabel("MaxTemp")
# plt.show()

#Efficiency Measurement (Loss Function)
print(f'MAE = {metrics.mean_absolute_error(y_test, y_predict)}')
print(f'MSE = {metrics.mean_squared_error(y_test, y_predict)}')
print(f'RMSE = {np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')

# R squared (a statistical measure of how well the regression line approximates the actual data.)
print(f'R squared = {metrics.r2_score(y_test, y_predict) * 100:.2f}%')