import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rn = np.random
x = rn.rand(50) * 10
y = 2*x + rn.randn(50)

# Linear regression model
model = LinearRegression()
new_x = x.reshape(-1, 1) # ทำให้เป็น array 2 มิติ

# Train model
model.fit(new_x, y)

print(f'{model.score(new_x, y) * 100:.2f}%') # สัมพสิทธิ์การตัดสินใจ 
print(model.intercept_) # ค่า intercept
print(model.coef_)

# Test model
xfit = np.linspace(-1, 11).reshape(-1 ,1)

yfit = model.predict(xfit)

# Analysis model & Result
plt.scatter(x, y)
plt.plot(xfit, yfit) # testing data
plt.show()

