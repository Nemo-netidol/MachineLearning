import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100) # A 1 dimension array starts from -5 to 5 with 100 elements 
y = 2*x+1
# X คือตัวแปรอิสระ เรารู้ค่า
# Y คือตัวแปรตาม เราไม่รู้ค่าแต่หาค่าได้จากค่า X โดยใช้สมการ 

rn = np.random
x = rn.rand(50) * 10 # random only positive number
y = 2*x + rn.randn(50) # randn random both positive and negative number
# y = mx + c
# plt.scatter(x, y, '-g', label='y = 2*x+1')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
# plt.legend(loc='upper left')
# plt.title("y=2x+1")
# plt.grid()
plt.show()