import matplotlib.pyplot as plt
from sklearn import datasets
digit_dataset = datasets.load_digits() 

print(digit_dataset.target[0])
plt.imshow(digit_dataset.images[0], cmap=plt.get_cmap('gray'))
plt.show()