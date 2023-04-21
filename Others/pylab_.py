import pylab
from sklearn import datasets
n = int(input())
digit_dataset = datasets.load_digits() 
print(digit_dataset.target[n])
pylab.imshow(digit_dataset.images[n], cmap=pylab.cm.gray_r)
pylab.show()