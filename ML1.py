from sklearn import datasets

iris = datasets.load_iris()

print(iris.keys())
print(iris['filename'])