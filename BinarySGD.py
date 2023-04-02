from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import itertools

def displayImage(x_test):
        plt.imshow(
                    x_test.reshape(28, 28), 
                    cmap=plt.cm.binary, 
                    interpolation="nearest"
                    )
        plt.show()

def displayPredict(clf, actual_y, x_test):
        print("Actual value(y_test_0) =", actual_y)
        print("Prediction =", clf.predict([x_test])[0])
                
def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

mnist_raw  = loadmat("mnist-original.mat")

mnist = {
    "data" : mnist_raw["data"].T, 
    "target" : mnist_raw["label"][0]
}
x = mnist["data"]
y = mnist["target"]
# Training, Test

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 0-9
# 5000
# 5
# true, false
predict_number = 5500 # index 5000 == class 5 ? true : false
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_classifier = SGDClassifier()
sgd_classifier.fit(x_train, y_train_5)

y_train_predict = cross_val_predict(sgd_classifier, x_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_predict)
# print(cm)

y_test_predict = sgd_classifier.predict(x_test)

classes = ['Other Number', 'Number 5']
print(classification_report(y_test_5, y_test_predict, target_names=classes))
print("Accuracy Score :", accuracy_score(y_test_5, y_test_predict) * 100)

# plt.figure()
# displayConfusionMatrix(cm)
# plt.show()

# displayPredict(sgd_classifier, y_test_5[predict_number], x_test[predict_number])
# displayImage(x_test[predict_number])

# Cross validation score
# score = cross_val_score(sgd_classifier, x_train, y_train_5, scoring="accuracy", cv=3)
# print(score)
