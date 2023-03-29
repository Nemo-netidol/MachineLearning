from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'], # x : Feature(หน้าตา, คุณลักษณะ)
    iris_dataset['target'], # y : Labels/Outcome (ผลลัพธ์)
    random_state = 0, 
    test_size = 0.2
)
# Attribute "test_size" ถ้าไม่ set อะไรไว้จะมี default เป็น 25%
# Default proportion : training data 75% testing data 25%

# ถ้ามี X เป็นแบบนี้จะมี outcome y เป็นแบบนี้
    
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)