from scipy.io import loadmat #เอาไว้ใช้อ่านไฟล์ .mat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat") # ไฟล์นี้มี 70000 รูป
 
mnist = {
    "data" : mnist_raw["data"].T, # data ทุกตัว
    "target" : mnist_raw["label"][0] # label ตัวเลข 0 1 2 ... 9
}
x = mnist["data"]
y = mnist["target"] # Label ชื่อของเลข
number = x[10000] # index of data
number_img = number.reshape(28, 28) # a data image as 28x28 pixels

plt.imshow(
    number_img,
    cmap = plt.cm.binary,
    interpolation = "nearest"
)
print(y[10000])
plt.show()