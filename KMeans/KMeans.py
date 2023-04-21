from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
new_x, new_y = make_blobs(n_samples=10, centers=4, cluster_std=0.5, random_state=0)

model = KMeans(n_clusters=4)
model.fit(x)
center = model.cluster_centers_
y_pred = model.predict(x)
new_y_pred = model.predict(new_x)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.scatter(new_x[:, 0], new_x[:, 1], c='orange', s=120) # Samples
plt.scatter(center[0, 0], center[0, 1], c='blue', label='Centeroid 1')
plt.scatter(center[1, 0], center[1, 1], c='green', label='Centeroid 2')
plt.scatter(center[2, 0], center[2, 1], c='red', label='Centeroid 3')
plt.scatter(center[3, 0], center[3, 1], c='black', label='Centeroid 4')
plt.legend(frameon=True)
plt.show()