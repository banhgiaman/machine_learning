from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

img_link = 'KMeans_Clustering/kitten.jpeg'
img = plt.imread(img_link)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original image')

height = img.shape[0]
width = img.shape[1]
img = img.reshape(height*width, 3)

kmeans = KMeans(n_clusters=8, random_state=0).fit(img)
clusters = kmeans.cluster_centers_
labels = kmeans.labels_

compress_img = np.zeros((height, width, 3), dtype='uint8')

for i in range(height):
    for j in range(width):
        compress_img[i][j] = clusters[labels[i*width+j]]

plt.subplot(1, 2, 2)
plt.imshow(compress_img)
plt.title('Image after K-Means (K=8)')
plt.show()
