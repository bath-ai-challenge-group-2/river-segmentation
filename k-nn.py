from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kneed import KneeLocator

sum_squared_dist = []
K = range(1,10)
mask = k_masks[1]

'''
point_cloud = []
for x in range(256):
    for y in range(256):
        if mask[x,y] == 1:
            point_cloud.append(np.array([x,y]))
point_cloud = np.array(point_cloud)        
'''


for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(mask)
    sum_squared_dist.append(km.inertia_)
    
kn = KneeLocator(K, sum_squared_dist, curve='convex', direction='decreasing')
print(kn.knee)

km_best = KMeans(n_clusters=2, random_state=0)
km_best = km_best.fit(mask)

labels = km_best.labels_
centers = km_best.cluster_centers_

max_label = np.argmax(np.bincount(labels))
    
new_mask = np.zeros([256,256])
for label, point in zip(labels, point_cloud):
    if label == 0:
        new_mask[point[0],point[1]] = 1
        
plt.imshow(segmented_image)   

plt.scatter(point_cloud[:,0], point_cloud[:,1], c = labels)
new_labels = np.where(labels == max_label, 1, 0)
seg_image = centers[new_labels]