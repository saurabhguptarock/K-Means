import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=500, centers=5, random_state=3)
k = 5
color = ['green', 'red', 'blue', 'yellow', 'orange']

clusters = {}
for i in range(k):
    center = 10 * (2 * np.random.random((x.shape[1],)) - 1)
    points = []

    cluster = {
        'center': center,
        'points': points,
        'color': color[i]
    }
    clusters[i] = cluster


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))


def assignPointsToCluster():
    for ix in range(x.shape[0]):
        dist = []
        curr_x = x[ix]
        for kx in range(k):
            d = distance(curr_x, clusters[kx]['center'])
            dist.append(d)
        curr_clustur = np.argmin(dist)
        clusters[curr_clustur]['points'].append(curr_x)


def updateCenter():
    for kx in range(k):
        pts = np.array(clusters[kx]['points'])
    if pts.shape[0] > 0:
        new_u = pts.mean(axis=0)
        clusters[kx]['center'] = new_u
        clusters[kx]['points'] = []


def plotClusters():
    for kx in range(k):
        pts = np.array(clusters[kx]['points'])
        try:
            plt.scatter(pts[:, 0], pts[:, 1], color=clusters[kx]['color'])
        except:
            pass

        uk = clusters[kx]['center']
        plt.scatter(uk[0], uk[1], color='k', marker='*')
        plt.show()


for _ in range(k):
    assignPointsToCluster()
    plotClusters()
    updateCenter()
