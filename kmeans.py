import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--path", required=True,
	help="path to the dataset")
args = vars(ap.parse_args())

class kmeans:

    WssScores = []
    finalClusters = []

    # Contructor..
    # selectKmeans function is called after initialization
    
    def __init__(self, dataset, minClusters = 2, maxClusters = 5, maxIterations = 10000, tolerance = 0.01):

        self.tolerance = tolerance
        self.maxIterations = maxIterations
        self.iterations = 0
        self.maxK = maxClusters
        self.minClusters = minClusters
        self.k = minClusters
        self.WssScores = []
        self.finalClusters = []
        self.data = dataset
        self.data = self.data.drop(df.columns[0], axis=1)
        self.data = self.data.fillna(0)
        self.df = self.data.to_numpy()
        self.r, self.c = self.df.shape
        self.kmeans = np.zeros((self.k, self.c))
        self.prevMeans = np.zeros((self.k, self.c))
        self.count = np.ones((self.k, 1))
        self.selectKmeans()

    # Compute if tolerance level is achieved..
    def determineTolerance(self):
        return (np.sum((self.kmeans - self.prevMeans)*(self.kmeans - self.prevMeans)) <= self.tolerance)

    # WSS Score computation to choose the right cluster size..
    def computeWSS(self, c):

        d = []
        Mat = np.array([])
        for i in range(0, self.k):
            M = np.array([list(self.kmeans[i])])
            r = list(filter(lambda x: c[x] == i, range(len(c))))
            Mat = np.repeat(M, repeats = len(r), axis = 0)
            d.append(np.sum(np.sum((self.df[r, :] - Mat)*(self.df[r, :] - Mat), axis = 1)))
        return sum(d)

    # For the current value of k, choose k random means to start with..
    # call clusterPoints()
    
    def selectKmeans(self):

        if self.k <= self.maxK:
            self.iterations = 0
            self.kmeans = np.zeros((self.k, self.c))
            self.prevMeans = np.zeros((self.k, self.c))
            for i in range(0, self.k):
                self.kmeans[i] = self.df[random.randint(0, self.r - 1)]
            self.clusterPoints()
        else:
            print('Job Done')

    # Construct a distance matrix -->> for every cluster, find distance between every point and the cluster mean..
    # Based on the distance, cluster the point to the one with the smallest distance.
    # Recompute the kmeans after storing the current values into prevMeans.

    # Check for termination conditions and update state variables accordingly..
    def clusterPoints(self):

        meansMat = np.array([])
        dm = np.array([])
        for means in self.kmeans:
            M = np.array([list(means)])
            meansMat = np.repeat(M, repeats = self.r, axis = 0)
            dm = np.append(d, np.sqrt(np.sum((self.df - meansMat)*(self.df - meansMat), axis = 1)))
        dm = d.reshape((self.k, self.r))
        c = list(np.argwhere(dm == dm.min(axis = 0))[:, 0])
        self.prevMeans = self.kmeans
        for i in range(0, self.k):
            r = list(filter(lambda x: c[x] == i, range(len(c))))
            self.kmeans[i] = (np.sum(self.df[r, :], axis = 0))/len(r)
        self.iterations += 1
        if self.iterations == self.maxIterations or self.determineTolerance():
            self.WssScores.append(self.computeWSS(c))
            self.finalClusters.append(self.kmeans)
            self.iterations = 0
            self.k += 1
            self.selectKmeans()
        else:
            self.count = np.ones((self.k, 1))
            self.clusterPoints()

    # Plot the Wss Score..
    def plotWssScores(self):

        x = [i for i in range(self.minClusters, self.maxK+1)]
        plt.plot(x, self.WssScores)
        plt.xlabel('Clusters')
        plt.ylabel('WSS Scores')
        plt.title('WSS Plot')
        plt.show()
        
def main():

    df = pd.read_csv(args["path"])
    data = kmeans(df)
    data.plotWssScores()

if __name__ == '__main__':
    main()
    


    
