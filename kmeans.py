import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math

class kmeans:

    WssScores = []
    finalClusters = []
    
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
        self.df = self.data.to_numpy()
        self.r, self.c = self.df.shape
        self.kmeans = np.zeros((self.k, self.c))
        self.prevMeans = np.zeros((self.k, self.c))
        self.count = np.ones((self.k, 1))
        self.selectKmeans()

    def determineTolerance(self):
        return (np.sum((self.kmeans - self.prevMeans)*(self.kmeans - self.prevMeans)) <= self.tolerance)

    def computeWSS(self, c):

        d = []
        Mat = np.array([])
        for i in range(0, self.k):
            M = np.array([list(self.kmeans[i])])
            r = list(filter(lambda x: c[x] == i, range(len(c))))
            Mat = np.repeat(M, repeats = len(r), axis = 0)
            d.append(np.sum(np.sum((self.df[r, :] - Mat)*(self.df[r, :] - Mat), axis = 1)))
        return sum(d)
            
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
        
    def clusterPoints(self):

        meansMat = np.array([])
        d = np.array([])
        for means in self.kmeans:
            M = np.array([list(means)])
            meansMat = np.repeat(M, repeats = self.r, axis = 0)
            d = np.append(d, np.sqrt(np.sum((self.df - meansMat)*(self.df - meansMat), axis = 1)))
        d = d.reshape((self.k, self.r))
        c = list(np.argwhere(d == d.min(axis = 0))[:, 0])
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

    def plotWssScores(self):

        x = [i for i in range(self.minClusters, self.maxK+1)]
        plt.plot(x, self.WssScores)
        plt.xlabel('Clusters')
        plt.ylabel('WSS Scores')
        plt.title('WSS Plot')
        plt.show()
        
def main():

    df = pd.read_csv("C:\\Users\\Gautham\\Documents\\Projects\\data\\ccdata\\CC_GENERAL.csv")
    df = df.drop(['CUST_ID'], axis = 1)
    df = df.fillna(0)
    data = kmeans(df)
    data.plotWssScores()

if __name__ == '__main__':
    main()
    


    
