import time
import numpy as np
import scipy.cluster.hierarchy as shc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.cluster import AgglomerativeClustering

class Agglomerative_Clustering:
    
    def __init__(self, corr_matrix, metric: str):
        self._metric = metric
        self.df = None
        try:
            if metric not in ['similarity', 'sim', 'euclidean', 'l2']:
                raise ValueError('Only the euclidean (l2) and similarity (sim) metrics are available as a string.')
            self._distance_matrix = self.__distance_matrix__(corr_matrix, metric)
        except ValueError as ve:
            print(ve)

    '''Plot a dendogram using the "ward" method'''
    def plot_dendogram(self):
        plt.figure(figsize=(16,10))
        plt.title("Dendogram")
        plt.ylabel('Euclidean Distance')
        plt.xlabel('Batches')
        dendogram = shc.dendrogram(shc.linkage(self._distance_matrix, method='ward')) 
        plt.show()

    '''
    Plot a scatter plot with all the clusters
     - Possible output. dataframe with the cluster classification of the ACC values
    '''
    def clustering(self, df_acc, threshold):
        '''Call the aglomerative clustering algorithm from sklearn'''
        cluster = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=threshold)
        cluster.fit(self._distance_matrix)
        '''Create a new dataframe with the clustering labels'''
        df_cluster = df_acc.copy()
        df_cluster['Cluster'] = cluster.labels_+1
        self.df = df_cluster
        '''It is needed to add 1 to have the correct labels'''
        '''Scatter plot with all the clusters being identified'''
        for n in range(1, len(df_cluster['Cluster'].unique())+1):
            plt.scatter(data=df_cluster[df_cluster['Cluster']==n], x='Epoch', y='ACC')
        plt.show()
    
    '''Distance matrix, calling from the support function file'''
    def __distance_matrix__(self, M, metric):
        start_time = time.time()
        distance_m = np.zeros((M.shape[0],M.shape[0]))
        for i in range(M.shape[0]):
            for j in range(M.shape[0]):
                distance_m[i][j] = distance_matrix(M[i], M[j], metric)
            print(i, ": --- %s seconds ---" % (time.time() - start_time))
        return distance_m