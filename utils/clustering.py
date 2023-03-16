from utils.support_functions import *
from utils.correlation_analysis import Correlation_Analysis

import time
import numpy as np
import scipy.cluster.hierarchy as shc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class Agglomerative_Clustering(Correlation_Analysis):
    
    def __init__(self,  corr, metric: str):
        self.corr_matrices = corr.corr_matrices
        self._coeff_label = corr._coeff_label
        self._metric = metric
        self.df = None
        try:
            if self._metric not in ['similarity', 'euclidean', 'sim', 'l2']:
                raise ValueError('Only the euclidean (l2) and similarity (sim) metrics are available as a string.')
        except ValueError as ve:
            print(ve)

    def get_distance_matrix(self):
        try:
            if self._metric == 'similarity' or self._metric == 'sim': metric_label = 'similarity'
            elif self._metric == 'euclidean' or self._metric == 'l2': metric_label = 'euclidean'
            D = np.loadtxt(f'../data/processed/{self._coeff_label}/{self._coeff_label}_{metric_label}_matrix.csv',
                            delimiter=',')
            print('Distance matrix loaded')
        except:
            print('There is no distance correlation with the selected metric. \n Calculating...')
            '''Distance matrix, called from the support function file'''
            start_time = time.time()
            D = np.zeros((self.corr_matrices.shape[0],self.corr_matrices.shape[0]))
            for i in range(self.corr_matrices.shape[0]):
                for j in range(self.corr_matrices.shape[0]):
                    D[i][j] = distance_matrix(self.corr_matrices[i], self.corr_matrices[j], self._metric)
                print(i, ": --- %s seconds ---" % (time.time() - start_time))
            np.savetxt(f'../data/processed/{self._coeff_label}/{self._coeff_label}_{metric_label}_matrix.csv',
                        D, delimiter=',')
        self._distance_matrix = D
        '''Plot the distance matrix'''
        five_year_epochs = [0,31,62,94,125]
        years = [2000,2005,2010,2015,2020]    
        fig, ax = plt.subplots(figsize=(10,7))
        ax = sns.heatmap(D, cmap='BuPu',vmin=0,vmax=np.max(D))
        plt.title('Distance Matrix', fontsize=15)
        ax.set_xticks(five_year_epochs)
        ax.set_xticklabels(years)
        ax.set_yticks(five_year_epochs)
        ax.set_yticklabels(years)
        plt.show()

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
        return df_cluster