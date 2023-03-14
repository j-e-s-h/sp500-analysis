from support_functions import *

import time
import dcor
import numpy as np
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as shc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score

class Correlation_Analysis:

    def __init__(self, df):
        self.data_matrix = df.drop(columns=['Ticker'], axis=1).to_numpy()
        self.dates = list(df.drop(columns=['Ticker'], axis=1).columns)

    '''Plot a heatmap of the correlation matrix with all the data'''
    def general_correlation_matrix(self, corr_coeff='pearson'):
        if corr_coeff == 'pearson':
            C = self.__pearson_corr__(row_normalization(row_returns(self.data_matrix)))
        elif corr_coeff == 'spearman':
            C, _ = self.__spearman_corr__(row_returns(self.data_matrix))
        elif corr_coeff == 'distance':
            start_time = time.time()
            M = row_returns(self.data_matrix)
            dims = (M.shape[0], M.shape[0])
            C = np.zeros((dims))
            for j in range(dims[0]):
                for k in range(dims[0]):
                    C[j][k] = self.__distance_corr__(M[j], M[k])
                print(j, ": --- %s seconds ---" % (time.time() - start_time))
        else:
            raise ValueError('The only compatible correlation coefficients as a string are: pearson, spearman, distance')
        self.__general_heatmap__(C)
        return C

    '''
    Give the dataframe a format of a numpy array with dimensions of 374x5200, then spilt the data
    in epochs with 40 working days each, a total of 130 matrix with dimension of 374x40. Finally,
    calculate returns and the correlation matrix per each epoch.
    '''
    def split_data_correlation(self, corr_coeff='pearson'):
        data_epochs = np.array_split(row_returns(self.data_matrix), 130, axis=1)
        dims = (data_epochs[0].shape[0], data_epochs[0].shape[0])
        C = np.zeros((len(data_epochs), dims[0], dims[0]))
        if corr_coeff == 'pearson':
            for i in range(len(data_epochs)):
                C[i,:,:] = self.__pearson_corr__(row_normalization(data_epochs[i]))
        elif corr_coeff == 'spearman':
            for i in range(len(data_epochs)):
                C[i,:,:], _ = self.__spearman_corr__(data_epochs[i])
        elif corr_coeff == 'distance':
            start_time = time.time()
            dims = (data_epochs[0].shape[0],data_epochs[0].shape[0])
            C = np.zeros((len(data_epochs),dims[0],dims[0]))
            for i in range(len(data_epochs)):
                for j in range(dims[0]):
                    for k in range(dims[0]):
                        C[i][j][k] = self.__distance_corr__(data_epochs[i][j],
                                                        data_epochs[i][k])        
                print(i, ": --- %s seconds ---" % (time.time() - start_time))
        else:
            raise ValueError(f'The only compatible correlation coefficients are: pearson, spearman, distance')
        return C

    '''
    Split the dataframe in 130 epochs and calculate the returns and correlation matrix, but
    maintain only the preselected epochs
     - output. 
        * correlation matrices of the selected epcohs
        * dates of all the epochs
    '''
    def relevant_correlation_matrices(self, relevant_epochs, corr_coeff='pearson'):
        dates_epochs = np.array_split(self.dates, 130)
        data_epochs = np.array_split(row_returns(self.data_matrix), 130, axis=1)
        selected_epochs = [data_epochs[x] for x in relevant_epochs]
        dims = (selected_epochs[0].shape[0],selected_epochs[0].shape[0])
        C = np.zeros((len(selected_epochs),dims[0],dims[0]))
        if corr_coeff == 'pearson':
            for i in range(len(selected_epochs)):
                C[i,:,:] = self.__pearson_corr__(row_normalization(selected_epochs[i]))
        elif corr_coeff == 'spearman':
            for i in range(len(selected_epochs)):
                C[i,:,:], _ = self.__spearman_corr__(selected_epochs[i])
        elif corr_coeff == 'distance':
            start_time = time.time()
            for i in range(len(selected_epochs)):
                for j in range(dims[0]):
                    for k in range(dims[0]):
                        C[i][j][k] = self.__distance_corr__(selected_epochs[i][j],
                                                        selected_epochs[i][k])        
                print(i, ": --- %s seconds ---" % (time.time() - start_time))
        else: raise ValueError(f'The only compatible correlation coefficients are: pearson, spearman, distance.')
        self.__selected_heatmaps__(C, dates_epochs, relevant_epochs)
        return C, dates_epochs
                        
    def __pearson_corr__(self, M):
        return (np.dot(M,M.T)/len(M[0]))
    
    def __spearman_corr__(self, M):
        return stats.spearmanr(M, axis=1)

    def __distance_corr__(self, X, Y):
        return dcor.distance_correlation(X, Y)

    def __general_heatmap__(self, rho):
        ticks = [41,69,119,174,222,241,268,281,306,354,373]
        ticklabels = ['CD','CS','HC','IN','IT','MA','RE','TS','UT','FI','EN']
        fig, ax = plt.subplots(figsize=(10,7))
        ax = sns.heatmap(rho, cmap='jet', vmin=-1, vmax=1)
        plt.title('01-01-2000 - 09-01-2020')
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))
        plt.show()

    def __selected_heatmaps__(self, rho, epoch_dates, epochs):
        for i in range(len(epochs)):
            print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                            str(epoch_dates[int(epochs[i])][0]),
                                            str(epoch_dates[int(epochs[i])][-1])))
            
            '''Plot'''
            ticks = [41,69,119,174,222,241,268,281,306,354,373]
            ticklabels = ['CD','CS','HC','IN','IT','MA','RE','TS','UT','FI','EN']
            fig, ax = plt.subplots(figsize=(8,6))
            ax = sns.heatmap(rho[i], cmap='jet', vmin=-1, vmax=1)
            ax.set_title('Correlation Matrix Heatmap')
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))
            plt.show()



class Moments:

    def __init__(self, corr_matrices):
        self.corr_matrices = corr_matrices
        self.df = None

    '''
    Create a dataframe with the four first moments of each epoch in the correlation 
    matrices array
    '''
    def get_dataframe(self):
        acc = []
        std = []
        skew = []
        kurt = []
        for i in range(len(self.corr_matrices)):
            moments = self.__moments__(self.corr_matrices[i])
            acc.append(moments[0])
            std.append(moments[1])
            skew.append(moments[2])
            kurt.append(moments[3])
        df = pd.DataFrame(acc, columns=['ACC'])
        df['STD'] = pd.Series(std, index=df.index)
        df['SKEW'] = pd.Series(skew, index=df.index)
        df['KURT'] = pd.Series(kurt, index=df.index)
        self.df = df
        return df, acc, std, skew, kurt

    '''
    Different plots:
     * Average Correlation Coefficient (ACC)
     * Standard Deviation
     * Skewness
     * Kurtosis
     * Standard Deviation vs Average Correlation Coefficient
     * Skewness vs Average Correlation Coefficient
     * Kurtosis vs Average Correlation Coefficient   
    '''
    def get_plot(self, moment1, moment2=None):
        bubbles = [1,44,113]
        crisis = [54,55,56,57,64,65,66,72,73,74,98,126,127]

        if moment1 == 'acc':
            label1 = 'Average Correlation Coefficient'
            array1 = np.array(self.df['ACC'])
            color1 = 'steelblue'
        elif moment1 == 'std':
            label1 = 'Standard Deviation'
            array1 = np.array(self.df['STD'])
            color1 = 'red'
        elif moment1 == 'skew':
            label1 = 'Skewness'
            array1 = np.array(self.df['SKEW'])
            color1 = 'green'
        elif moment1 == 'kurt':
            label1 = 'Kurtosis'
            array1 = np.array(self.df['KURT'])
            color1 = 'purple'
        else: raise ValueError('The compatible values are lists of: acc, std, skew, kurt.')

        if moment2 == None:
            fig, ax = plt.subplots(figsize=(14,8), constrained_layout=True)
            ax.plot(array1, marker='x', linewidth=0, markersize=5, color=color1)
            ax.vlines(x=crisis, color='gray', alpha=0.5, label='Periods with Crises',
                       ymin=np.min(array1), ymax=np.max(array1))
            ax.vlines(x=bubbles, color=color1, alpha=0.3, label='Bubble Peak',
                       ymin=np.min(array1), ymax=np.max(array1), linestyles='solid')
            ax.hlines(y=np.mean(array1), linestyles='dashed', color=color1, xmin=-1, xmax=len(array1)+1,
                       label='Mean: {}'.format(np.around(np.mean(array1), decimals=4)))
            ax.set_title(f'{label1} per Epoch', fontsize=20)
            ax.set_ylabel(label1, fontsize=15)
            ax.set_xticks([0,31,62,94,125])
            ax.set_xticklabels([2000,2005,2010,2015,2020])
            ax.set_xlim([-1, len(array1)+1])
            ax.legend(loc='best', fontsize=10)
            sns.despine(right=True)
            plt.show()
        else:
            if moment2 == 'acc':
                label2 = 'Average Correlation Coefficient'
                array2 = np.array(self.df['ACC'])
                color2 = 'steelblue'
            elif moment2 == 'std':
                label2 = 'Standard Deviation'
                array2 = np.array(self.df['STD'])
                color2 = 'red'
            elif moment2 == 'skew':
                label2 = 'Skewness'
                array2 = np.array(self.df['SKEW'])
                color2 = 'green'
            elif moment2 == 'kurt':
                label2 = 'Kurtosis'
                array2 = np.array(self.df['KURT'])
                color2 = 'purple'
            else: ValueError('The compatible values are lists of: acc, std, skew, kurt')
            fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
            poly = np.polyfit(array2, array1, 2)
            p = np.poly1d(poly)
            y_pred = p(array2)
            t = np.linspace(np.min(array2)-.05, np.max(array2)+.05,200)
            r2 = r2_score(array1, y_pred)
            print('Order 2;   r^2 score: {:.4f}'.format(float(r2)))
            label = f'f(x) = {p[2]:.2f}x^2 + {p[1]:.2f}x * {p[0]:.2f}'
            ax.plot(t, p(t), '--', label=label, color=color2)
            ax.plot(array2, array1, marker='o', linewidth=0, markersize=5, color=color1)
            ax.set_title(f'{label1} vs {label2}', fontsize=20)
            ax.set_ylabel(label1, fontsize=15)
            ax.set_xlabel(label2, fontsize=15)
            plt.legend(fontsize=10)
            sns.despine(right=True)
            plt.show()

    def __matrix_mean__(self, M):
        count = 0
        xsum = 0
        for i in range(len(M)):
            for j in range(i+1, len(M)):
                xsum = xsum + M[i,j]
                count +=1
        return xsum/count

    def __matrix_std__(self, M):
        mean = self.__matrix_mean__(M)
        xsum = 0
        count = 0
        for i in range(len(M)):
            for j in range(i+1, len(M)):
                xsum = xsum + (M[i,j] - mean)**2
                count += 1
        return np.sqrt(xsum/count)
 
    def __moments__(self, M):
        mean = self.__matrix_mean__(M)
        std = self.__matrix_std__(M)
        skew_sum = 0
        kurt_sum = 0
        count = 0
        for i in range(len(M)):
            for j in range(i+1, len(M)):
                skew_sum = skew_sum + (M[i,j] - mean)**3
                kurt_sum = kurt_sum + (M[i,j] - mean)**4
                count += 1
        skew = skew_sum / (count*std**3)
        kurt = kurt_sum / (count*std**4)
        return mean, std, skew, kurt



class Eig:

    def __init__(self, corr_matrices, plot_per_epoch: bool):
        self._split = plot_per_epoch
        self._corr_matrices = corr_matrices
        self._corr_values = []
        self._eigenvalues = []
        self._eigenvectors = []
        '''Save all the participation ratios and correlation values per epoch'''
        for i in range(len(corr_matrices)):
            self._corr_values.append(np.concatenate(corr_matrices[i], axis=0))
            eigval, eigvec = np.linalg.eig(corr_matrices[i])
            self._eigenvalues.append(eigval)
            self._eigenvectors.append(eigvec)
        self._pr = participation_ratios(np.array(self._eigenvectors))

    def plots(self, epochs=None, epoch_dates=None):
        if self._split == True :
            pr = self._pr
            ipr = 1 / pr 
            corr_values = self._corr_values
            eigvals = self._eigenvalues
            for i in range(len(self._corr_matrices)):
                print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                            str(epoch_dates[int(epochs[i])][0]),
                                            str(epoch_dates[int(epochs[i])][-1])))
                '''Participation ratios plots per epoch'''
                self.__pr_plots__(pr[i], ipr[i])
                '''Probability density distribution of correlation coefficients per epoch'''
                self.__correlation_distribution__(corr_values[i])
                '''Probability density distribution of particiátion ratios per epoch'''
                self.__pr_distribution__(pr[i])
                '''Probability density distribution of eigenvalues per epoch'''
                self.__eigvals_distribution__(eigvals[i])

        else:
            '''Calculate the mean for the participation ratios and their inverse'''
            mean_pr = np.array(self._pr).transpose().mean(axis=1)
            mean_ipr = 1/mean_pr
            '''Participation ratios plots'''
            self.__pr_plots__(mean_pr, mean_ipr)
            '''Probability density distribution of correlation coefficients'''
            self.__correlation_distribution__(self._corr_values)
            '''Probability density distribution of participation ratios'''
            self.__pr_distribution__(self._pr)
            '''Probability density distribution of eigenvalues'''
            self.__eigvals_distribution__(self._eigenvalues)

            
    def __pr_plots__(self, pr, ipr):
        fig = plt.figure(figsize=(16,5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(pr, alpha=0.7)
        ax1.hlines(y=np.mean(pr), linestyles='dashed', color='red',
                    xmin=0, xmax=np.max(len(pr)))
        ax1.set_ylabel('PR(λ)')
        ax1.set_xlabel('λ')
        ax1.tick_params(labelbottom=False)
        ax1.set_title('Participation Ratios')
        ax2.plot(ipr, alpha=0.7, color='orange')
        ax2.hlines(y=np.mean(ipr), linestyles='dashed', color='red',
                    xmin=0, xmax=np.max(len(ipr)))
        ax2.set_ylabel('IPR(λ)')
        ax2.set_xlabel('λ')
        ax2.tick_params(labelbottom=False)
        ax2.set_title('Inverse Participation Ratios')
        plt.show()

    def __correlation_distribution__(self, corr_values):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(np.array(corr_values).flatten(), bins=100, range=(-1,0.99), density= True, alpha=0.1, color='gray')
        density = stats.gaussian_kde((np.array(corr_values).flatten()))
        x = np.linspace(-1, 1, 100)
        ax.plot(x, density(x), '--', linewidth=4, color='cornflowerblue')
        ax.vlines(x=np.mean(corr_values), linestyles='dashed', color='red',
           ymin=0, ymax=density(np.mean(corr_values)), label='Mean: {:.4f}'.format(np.mean(corr_values)))
        ax.set_xlim([-1,0.99])
        ax.set_title('Correlation Coefficient Probability Density Function', fontsize=15)
        ax.set_ylabel('ρ$(C_{ij})$', fontsize=12)
        ax.set_xlabel('$C_{ij}$', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        sns.despine(right=True)
        plt.show()
            
    def __pr_distribution__(self, pr):
        fig, ax = plt.subplots(figsize=(16,10))
        ax.hist(pr.flatten(), range=(0,374),
                alpha=0.7, bins=100, label='Participation Ratios')
        plt.title('Participation Ratios Distribution', fontsize=20)
        ax.set_xlabel('PR(λ)', fontsize=20)
        ax.set_ylabel('ρ(PR)', fontsize=20)
        sns.despine(right=True)
        plt.show()

    def __eigvals_distribution__(self, eigvals):
        fig = plt.figure(figsize=(16,10))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.4,0.4,0.55,0.55])
        '''Distribution without zoom'''
        ax1.hist(np.array(np.real(eigvals)).flatten(), density=True, alpha=0.7,
                range=(0,300), bins=150)
        ax1.set_title('Eigenvalues Probability Density Function', fontsize=20)
        ax1.set_xlabel("λ", fontsize=20)
        ax1.set_yscale('log')
        ax1.set_ylabel("ρ(λ)", fontsize=20)
        '''Distribution with zoom'''
        ax2.hist(np.array(np.real(eigvals)).flatten(), density=True, alpha=0.7,
                range=(0,20), bins=150)
        ax2.set_xlabel("λ", fontsize=20)
        ax2.set_yscale('log')
        ax2.set_ylabel("ρ(λ)", fontsize=20)
        sns.despine(right=True)
        plt.show()



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