from utils.support_functions import *

import time
import dcor
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter


class Correlation_Analysis:

    def __init__(self, df, corr_coeff='pearson'):
        self.data_matrix = df.drop(columns=['Ticker'], axis=1).to_numpy()
        self.dates = list(df.drop(columns=['Ticker'], axis=1).columns)
        self.corr_coeff = corr_coeff

    '''Plot a heatmap of the correlation matrix with all the data'''
    def general_correlation_matrix(self):
        if self.corr_coeff == 'pearson':
            C = self.__pearson_corr__(row_normalization(row_returns(self.data_matrix)))
        elif self.corr_coeff == 'spearman':
            C, _ = self.__spearman_corr__(row_returns(self.data_matrix))
        elif self.corr_coeff == 'distance':
            try: 
                C = np.loadtxt('../data/interim/dcc_corr_matrix.txt')
                print('Distance correlation matrix loaded.')
            except: 
                print('There is no distance correlartion matrix saved. \n Calculating...')
                start_time = time.time()
                M = row_returns(self.data_matrix)
                dims = (M.shape[0], M.shape[0])
                C = np.zeros((dims))
                for j in range(dims[0]):
                    for k in range(dims[0]):
                        C[j][k] = self.__distance_corr__(M[j], M[k])
                    print(j, ": --- %s seconds ---" % (time.time() - start_time))
                matrix_reshaped = C.reshape(C.shape[0], -1)
                np.savetxt('dcc_corr_matrix.txt', matrix_reshaped)
        else:
            raise ValueError('The only compatible correlation coefficients as a string are: pearson, spearman, distance')
        self.__general_heatmap__(C)
        self.corr_matrix = C
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
        self.corr_matrices = C
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