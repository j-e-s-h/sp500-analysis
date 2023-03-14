from utils.support_functions import *
from utils.correlation_analysis import Correlation_Analysis

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class Eig(Correlation_Analysis):

    def __init__(self, corr, plot_per_epoch: bool):
        self.corr_matrices = corr.corr_matrices
        self.corr_matrix = corr.corr_matrix
        self.eigenvalues = []
        self._corr_values = []
        self._eigenvectors = []
        self._split = plot_per_epoch
        '''Save all the participation ratios and correlation values per epoch'''
        for i in range(len(self.corr_matrices)):
            self._corr_values.append(np.concatenate(self.corr_matrices[i], axis=0))
            eigval, eigvec = np.linalg.eig(self.corr_matrices[i])
            self.eigenvalues.append(eigval)
            self._eigenvectors.append(eigvec)
        self.pr = participation_ratios(np.array(self._eigenvectors))

    def plots(self, epochs=None, epoch_dates=None):
        if self._split == True :
            pr = self.pr
            ipr = 1 / pr 
            corr_values = self._corr_values
            eigvals = self.eigenvalues
            for i in range(len(self.corr_matrices)):
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
            mean_pr = np.array(self.pr).transpose().mean(axis=1)
            mean_ipr = 1/mean_pr
            '''Participation ratios plots'''
            self.__pr_plots__(mean_pr, mean_ipr)
            '''Probability density distribution of correlation coefficients'''
            self.__correlation_distribution__(self._corr_values)
            '''Probability density distribution of eigenvalues'''
            self.__eigvals_distribution__(self.eigenvalues)
            '''Probability density distribution of participation ratios'''
            self.__pr_distribution__(self.pr)
            '''Temporal evolution of maximum and minimum pr'''
            self.__pr_per_epoch__()
        
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
        fig, ax = plt.subplots(figsize=(16,10))
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
        ax.set_xlabel('IPR(λ)', fontsize=20)
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
        ax2.set_yscale('log')
        sns.despine(right=True)
        plt.show()

    def __pr_per_epoch__(self):
        bubbles = [1,44,113]
        crises = [54,55,56,57,64,65,66,72,73,74,98,126,127]
        five_year_batch = [0,31,62,94,125]
        years = [2000,2005,2010,2015,2020]
        fig, ax = plt.subplots(figsize=(16,10))
        ax.plot(self.pr[:,0], marker='o', linewidth=0, color='darkblue', label='PR of Maximum Eigenvalue')
        ax.plot(self.pr[:,-1], marker='x', linewidth=0, color='dodgerblue', label='PR of Minimum Eigenvalue')
        ax.vlines(x=crises, color='grey', alpha=0.5, ymin=0, ymax=375, label='Periods with Crises')
        ax.vlines(x=bubbles, color='darkblue', alpha=0.3, ymin=0, ymax=375, linestyles='solid', 
                   label='Bubble Peak')
        ax.hlines(y=374/3, color='maroon', linestyles='dashed', xmin=0, xmax=130,label='N / 3')
        ax.set_title('Maximum and MinimumEigenvalue per Epoch', fontsize=20)
        ax.set_xticks(five_year_batch)
        ax.set_xticklabels(years)
        ax.set_ylabel('Participation Ratios', fontsize=20)
        ax.set_ylim(0,375)
        ax.legend(loc='best', fontsize=10)
        sns.despine(right=True)


    def general_plots(self):
        eigvals, eigvecs = np.linalg.eig(self.corr_matrix)
        pr = self.__pr_general_matrix__(eigvecs)
        ipr = 1 / pr
        self.__eig_vs_pr__(eigvals, pr)
        self.__eig_vs_ipr__(eigvals, ipr)

    def __eig_vs_pr__(self, eigvals, pr):
        fig = plt.figure(figsize=(16,10))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.6,0.22,0.38,0.3])
        ax1.plot(np.sort(eigvals)[::-1], pr, marker='x', alpha=0.7)
        ax1.hlines(y=len(eigvals)/3, linestyles='dashed', color='red', xmin=0, xmax=50)
        ax1.set_xlabel("λ", fontsize=20)
        ax1.set_ylabel('PR($u$)', fontsize=20)
        ax1.set_title('Eigenvalues vs Participation Ratios', fontsize=20)
        ax2.plot(np.sort(eigvals)[::-1], pr, marker='x', alpha=0.7)
        ax2.hlines(y=len(eigvals)/3, linestyles='dashed', color='red', xmin=0,
                xmax=np.max(eigvals))
        ax2.set_xlim([0,5])
        ax2.set_ylim([0,150])
        plt.show()

    def __eig_vs_ipr__(self, eigvals, ipr):
        fig = plt.figure(figsize=(16,10))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.4,0.4,0.55,0.5])
        ax1.plot(np.sort(eigvals)[::-1], ipr, marker='x', alpha=0.7)
        ax1.hlines(y=3/len(eigvals), linestyles='dashed', color='red', xmin=0, xmax=np.max(eigvals))
        ax1.set_xlabel("λ", fontsize=20)
        ax1.set_ylabel('IPR($u$)', fontsize=20)
        ax1.set_title('Eigenvalues vs Inverse Participation Ratios', fontsize=20)
        ax2.plot(np.sort(eigvals)[::-1], ipr, marker='x', alpha=0.7)
        ax2.hlines(y=3/len(eigvals), linestyles='dashed', color='red', xmin=0,
                xmax=np.max(eigvals))
        ax2.set_xlim([0,5])
        ax2.set_ylim([0,0.5])
        plt.show()

    def __pr_general_matrix__(self, M):
        pr = np.zeros((M.shape[0]))
        for j in range(M.shape[0]):
            C_i = M[:,j]
            ipr = np.power(np.absolute(C_i),4).sum()
            pr[j] = 1/ipr
        return pr