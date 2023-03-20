from utils.support_functions import *
from utils.correlation_analysis import Correlation_Analysis

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class Eig(Correlation_Analysis):

    def __init__(self, corr):
        self.corr_matrices = corr.corr_matrices
        try: self.corr_matrix = corr.corr_matrix
        except: pass
        self.eigenvalues = []
        self.corr_values = []
        self._eigenvectors = []
        '''Save all the participation ratios and correlation values per epoch'''
        for i in range(len(self.corr_matrices)):
            self.corr_values.append(np.concatenate(self.corr_matrices[i], axis=0))
            eigval, eigvec = np.linalg.eig(self.corr_matrices[i])
            self.eigenvalues.append(eigval)
            self._eigenvectors.append(eigvec)
        self.pr = participation_ratios(np.array(self._eigenvectors))

    '''Plots for only the the selected epochs'''
    def plots_per_epoch(self, epochs, epoch_dates=None):
        ipr = 1 / self.pr
        for i in range(len(epochs)):
            print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                        str(epoch_dates[int(epochs[i])][0]), str(epoch_dates[int(epochs[i])][-1])))
            '''Participation ratios plots per epoch'''
            self.pr_plots(self.pr[i], ipr[i])
        for i in range(len(epochs)):
            print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                        str(epoch_dates[int(epochs[i])][0]), str(epoch_dates[int(epochs[i])][-1])))
            '''Probability density distribution of correlation coefficients per epoch'''
            self.correlation_distribution(self.corr_values[i])
        for i in range(len(epochs)):
            print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                        str(epoch_dates[int(epochs[i])][0]), str(epoch_dates[int(epochs[i])][-1])))
            '''Probability density distribution of particiátion ratios per epoch'''
            self.pr_distribution(self.pr[i])
        for i in range(len(epochs)):
            print('Epoch: {}     Date Period: {} - {}'.format(str(epochs[i]+1),
                                        str(epoch_dates[int(epochs[i])][0]), str(epoch_dates[int(epochs[i])][-1])))
            '''Probability density distribution of eigenvalues per epoch'''
            self.eigvals_distribution(self.eigenvalues[i])
    
    '''PR and IPR plots'''
    def pr_plots(self, pr, ipr):
        fig = plt.figure(figsize=(10,4))
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
        fig.tight_layout(w_pad=1)
        plt.show()

    '''Distribution of the accumulated correlation coefficients of all the matrices'''
    def correlation_distribution(self, corr_values):
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
    
    '''Distribution of the accumulated pr values of all the matrices'''
    def pr_distribution(self, pr):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(pr.flatten(), range=(0,374), alpha=0.7, bins=100, label='Participation Ratios')
        plt.title('Participation Ratios Distribution', fontsize=15)
        ax.set_xlabel('PR(λ)', fontsize=12)
        ax.set_ylabel('ρ(PR)', fontsize=12)
        sns.despine(right=True)
        plt.show()

    '''Distribution of the accumulated eigenvalues of all the matrices'''
    def eigvals_distribution(self, eigvals):
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.4,0.4,0.55,0.55])
        '''Distribution without zoom'''
        ax1.hist(np.array(np.real(eigvals)).flatten(), density=True, alpha=0.7,
                range=(0,300), bins=150)
        ax1.set_title('Eigenvalues Probability Density Function', fontsize=15)
        ax1.set_xlabel("λ", fontsize=12)
        ax1.set_yscale('log')
        ax1.set_ylabel("ρ(λ)", fontsize=12)
        '''Distribution with zoom'''
        ax2.hist(np.array(np.real(eigvals)).flatten(), density=True, alpha=0.7,
                range=(0,20), bins=150)
        ax2.set_yscale('log')
        sns.despine(right=True)
        plt.show()

    '''Temporal evolution of the pr values associated with the maximum and minimum eigenvalue'''
    def pr_evolution(self):
        bubbles = [1,44,113]
        crises = [54,55,56,57,64,65,66,72,73,74,98,126,127]
        five_year_batch = [0,31,62,94,125]
        years = [2000,2005,2010,2015,2020]
        fig, ax = plt.subplots(figsize=(14,8), constrained_layout=True)
        ax.plot(self.pr[:,0], marker='o', linewidth=0, color='darkblue', label='PR of Maximum Eigenvalue')
        ax.plot(self.pr[:,-1], marker='x', linewidth=0, color='dodgerblue', label='PR of Minimum Eigenvalue')
        ax.vlines(x=crises, color='grey', alpha=0.5, ymin=0, ymax=375, label='Periods with Crises')
        ax.vlines(x=bubbles, color='darkblue', alpha=0.3, ymin=0, ymax=375, linestyles='solid', 
                   label='Bubble Peak')
        ax.hlines(y=374/3, color='maroon', linestyles='dashed', xmin=0, xmax=130,label='N / 3')
        ax.set_title('Maximum and Minimum Eigenvalue per Epoch', fontsize=20)
        ax.set_xticks(five_year_batch)
        ax.set_xticklabels(years)
        ax.set_ylabel('Participation Ratios', fontsize=20)
        ax.set_ylim(0,375)
        ax.legend(loc='best', fontsize=10)
        sns.despine(right=True)

    '''PR and IPR plots for the general matrix'''
    def general_plots(self):
        eigvals, eigvecs = np.linalg.eig(self.corr_matrix)
        pr = participation_ratios_one(eigvecs)
        ipr = 1 / pr
        self.__eig_vs_pr__(eigvals, pr)
        self.__eig_vs_ipr__(eigvals, ipr)


    def __eig_vs_pr__(self, eigvals, pr):
        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.6,0.22,0.38,0.3])
        ax1.plot(np.sort(eigvals)[::-1], pr, marker='x', alpha=0.7)
        ax1.hlines(y=len(eigvals)/3, linestyles='dashed', color='red', xmin=0, xmax=50)
        ax1.set_xlabel("λ", fontsize=12)
        ax1.set_ylabel('PR($u$)', fontsize=12)
        ax1.set_title('Eigenvalues vs Participation Ratios', fontsize=15)
        ax2.plot(np.sort(eigvals)[::-1], pr, marker='x', alpha=0.7)
        ax2.hlines(y=len(eigvals)/3, linestyles='dashed', color='red', xmin=0,
                xmax=np.max(eigvals))
        ax2.set_xlim([0,5])
        ax2.set_ylim([0,150])
        plt.show()

    def __eig_vs_ipr__(self, eigvals, ipr):
        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_axes([0.1,0.1,0.9,0.9])
        ax2 = fig.add_axes([0.4,0.4,0.55,0.5])
        ax1.plot(np.sort(eigvals)[::-1], ipr, marker='x', alpha=0.7)
        ax1.hlines(y=3/len(eigvals), linestyles='dashed', color='red', xmin=0, xmax=np.max(eigvals))
        ax1.set_xlabel("λ", fontsize=12)
        ax1.set_ylabel('IPR($u$)', fontsize=12)
        ax1.set_title('Eigenvalues vs Inverse Participation Ratios', fontsize=15)
        ax2.plot(np.sort(eigvals)[::-1], ipr, marker='x', alpha=0.7)
        ax2.hlines(y=3/len(eigvals), linestyles='dashed', color='red', xmin=0,
                xmax=np.max(eigvals))
        ax2.set_xlim([0,5])
        ax2.set_ylim([0,0.5])
        plt.show()