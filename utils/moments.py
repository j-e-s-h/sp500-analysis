import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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