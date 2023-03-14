import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

class RMT:

    def __init__(self, n=374, t=40):
        self.n = n
        self.t = t


    '''Ensemble construction of a number of N matrices'''
    def ensemble(self, ACC, type, N=1000):
        '''
        We have to ensembles:
        - WOE. Winshart Ortogonal Ensemble (with uncorrelated data)
        - CWOE. Correlated Wishart Ortogonal Ensemble 
        '''
        n = self.n
        t = self.t
        eigval = []

        if type == 'WOE':
            '''
            Case with uncorrelated random data
            '''
            pr = np.zeros((N,n))
            for i in range(N):
                M = self.__woe_ensemble__(n, t)
                val, vec = scipy.linalg.eig(M)
                eigval.append(val)
                for j in range(n):
                    pr[i,j] = 1/(np.power(np.absolute(vec[:,j]),4).sum()) 
        elif type == 'CWOE':
            '''
            Case with the random data is correlated with a correlation value, the average 
            correlation coefficient of an analogue real matrix.
            '''
            N = len(ACC)
            pr = np.zeros((N,n))
            for i in range(N):
                M = self.__cwoe_ensemble__(n, t,ACC[i])
                val, vec = scipy.linalg.eig(M)
                eigval.append(val)
                for j in range(n):
                    pr[i,j] = 1/((np.power(np.absolute(vec[:,j]),4).sum()))

        '''Calculate the mean for the participation ratios and their inverse'''
        mean_pr = np.array(pr).transpose().mean(axis=1)
        mean_ipr = 1/mean_pr 
        '''Participation ratios plots'''
        self.__pr_plots__(mean_pr, mean_ipr)
        '''Probability density distribution of participation ratios'''
        self.__pr_distribution__(pr)
        '''Probability density distribution of eigenvalues'''
        self.__eigvals_distribution__(eigval)


    '''CWOE ensemble construction of a number of particular real matrices'''
    def CWOE_selected(self, ACC, epochs):
        n, t = self.n, self.t
        N = len(ACC)
        pr = np.zeros((N,n))
        '''
        Case with the random data is correlated with a correlation value, the average 
        correlation coefficient of an analogue real matrix.
        '''
        
        for i in range(N):
            M = self.__cwoe_ensemble__(ACC[i], n, t)
            val, vec = scipy.linalg.eig(M)
            eigval.append(val)
            for j in range(n):
                pr[i,j] = 1/((np.power(np.absolute(vec[:,j]),4).sum()))
        
        
        
        '''
        Create N-random matrices with shape (n,t), where N is the number of
        epochs, and get their eigenvalues
        '''
        if self.cwoe_eigval is not None:
            eigval, pr = self.cwoe_eigval, self.cwoe_pr
        else:
            N = len(ACC)
            eigval = []
            pr = np.zeros((N,n))
            for i in range(N):
                M = self.__cwoe_ensemble__(ACC[i], n, t)
                val, vec = scipy.linalg.eig(M)
                '''Eigenvalues'''
                eigval.append(val)
                '''Participation Ratios'''
                for j in range(n):
                    pr[i,j] = 1/((np.power(np.absolute(vec[:,j]),4).sum()))
            '''Save the eigenvalues and pr values'''
            self.cwoe_eigval, self.cwoe_pr = eigval, pr
        '''Calculate the inverse participation ratios'''
        ipr = 1 / pr
        for i in range(len(epochs)):
            '''Plotting'''
            print('Epoch: {}     ACC: {:.4f}'.format(str(epochs[i]+1), ACC[epochs[i]]))
            '''Participation ratios plots per epoch'''
            self.__pr_plots__(pr[epochs[i]], ipr[epochs[i]])

            '''Probability density distribution of particiátion ratios per epoch'''
            self.__pr_distribution__(pr[epochs[i]])
            '''Probability density distribution of eigenvalues per epoch'''
            self.__eigvals_distribution__(eigval[epochs[i]])
        



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

    '''
    Xi squared function, used in the cwoe ensemble:
    - input. 
        ° c. average correlation coefficient
        ° n. dimension of a cuadratic matrix
    - output. matrix of nxn
    '''
    def __xi_squared__(self, c, n):
        xi = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    xi[i,j]=c
                elif i == j:
                    xi[i,j]=1
        return scipy.linalg.sqrtm(xi)

    '''
    CWOE ensemble:
    - input.
        ° c. average correlation coefficient
        ° n. dimension of the rows of a non-cuadratic matrix
        ° t. dimension of the columns of the matrix
    - output. wishart correlated matrix
    '''
    def __cwoe_ensemble__(self, n, t, c):
        Z = np.random.normal(0, 1, (n,t))
        X = self.__xi_squared__(c,n)
        B = np.dot(X,Z)
        '''Wishart correlated matrix'''
        M = (np.dot(B,B.T))/t
        return M

    '''
    WOE ensemble:
    - input.
        ° n. dimension of the rows of a non-cuadratic matrix
        ° t. dimension of the columns of the matrix
    - output. wishart correlated matrix
    '''
    def __woe_ensemble__(self, n, t):
        m = np.random.normal(0,1,(n,t))
        '''Wishart matrix'''
        M = (np.dot(m,m.T))/t
        return M