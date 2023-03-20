import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

class RMT:

    def __init__(self, n=374, t=40):
        self.n = n
        self.t = t
        self.cwoe_eigval = None
        self.cwoe_pr = None

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
            if self.cwoe_eigval is not None:
                eigval, pr = self.cwoe_eigval, self.cwoe_pr
            else:
                N = len(ACC)
                pr = np.zeros((N,n))
                for i in range(N):
                    M = self.__cwoe_ensemble__(n, t, ACC[i])
                    val, vec = scipy.linalg.eig(M)
                    eigval.append(val)
                    for j in range(n):
                        pr[i,j] = 1/((np.power(np.absolute(vec[:,j]),4).sum()))
                self.cwoe_eigval, self.cwoe_pr = eigval, pr
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
        eigval = []
        '''
        Create N-random matrices with shape (n,t), where N is the number of
        epochs, and get their eigenvalues
        '''
        if self.cwoe_eigval is not None:
            eigval, pr = self.cwoe_eigval, self.cwoe_pr
        else:
            N = len(ACC)
            pr = np.zeros((N,n))
            for i in range(N):
                M = self.__cwoe_ensemble__(n, t, ACC[i])
                val, vec = scipy.linalg.eig(M)
                eigval.append(val)
                for j in range(n):
                    pr[i,j] = 1/((np.power(np.absolute(vec[:,j]),4).sum()))
            self.cwoe_eigval, self.cwoe_pr = eigval, pr
        ipr = 1 / pr
        for i in range(len(epochs)):
            print('Epoch: {}     ACC: {:.4f}'.format(str(epochs[i]+1), ACC[epochs[i]]))
            '''Participation ratios plots per epoch'''
            self.__pr_plots__(pr[epochs[i]], ipr[epochs[i]])
        for i in range(len(epochs)):
            print('Epoch: {}     ACC: {:.4f}'.format(str(epochs[i]+1), ACC[epochs[i]]))
            '''Probability density distribution of particiátion ratios per epoch'''
            self.__pr_distribution__(pr[epochs[i]])
        for i in range(len(epochs)):
            print('Epoch: {}     ACC: {:.4f}'.format(str(epochs[i]+1), ACC[epochs[i]]))
            '''Probability density distribution of eigenvalues per epoch'''
            self.__eigvals_distribution__(eigval[epochs[i]])
        

    def __pr_plots__(self, pr, ipr):
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
            
    def __pr_distribution__(self, pr):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(pr.flatten(), range=(0,374),
                alpha=0.7, bins=100, label='Participation Ratios')
        plt.title('Participation Ratios Distribution', fontsize=15)
        ax.set_xlabel('PR(λ)', fontsize=12)
        ax.set_ylabel('ρ(PR)', fontsize=12)
        sns.despine(right=True)
        plt.show()

    def __eigvals_distribution__(self, eigvals):
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