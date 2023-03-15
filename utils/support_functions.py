import numpy as np

'''
In a matrix, normalize the rows
'''
def row_normalization(M):
    dims = M.shape
    A = np.zeros(dims)
    for i in range(dims[0]):
         A[i] = (M[i]-np.mean(M[i]))/np.std(M[i])
    return A    
        
'''
In a matrix, calculate the returns in terms of the rows (time series)
'''
def row_returns(M):
    N = M.T
    dims = (len(N)-1,len(N[0]))
    returns = np.zeros(dims)
    for i in range(dims[0]):
        returns[i] = (N[i+1]-N[i])/N[i]
    return returns.T


'''
Participation Ratios:
 - input. tensor of k-matrices with N-eigenvectors arrays each
 - output. matrix of k-arrays with the respective participation ratios
'''
def participation_ratios(tensor):
    dim1 = tensor.shape[0]
    dim2 = tensor.shape[1]
    pr = np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            c = tensor[i]
            C = c[:,j]
            ipr = np.power(np.absolute(C),4).sum()
            pr[i][j] = 1/ipr
    return pr

'''Participation ratios of a single matrix'''
def participation_ratios_one(M):
    pr = np.zeros((M.shape[0]))
    for j in range(M.shape[0]):
        C_i = M[:,j]
        ipr = np.power(np.absolute(C_i),4).sum()
        pr[j] = 1/ipr
    return pr


'''
Distance matrix based on a particular metric:
 * Similarity metric. A personalized metric 
 * L2 metric. The well-known Euclidean metric

 -input. tensor with k-correlation matrices of shape NxN
 -output. cuadratic matrix of shape kxk
'''
def distance_matrix(X, Y, metric):
    dims = X.shape
    diff = np.zeros(dims[0]*dims[-1])
    k = 0
    
    if metric == 'similarity' or metric == 'sim':
        '''Similarity Metric'''
        for i in range(len(X)):
            for j in range(len(X)):
                diff[k] = np.abs(X[i][j] - Y[i][j])
                k += 1
        avg = np.sum(diff)/ (dims[0]*dims[-1])
        return avg
    elif metric == 'euclidean' or metric == 'l2': 
        '''Euclidean Metric'''
        for i in range(len(X)):
            for j in range(len(X)):
                diff[k] = np.abs(X[i][j] - Y[i][j])
                k += 1
        l2 = np.linalg.norm(diff)
        return l2