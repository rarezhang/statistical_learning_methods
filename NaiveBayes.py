"""
Naive Bayes

"""

import numpy as np 
from collections import Counter


class NaiveBayes:
    """
    Naive Bayes 
    """
    
    def __init__(self):
        """
        """
        pass
        
        
    def fit(self, X, Y):
        """
        training 
        :para X: array like, (n, m): n: number of samples; m: number of features
        :para Y: array like, {-1, 1} 
        """
        n = X.shape[0]  # number of samples
        m = X.shape[1]  # number of features
        
        # compute prior
        # u: array, unique elements
        # c: array, the number of times each of the unique values comes up
        u, c = np.unique(Y, return_counts=True)
        count = dict(zip(u, c))  

        self.prior = dict(zip(u, c/n))  # true prior
        # e.g., {-1: 0.4, 1: 0.6}
        

        # compute conditional probability 
        self.conditional = dict()
        for label in u:
            ind = np.where(Y==label)  # index for each label in Y 
            x = X[ind].T
            
            for xx in x:
                xu, xc = np.unique(xx, return_counts=True)
                cond = dict(zip(xu, xc/count[label]))
                if label in self.conditional:
                    self.conditional[label].update(cond)
                else:
                    self.conditional[label] = cond
        # e.g., 
        # {-1: {'1': 0.5, '2': 0.3, '3': 0.16, 'L': 0.16, 'M': 0.33, 'S': 0.5}, 
        # 1: {'1': 0.22, '2': 0.33, '3': 0.44, 'L': 0.44, 'M': 0.44, 'S': 0.11}}
          

    def predict(self, XX):
        """
        prediction 
        """
        # compute posterior
        self.posterior = {}
        for label in self.prior:
            pos = self.prior[label]
            for f in XX:
                pos *= self.conditional[label][f]
            self.posterior[label] = pos

        # sort posterior 
        # Getting key with maximum value in dictionary
        return max(self.posterior, key=self.posterior.get)
    
if __name__ == "__main__":

    X = np.array([
    ['1', 'S'], # -1
    ['1', 'M'], # -1
    ['1', 'M'], # 1
    ['1', 'S'], # 1
    ['1', 'S'], # -1
    ['2', 'S'], # -1
    ['2', 'M'], # -1
    ['2', 'M'], # 1
    ['2', 'L'], # 1
    ['2', 'L'], # 1
    ['3', 'L'], # 1
    ['3', 'M'], # 1
    ['3', 'M'], # 1
    ['3', 'L'], # 1
    ['3', 'L'] ]) # -1
    
    Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    
    nb = NaiveBayes()
    nb.fit(X, Y)
    
    XX = ['2', 'S']
    y_pred = nb.predict(XX)
    print(f'data points: {XX}  prediction: {y_pred}')