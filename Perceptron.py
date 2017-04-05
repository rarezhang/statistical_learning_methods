"""
Perceptron 

python implementation of the Perceptron algorithm 
based on [A Course in Machine Learning](http://ciml.info/) 
chapter 04: The Perceptron
"""

import numpy as np 


class Perceptron:
    """
    Perceptron 
    """    
    
    def __init__(self, max_iter):
        """
        :para max_iter: 
        """
        self.max_iter = max_iter
        
    @staticmethod    
    def _sign(a):
        """
        sign function 
        if a >= 0 return 1; if a < 0 return -1 
        :para a: activation
        """
        return 1 if a >= 0 else -1
        
    
        
    def fit(self, X, Y):
        """
        training 
        :para X: array like, (n, m): n: number of samples; m: number of features
        :para Y: array like, {-1, 1} 
        """
        '''
        # use python list 
        assert isinstance(X, list) and isinstance(Y, list)
        n = len(X)  # number of samples
        m = len(X[0])  # number of features
        
        self.w = [0 for _ in range(m)]  # initialize weights
        self.b = 0  # initialize bias 
        
        for i in range(self.max_iter):
            for d in range(n):
                a = sum([ww*xx for ww,xx in zip(self.w, X[d])]) + self.b  # compute activation for one example a = w * X[d] + b
                if Y[d] * a <= 0:  # the sign of the activation and label are different 
                    self.w = [ww + yx for ww, yx in zip(self.w, [Y[d]*xx for xx in X[d]])]  # update weights: self.w += Y[d] * X[d] 
                    self.b += Y[d]
        return self
        '''
        
        # use numpy 
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        
        n = X.shape[0]  # number of samples
        m = X.shape[1]  # number of features
        
        self.w = np.zeros(m)  # initialize weights
        self.b = 0  # initialize bias 
        
        for i in range(self.max_iter):
            for d in range(n):
                a = np.dot(self.w, X[d]) + self.b  # compute activation for one example a = w * X[d] + b
                if Y[d] * a <= 0:  # the sign of the activation and label are different 
                    self.w += Y[d] * X[d]  # update weights: self.w += Y[d] * X[d] 
                    self.b += Y[d]
        return self
        
        
    def predict(self, XX):
        """
        testing or predict 
        """
        a = sum([ww*xx for ww, xx in zip(self.w, XX)]) + self.b # compute activation for the test example: a = w * XX + b 
        return self._sign(a)
        
    def averaged_fit(self, X, Y):
        """
        original perceptron: counts later points more than it counts earlier point        
        to get better generalization --> averaged perceptron 
        maintain a collection of weight vectors and survival times. 
        at test time, predict according to the average weight vector
        
        :para X: array like, (n, m): n: number of samples; m: number of features
        :para Y: array like, {-1, 1} 
        """
        # use numpy 
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        
        n = X.shape[0]  # number of samples
        m = X.shape[1]  # number of features
        
        # initialize weights and bias        
        self.w = np.zeros(m)  # initialize weights
        self.b = 0  # initialize bias 
        
        # initialize cached weights and bias
        self.u = np.zeros(m)  # initialize cached weights
        self.bb = 0  # initialize cached bias
        
        # initialize example counter to 1
        c = 1

        for i in range(self.max_iter):
            for d in range(n):
                a = np.dot(self.w, X[d]) + self.b  # compute activation for one example a = w * X[d] + b
                if Y[d] * a <= 0:  # the sign of the activation and label are different 
                    self.w += Y[d] * X[d]  # update weights: self.w += Y[d] * X[d] 
                    self.b += Y[d]
                    
                    self.u += (Y[d] * c) * X[d] # update cached weights
                    self.bb += Y[d] * c  # update cached bias
                c += 1
        # averaged weights and bias         
        self.w = self.w - (1/c)*self.u
        self.b = self.b - (1/c)*self.bb    
        return self


        
if __name__ == "__main__":
    
    X = np.asarray([[3, 3], [4, 3], [1, 1]])
    Y = np.asarray([1, 1, -1])
    
    perceptron = Perceptron(max_iter=20)
    perceptron.fit(X, Y)
    print(f'perceptron weight: {perceptron.w} and bias: {perceptron.b}')
    y = [perceptron.predict(XX) for XX in X]
    print(f'perceptron prediction: {y}')
        
    perceptron.averaged_fit(X, Y)
    print(f'averaged perceptron weight: {perceptron.w} and bias: {perceptron.b}')
    y = [perceptron.predict(XX) for XX in X]
    print(f'averaged perceptron prediction: {y}')
  
    