"""
KNN 

python implementation of the KNN algorithm
"""

import numpy as np 
from scipy.spatial import KDTree
from collections import Counter 

class KNN:
    """
    K-Nearest Neighbors
    """
    
    def __init__(self, K):
        """
        :para K
        """
        self.k = K
        
    def _kd_tree(self, X):
        """
        return kd tree
        """
        self.kd_tree = KDTree(X)
        
        
    def knn_fit(self, X, Y):
        """
        training, store distance
        use kd tree 
        :para X: array like
        """
        self._kd_tree(X)
        self.y = Y
        return self 
        
        
    def knn_predict(self, XX):
        """
        prediction 
        """
        # kd_tree.query
        # return i: the locations of the neighbors in kd_tree  
        _, i = self.kd_tree.query(XX, k=self.k)
        # labels of the k-th closest data point
        labels = self.y[i]
        print(labels)
        # majority voting based on labels
        # Counter.most_common(1) return [(label, number)]
        return Counter(labels).most_common(1)[0][0]
        
        
        


if __name__ == "__main__":
    
    X = np.asarray([[2,3], # 1
                    [5,4], # 1
                    [9,6], # -1
                    [4,7], # 1
                    [8,1], # -1
                    [7,2]]) # -1

    Y = np.asarray([1, 1, -1, 1, -1, -1])
    
    knn = KNN(K=3)
    knn.knn_fit(X, Y)
    
    x = [5,4]
    y_pred = knn.knn_predict(x)
    print(f'data point: {x}, prediction: {y_pred}')
