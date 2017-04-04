# statistical learning methods  

reading notes of 
[统计学习方法](https://book.douban.com/subject/10590856/) and [A Course in Machine Learning](http://ciml.info/)  


### statistical learning methods  
- methods = model + strategy + algorithm  
    + model = conditional probability distribution + decision function  
    + strategy = [loss function (cost function)](####loss-function-and-cost-function)  
    + algorithm: solve optimization problems  
- supervised learning  
    + classification: identifying to which of a set of categories a new observation belongs, on the basis of a training set of data  
    + tagging problem (structure prediction): 
        * input: observation sequence; output: tag sequence or status sequence  
        * e.g., part of speech tagging  
    + regression: estimate relationship between a dependent variable and one or more independent variables  
        * linear regression and multiple linear regression  
- unsupervised learning  
- semi-supervised learning  
- reinforcement learning  


#### loss function and cost function
- a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event      
    + expected loss:  
    ![expected loss](https://cloud.githubusercontent.com/assets/5633774/24621923/1349a7b6-1858-11e7-842e-e7af7067cdf9.png)  
    + smaller the loss, better the model --> [goal: select model with smallest expected loss](####erm-and-srm)  
- most commonly used methods:  
    + ```0-1``` loss function:  
    ![0-1 loss function](https://cloud.githubusercontent.com/assets/5633774/24621553/bd6ac452-1856-11e7-8ca6-6deb19f70230.png)  
    + quadratic loss function:  
    ![quadratic loss function](https://cloud.githubusercontent.com/assets/5633774/24621638/0aba3a30-1857-11e7-93fb-4c02fb97d5df.png)  
    + absolute loss function:  
    ![absolute loss function](https://cloud.githubusercontent.com/assets/5633774/24621658/1dd892e2-1857-11e7-8765-f967289f7ccc.png)  
    + logarithmic loss function (log-likelihood loss function):  
    ![logarithmic loss function](https://cloud.githubusercontent.com/assets/5633774/24621707/478cf2e0-1857-11e7-8dbd-566f6b75703f.png)  

#### ERM and SRM
- empirical risk minimization (**ERM**)  
    + compute an empirical risk by averaging the loss function on the training set:  
    ![ERM](https://cloud.githubusercontent.com/assets/5633774/24622262/32f4aef2-1859-11e7-9def-8d78711d4ee2.png)  
    + e.g., maximum likelihood estimation (MLE)  
    + disadvantage: over-fitting when sample size is small  
- structural risk minimization (**SRM**)  
    + regularization: balancing the model's complexity against its success at fitting the training data (penalty term)  
    ![SRM](https://cloud.githubusercontent.com/assets/5633774/24622477/0d288936-185a-11e7-9619-de3d5940ea33.png)  
    + e.g., maximum posterior probability estimation (MAP)  
    
    
### training error and test error  
- training error & test error  
![training error](https://cloud.githubusercontent.com/assets/5633774/24626957/eaed9a92-1867-11e7-83ab-d1529faa3d1f.png)  
![test error](https://cloud.githubusercontent.com/assets/5633774/24626979/ffb96992-1867-11e7-8aa7-dd55eb952c6b.png)  
    + note: **L** --> lost function  
- error rate: ```Lost = 0-1 loss function```  
![error rate](https://cloud.githubusercontent.com/assets/5633774/24627073/62a77f12-1868-11e7-9d18-b5a9341b09f4.png)  
    + note: **I** --> indicator function: ```if y=f(x) I=0, o.w. I=1```  
- accuracy:      
![accuracy](https://cloud.githubusercontent.com/assets/5633774/24627177/bc137880-1868-11e7-9fb7-32f00d624d6c.png)  
![error rate+accuracy](https://cloud.githubusercontent.com/assets/5633774/24627200/cc96dc92-1868-11e7-9218-4ef3b4dcb9a5.png)  
    
    
### regularization and cross validation
- regularization: introduce additional information (regularization term) to a general loss function --> in order to to prevent overfitting  
![regularization](https://cloud.githubusercontent.com/assets/5633774/24634703/18990280-1884-11e7-8b3c-0b53087688a5.png)  
    + Occam's razor: a problem-solving principle --> among competing hypotheses, the one with the fewest assumptions should be selected  
- cross validation: a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set    
    

### generalization error 
- measure of how accurately an algorithm is able to predict outcome values for previously unseen data  
![generalization error](https://cloud.githubusercontent.com/assets/5633774/24634944/5b5e76a8-1885-11e7-88c6-530511000db1.png)  
- the performance of a machine learning algorithm is measured by plots of the generalization error values through the learning process (learning curves) --> smaller the generalization error, better the model  
    + generalization error bound: the generalization error is less than some error bound --> smaller the generalization error bound, better the model  

### generative model and discriminative model 
- generative model: learns the joint probability distribution p(x,y)  
    + e.g., Naive Bayes, Hidden Markov Model  
    + converge fast  
- discriminative model: learns the conditional probability distribution p(y|x) --> the probability of y given x
    + e.g., K Nearest Neighbor, Perceptron, Decision Tree, Logistic Regression, Maximum Entropy, Support Vector Machine, Boosting Method, Conditional Random Field  
    + higher accuracy  


### linearly separable data set (linear separability)
- there exists at least one **hyperplane** (line) in the feature space with all points from one class on one side of the line and all points from another class on the other side   


### distance metric
![relation distance](https://cloud.githubusercontent.com/assets/5633774/24673023/ab430aa2-192b-11e7-8312-fc2e56c877ab.png)  
![distance metric](https://cloud.githubusercontent.com/assets/5633774/24672764/bc4f9366-192a-11e7-809b-4df1d588cb0d.png)  
- Euclidean distance: ```p=2```  
![euclidean](https://cloud.githubusercontent.com/assets/5633774/24672801/e468b666-192a-11e7-96e8-ac0c97721f16.png)  
- Manhattan distance: ```p=1```  
![manhattan](https://cloud.githubusercontent.com/assets/5633774/24672833/03c77e02-192b-11e7-95d6-11b07434da70.png)  
- Max distance between coordinates: ```p=∞```  
![max coordinates](https://cloud.githubusercontent.com/assets/5633774/24672997/8e0d60d6-192b-11e7-833f-2a4e1d8ea05d.png)  
------------------------------------------
    
1. [Perceptron](https://github.com/rarezhang/statistical_learning_methods/blob/master/Perceptron.py)  
supervised learning; binary classifiers; only works for linearly separable data; allows for online learning; error driven  
    - feature space:  
    ![perceptron feature space](https://cloud.githubusercontent.com/assets/5633774/24635791/a789c244-188a-11e7-9fc3-6ad9db126e58.png)  
    - output space:  
    ![perceptron output space](https://cloud.githubusercontent.com/assets/5633774/24635806/bde4a824-188a-11e7-828f-c244ccf5cf90.png)  
    - feature space --> output space:  
    ![perceptron](https://cloud.githubusercontent.com/assets/5633774/24635834/e510222a-188a-11e7-9616-b92ac3b3e3d6.png)  
    ![perceptron sign](https://cloud.githubusercontent.com/assets/5633774/24635839/f6e79b18-188a-11e7-9ff7-926659571e08.png)  
    - separating hyperplane: **_w_** and **_b_**  
    ![perceptron](https://cloud.githubusercontent.com/assets/5633774/24635873/280f7242-188b-11e7-933f-6f4f2b548d80.png)  
    - loss function (empirical risk function):  
    ![perceptron loss fun](https://cloud.githubusercontent.com/assets/5633774/24635994/e310438c-188b-11e7-828e-eff91b014927.png)  
    M: misclassified samples  
    - optimization goal  
    ![perceptron optimization goal](https://cloud.githubusercontent.com/assets/5633774/24636056/51310ad6-188c-11e7-9b60-975e3a04b2db.png)  

    


2. [K-Nearest Neighbor]()  
    - feature space:  
    ![knn feature space](https://cloud.githubusercontent.com/assets/5633774/24672596/2f8d7ce0-192a-11e7-956c-6e1bb2ca2f45.png)  
    - output space:  
    ![knn output space](https://cloud.githubusercontent.com/assets/5633774/24672626/41d6bd8a-192a-11e7-873d-b65d2925dad4.png)  
    - feature space --> output space: majority voting rule  
    ![knn](https://cloud.githubusercontent.com/assets/5633774/24672649/571feb8a-192a-11e7-8bc0-ef17b9ca9039.png)  
    note: **I** --> indicator function: ```if y=c I=1, o.w. I=0```  
    - **_k_**:  
        + small k (decrease approximation error; increase estimation error; sensitive to neighbors) --> complicated model --> prone to over-fitting  
        + large k (decrease estimation error; increase approximation error) --> simple model --> prone to under-fitting  
    - k-dimensional tree (kd tree): a space-partitioning data structure for organizing points in a k-dimensional space  
        
        
3. [Naive Bayes]()  
4. [Decision Tree]()  
5. [Logistic Regression]()  
6. [Support Vector Machine]()  
7. [AdaBoost]()  
8. [EM]()  
9. [Hidden Markov Model]()  
10. [Random Field]()  

11. Latent Dirichlet allocation 
12. Monte Carlo


