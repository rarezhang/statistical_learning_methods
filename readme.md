# statistical learning methods  

reading notes of [this book](https://book.douban.com/subject/10590856/)  

### statistical learning methods  
- methods = model + strategy + algorithm  
    + model = conditional probability distribution + decision function  
    + strategy = [loss function (cost function)](#### loss function - cost function)  
    + algorithm    
- supervised learning  
- unsupervised learning  
- semi-supervised learning  
- reinforcement learning  


#### loss function - cost function
- a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event      
    + expected loss:  
    ![expected loss](https://cloud.githubusercontent.com/assets/5633774/24621923/1349a7b6-1858-11e7-842e-e7af7067cdf9.png)  
    + smaller the loss, better the model --> [goal: select model with smallest expected loss](#### ERM & SRM)  
- most commonly used methods:  
    + ```0-1``` loss function:  
    ![0-1 loss function](https://cloud.githubusercontent.com/assets/5633774/24621553/bd6ac452-1856-11e7-8ca6-6deb19f70230.png)  
    + quadratic loss function:  
    ![quadratic loss function](https://cloud.githubusercontent.com/assets/5633774/24621638/0aba3a30-1857-11e7-93fb-4c02fb97d5df.png)  
    + absolute loss function:  
    ![absolute loss function](https://cloud.githubusercontent.com/assets/5633774/24621658/1dd892e2-1857-11e7-8765-f967289f7ccc.png)  
    + logarithmic loss function (log-likelihood loss function):  
    ![logarithmic loss function](https://cloud.githubusercontent.com/assets/5633774/24621707/478cf2e0-1857-11e7-8dbd-566f6b75703f.png)  

#### ERM & SRM
- empirical risk minimization (**ERM**)  
    + compute an empirical risk by averaging the loss function on the training set:  
    ![ERM](https://cloud.githubusercontent.com/assets/5633774/24622262/32f4aef2-1859-11e7-9def-8d78711d4ee2.png)  
    + e.g., maximum likelihood estimation (MLE)  
    + disadvantage: over-fitting when sample size is small  
- structural risk minimization (**SRM**)  
    + regularization: balancing the model's complexity against its success at fitting the training data (penalty term)  
    ![SRM](https://cloud.githubusercontent.com/assets/5633774/24622477/0d288936-185a-11e7-9619-de3d5940ea33.png)  
    + e.g., maximum posterior probability estimation (MAP)  
    
    
    
    
1.  
2.  
3.  
4.  
5.  
6.  
7.  
8.  
9.  
10.  
