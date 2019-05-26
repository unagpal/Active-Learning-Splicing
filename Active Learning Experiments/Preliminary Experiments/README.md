**Code Functionality, by File**

* traintestsplit.py stores permutations of all data indices in .npy files in order to standardize the initial dataset across active learning models.

* CNP_MLP_AL.py runs active learning experiments with CNP models using MLP encoders. Acquisition is done: i) randomly and ii) by acquiring the pool point predicted to have maximum variance. 

* CNP_MLP_AL_Comparison.ipynb shows the results of preliminary active learning experiments in 5' splicing prediction using CNPs with MLP encoders.

* Bayesian_AL.ipynb runs active learning experiments on CNNs (using dropout as an approximation for model uncertainty in acquisition functions).
