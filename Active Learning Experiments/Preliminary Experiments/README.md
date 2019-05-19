**Code Functionality, by File**

* traintestsplit.py stores permutations of all data indices in .npy files in order to standardize the initial dataset across active learning models.

* CNP_MLP_AL.py runs active learning experiments with CNP models using MLP encoders. Acquisition is done: i) randomly and ii) by acquiring the pool point predicted to have maximum variance. 
