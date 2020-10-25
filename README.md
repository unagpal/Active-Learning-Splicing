#**Active Learning in CNNs via Expected Improvement Maximization**

All code used for generating the results reported in the paper is included in this GitHub repository.

#**Obtaining Results**

All code used for acquisition function comparison tables (results shown in: Tables 1 and 2) and graphs of model performance as a function of number of acquired points (results shown in: Figure 2, Figure 3, and Figure B.1) can be found in the acquisition function comparison subfolder of the active learning results folder for the corresponding dataset.
All code used for hypothesis tests comparing the active learning performance of EI to an alternative active learning algorithm using linear mixed models (LMMs) can be found in the AL_Performance_Hypothesis_Tests folder.

#**Libraries/Dependencies**

The libraries used include Keras (CNN models), Matplotlib (charts), Numpy, and Gurobi (robust k-Center MIP). For the R code used in running hypothesis tests to compare active learning performance using LMMs, libraries used include lme4 (for LMMs) and ggplot (for plots).

#**Hyperparameter Tuning**

Notes on methodology for hyperparameter tuning can be found in Appendix E. The hyperparameters used in executing active learning experiments can be found in the corresponding active learning code.

#**Notes**

- Datasets for MNIST and UTKFace are omitted here in order to reduce repository file size. Links are provided in the References section of the paper.

- Running code with a different directory setup may produce errors due to file path discrepancies. Changing file paths for the dataset and location of saved results may fix such issues.

- Keras CNN architectures for MNIST classification are adapted from code used in *Deep Bayesian Active Learning with image data*: https://github.com/Riashat/Deep-Bayesian-Active-Learning

- Robust k-Center mixed integer programming is done using [Gurobi](https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/refman.pdf) and 5 iterations are used for binary search over robust k-Center cost.
