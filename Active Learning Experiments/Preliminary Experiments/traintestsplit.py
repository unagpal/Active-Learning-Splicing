import numpy as np
import os

num_experiments = 3
n = 265137
size_test = n//10
size_train = n - size_test
all_indices = np.arange(n)

for experiment in range(num_experiments):
	train_indices = np.random.choice(all_indices, size=size_train, replace=False)
	test_indices = [i for i in all_indices if i not in train_indices]
	filepath_train = os.path.expanduser(".") + '/trainindices' + str(experiment+1) + '.npy'
	filepath_test = os.path.expanduser(".") + '/testindices' + str(experiment+1) + '.npy'
	np.save(filepath_train, train_indices)
	np.save(filepath_test, test_indices)