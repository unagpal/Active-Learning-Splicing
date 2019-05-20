Imports
from __future__ import absolute_import
from __future__ import print_function
from __future__ import print_function
from __future__ import absolute_import
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras import backend as K
#from tensorflow.keras import backend as k
#from tensorflow.python.keras import backend as k
from keras.layers.core import Lambda
from six.moves import range
import numpy as np
import scipy as sp
#from keras import backend as K
import random
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2
import math
import numpy as np
import sys
import random
import warnings
import pprint
from six.moves import range
import six
import time
import os
import threading
from tensorflow.python.framework import ops
try:
    import queue
except ImportError:
    import Queue as queue
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation, Flatten

#Importing 5' Data
# Imports and functions from 5' Model
#%matplotlib inline
from matplotlib.colors import LogNorm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,Input, Flatten
from keras import models
from keras import optimizers
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import numpy as np
import time
import sys
import six

nuc_arr = ['A','C','G','T']
#Function for calculating modified probability of splicing at SD1
def prob_SD1 (sd1_freq, sd2_freq):
    if (sd1_freq==0 and sd2_freq==0):
        return 0.0
    else:
        return sd1_freq/(sd1_freq+sd2_freq)
#Function converting nucleotide sequence to numerical array with 4 channels
def seq_to_arr (seq):
    seq_len = len(seq)
    arr_rep = np.zeros((seq_len, len(nuc_arr)))
    for i in range(seq_len):
        arr_rep[i][nuc_arr.index(seq[i])] = 1
    return arr_rep

#Creating a modified dataset with only the necessary information
#Storing model inputs and outputs
reads_path = 'GSM1911086_A5SS_spliced_reads.txt'
seq_path = 'GSM1911085_A5SS_seq.txt'
s1_indx = 1
s2_indx= 45
seq_len = 101
read_lines = []
seq_lines = []
data_table = []
with open(reads_path) as f:
    f.readline()
    for line in f:
        mod_line = line.split('\t')
        read_lines.append([mod_line[0], mod_line[s1_indx], mod_line[s2_indx]])
with open(seq_path) as f:
    f.readline()
    for line in f:
        mod_line = line.split('\t')
        seq_lines.append([mod_line[0], mod_line[1][:-1]])

n = len(read_lines)
prob_s1 = np.zeros(n)
inputs = np.zeros((n,seq_len, 4))

with open('5SS_compressed.txt', 'w') as f:
    for i in range(n):
        data_table.append([read_lines[i][0], seq_lines[i][1], read_lines[i][1], read_lines[i][2]])
        f.write(read_lines[i][0]+'\t'+seq_lines[i][1]+'\t'+read_lines[i][1]+ '\t'+read_lines[i][2] + '\n')
        prob_s1[i] = prob_SD1(float(read_lines[i][1]), float(read_lines[i][2]))
        inputs[i] = seq_to_arr(seq_lines[i][1])

#Progbar class
class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)

#Model Functionality Imports
#from . import backend as K
#from . import optimizers
#from . import objectives
#from . import callbacks as cbks
#from .utils.layer_utils import container_from_config
#from .utils.layer_utils import model_summary
#from .utils.generic_utils import Progbar
#from .layers import containers

def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]

def slice_X(X, start=None, stop=None):
    '''
    '''
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]

def permanent_dropout_layer(rate, seed):
    return Lambda(lambda x: K.dropout(x,level=rate,seed=seed))

def predict_stochastic(model, X, batch_size=128, verbose=0):
    '''Generate output predictions for the input samples
    batch by batch.
    # Arguments
        X: the input data, as a numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.f
    # Returns
        A numpy array of predictions.
    '''
    X = standardize_X(X)
    pred_Y = model.predict(X,batch_size=batch_size, verbose=verbose)
    return np.array(pred_Y)

def predict_loop(model, ins, batch_size=128, verbose=0):
    '''Abstract method to loop over some data in batches.
    '''
    nb_sample = len(ins[0])
    outs = []
    if verbose == 1:
        progbar = Progbar(target=nb_sample)
    #batches = make_batches(nb_sample, batch_size)
    #print('MAKEBATCH: ' + str(nb_sample))
    batch = make_batches(nb_sample,batch_size)[0]
    index_array = np.arange(nb_sample)
    batch_ids = index_array[batch[0]:batch[1]]
    ins_batch = slice_X(ins, batch_ids)
    batch_outs = model.predict(ins_batch,batch_size=batch_size, verbose=verbose)
    if type(batch_outs) != list:
        batch_outs = [batch_outs]
    for batch_out in batch_outs:
        shape = (nb_sample,) + batch_out.shape[1:]
        outs.append(np.zeros(shape))
    for batch_out in batch_outs:
        outs[batch[0]:batch[1]] = batch_out
    if verbose == 1:
        progbar.update(batch[1])
    #print(np.array(batch_outs).shape)
    return outs

print('Using Dropout Probability = 0.05 and Linearity = ReLU')

np.random.seed(1)

Queries = 1
acquisition_iterations = 100

all_rmse_rand = 0
all_rmse_bald = 0
batch_size = 256
dropout_iterations = 100
Experiments = 2
Experiments_All_MSE_bald = []
Experiments_All_MSE_rand = []
X = np.array(inputs)
y = np.array(prob_s1)
for e in range(Experiments):
    print('Experiment Number ' + str(e+1))
    all_train_indices = np.load(os.path.expanduser('~/ActiveLearningExperiments')+'/trainindices'+str(e+1)+'.npy')
    size_train = 1000
    pool_size = 5000
    train_data_indices = all_train_indices[0:size_train].tolist()
    test_indices = np.load(os.path.expanduser('~/ActiveLearningExperiments') + '/testindices' + str(e+1) + '.npy')
    pool_indices_all = [i for i in range(265137) if i not in train_data_indices and i not in test_indices]
    pool_indices = np.random.choice(np.array(pool_indices_all), pool_size, replace=False)
    X_train = X[train_data_indices]
    y_train = y[train_data_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    #index_train = permutation[ 0 : size_train ]
    #index_test = permutation[  size_train:size_test+size_train ]
    #index_pool = permutation[ size_test+size_train : ]
    #X_train = X[ index_train, :  ]
    #y_train = y[ index_train   ]
    #X_test = X[ index_test, :  ]
    #y_test = y[ index_test ]

    X_pool = X[pool_indices]
    y_pool = y[pool_indices]
    #X_pool = X[ index_pool, :  ]
    #y_pool = y[ index_pool ]
    print('Points in the Pool Set', pool_size)
    #normalise the dataset - X_train, y_train and X_test
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0  ] = 1
    mean_X_train = np.mean(X_train, 0)
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    std_X_pool = np.std(X_pool, 0)
    std_X_pool[ std_X_pool==0  ] = 1
    mean_X_pool = np.mean(X_pool, 0)
    mean_y_pool = np.mean(y_pool)
    std_y_pool = np.std(y_pool)

    #X_train = (X_train - mean_X_train) / std_X_train
    y_train_normalized = (y_train - mean_y_train ) / std_y_train
    X_train_bald = X_train
    y_train_bald = y_train
    X_train_rand = X_train
    y_train_rand = y_train
    #X_pool = (X_pool - mean_X_pool) / std_X_pool
    y_pool_normalised = (y_pool - mean_y_pool ) / std_y_pool
    X_pool_bald = X_pool
    y_pool_bald = y_pool
    X_pool_rand = X_pool
    y_pool_rand = y_pool

    #X_test = (X_test - mean_X_train) / std_X_train

    tau = 0.159708
    N = X_train.shape[0]
    dropout = 0.05
    seed = 38
    lengthscale = 1e2
    Weight_Decay = (1 - dropout) / (2. * N * lengthscale**2 * tau)
    model = Sequential()
    model.add(Conv1D(seq_len,(4), strides=1, input_shape=(seq_len,len(nuc_arr)), activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(permanent_dropout_layer(dropout,seed))
    model.add(Conv1D(seq_len//2, (4), strides=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(permanent_dropout_layer(dropout,seed))
    model.add(Dense(50, W_regularizer=l2(Weight_Decay), init='normal',  activation='relu'))
    model.add(permanent_dropout_layer(dropout,seed))
    model.add(Dense(1, W_regularizer=l2(Weight_Decay), init='normal'))
    #model.add(Dense(50, W_regularizer=l2(Weight_Decay), input_dim=404, init='normal', activation='relu'))
    #model.add(permanent_dropout_layer(dropout, seed))
    #model.add(Dense(32, W_regularizer=l2(Weight_Decay), init='normal', activation='relu'))
    #model.add(permanent_dropout_layer(dropout, seed))
    #model.add(Dense(13, W_regularizer=l2(Weight_Decay), init='normal', activation='relu'))
    #model.add(permanent_dropout_layer(dropout, seed))
    #model.add(Dense(1, W_regularizer=l2(Weight_Decay), init='normal'))


    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=0)
    All_Dropout_Scores = np.zeros(shape=(X_test.shape[0],1))
    y_predicted = np.zeros(shape=(All_Dropout_Scores.shape[0]))
    for d in range(dropout_iterations):
        dropout_score = predict_stochastic(model, X_test,batch_size=batch_size, verbose=0)
        All_Dropout_Scores = np.append(All_Dropout_Scores,dropout_score,axis=1)
    for j in range(All_Dropout_Scores.shape[0]):
        P_temp = All_Dropout_Scores[j,:]
        P = np.delete(P_temp,0)
        P_mean = np.mean(P)
        y_predicted[j] = P_mean
    mse = np.mean((y_test - y_predicted)**2)
    all_mse_rand = mse
    all_mse_bald = mse

    print('Starting Active Learning Experiments')

    for i in range(acquisition_iterations):

        print('Acquisition Iteration ' + str(i+1))

        #Select next x randomly
        x_pool_index_rand = np.random.randint(X_pool_rand.shape[0], size=1)[0]

        Pooled_X_rand = np.array([X_pool_rand[x_pool_index_rand, :]])
        Pooled_Y_rand = np.array([y_pool_rand[x_pool_index_rand]])
        X_pool_rand = np.delete(X_pool_rand, (x_pool_index_rand), axis=0)
        y_pool_rand = np.delete(y_pool_rand, (x_pool_index_rand), axis=0)
        X_train_rand = np.concatenate((X_train_rand, Pooled_X_rand), axis=0)
        y_train_rand = np.concatenate((y_train_rand, Pooled_Y_rand), axis=0)
        model.fit(X_train_rand, y_train_rand, epochs=50, batch_size=batch_size,verbose=0)
        All_Dropout_Scores = np.zeros(shape=(X_test.shape[0],1))
        Predicted_Mean = np.zeros(shape=(All_Dropout_Scores.shape[0]))
        for d in range(dropout_iterations):
            dropout_score = predict_stochastic(model, X_test, batch_size=batch_size, verbose=0)
            All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)
        for j in range(All_Dropout_Scores.shape[0]):
            P_temp = All_Dropout_Scores[j,:]
            P = np.delete(P_temp,0)
            Predicted_Mean[j] = np.mean(P)
        #Predicted_Mean = model.predict(X_test, batch_size=batch_size)
        mse_rand = np.mean((y_test - Predicted_Mean)**2)
        all_mse_rand = np.append(all_mse_rand, mse_rand)

        print('All MSE Rand:', all_mse_rand)


        All_Dropout_Scores = np.zeros(shape=(X_pool_bald.shape[0], 1))
        model.fit(X_train_bald, y_train_bald, epochs=50, batch_size=batch_size,verbose=0)
        #print('Dropout to compute variance estimates on Pool Set')
        for d in range(dropout_iterations):
            #print('POOLSHAPE: ' + str(np.array(X_pool).shape))
            dropout_score = predict_stochastic(model, X_pool_bald,batch_size=batch_size, verbose=0)
            All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)
        Variance = np.zeros(shape=(All_Dropout_Scores.shape[0]))
        Mean = np.zeros(shape=(All_Dropout_Scores.shape[0]))
        for j in range(All_Dropout_Scores.shape[0]):
            L_temp = All_Dropout_Scores[j, :]
            L = np.delete(L_temp, 0)
            L_var = np.var(L)
            L_mean = np.mean(L)
            Variance[j] = L_var
            Mean[j] = L_mean
        #Select next x with highest predictive variance
        v_sort = Variance.flatten()
        x_pool_index_bald = v_sort.argsort()[-Queries:][::-1]
        Pooled_X_bald = X_pool_bald[x_pool_index_bald, :]
        Pooled_Y_bald = y_pool_bald[x_pool_index_bald]

        X_pool_bald = np.delete(X_pool_bald, (x_pool_index_bald), axis=0)
        y_pool_bald = np.delete(y_pool_bald, (x_pool_index_bald), axis=0)
        X_train_bald = np.concatenate((X_train_bald, Pooled_X_bald), axis=0)
        y_train_bald = np.concatenate((y_train_bald, Pooled_Y_bald), axis=0)
        #print('TRAINSHAPE: ' + str(X_train.shape))
        model.fit(X_train_bald, y_train_bald, epochs=50, batch_size=batch_size,verbose=0)

        Predicted_Dropout = np.zeros(shape=(X_test.shape[0], 1))
        for d in range(dropout_iterations):
            predicted_dropout_scores = predict_stochastic(model, X_test, batch_size=batch_size, verbose=0)
            Predicted_Dropout = np.append(Predicted_Dropout, predicted_dropout_scores, axis=1)

        Predicted_Variance = np.zeros(shape=(Predicted_Dropout.shape[0]))
        Predicted_Mean = np.zeros(shape=(Predicted_Dropout.shape[0]))
        #print('Dropout to Compute Mean Predictions in Regression Task on Test Sete')
        for p in range(Predicted_Dropout.shape[0]):
            P_temp = Predicted_Dropout[p, :]
            P = np.delete(P_temp, 0)
            P_Var = np.var(P)
            P_Mean = np.mean(P)
            Predicted_Mean[p] = P_Mean
        mse_bald = np.mean((y_test - Predicted_Mean)**2)
        all_mse_bald = np.append(all_mse_bald, mse_bald)

        print('MSE BALD at Acquisition Iteration ', mse_bald)

        print('All MSE BALD:', all_mse_bald)

    Experiments_All_MSE_bald.append(all_mse_bald.tolist())
    Experiments_All_MSE_rand.append(all_mse_rand.tolist())
    #filepath = os.path.expanduser("~") + '/Experiment_' + str(e) + '.npy'
    #np.save(filepath, all_rmse)


print("Saving All MSE Over Experiments")

print('MSE BALD: ', Experiments_All_MSE_bald)
print('MSE Rand: ', Experiments_All_MSE_rand)
filepath_bald = os.path.expanduser("~") + '/BaldMSE.npy'
filepath_rand = os.path.expanduser("~") + '/RandMSE.npy'
np.save(filepath_bald, Experiments_All_MSE_bald)
np.save(filepath_rand, Experiments_All_MSE_rand)
