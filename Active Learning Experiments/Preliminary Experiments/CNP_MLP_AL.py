
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import collections
import os
import math

# Imports and functions from 5' Model
#from matplotlib.colors import LogNorm
#from sklearn.model_selection import KFold
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,Input, Flatten
from keras import models
from keras import optimizers
import numpy as np
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

#The NP takes as input a `CNPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class GPCurvesReader(object):

    def __init__(self,
                batch_size,
                acquired_number,
                x_train,
                x_test,
                y_train,
                y_test,
                x_size=404,
                y_size=1,
                l1_scale=0.4,
                sigma_scale=1.0,
                testing=False):

        self._batch_size = batch_size
        self._acquired_number = acquired_number
        self._x_size = x_size
        self._y_size = y_size
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):

        num_total_points = tf.shape(xdata)[1]

        # Expand and take the difference
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        norm = tf.reduce_sum(
            norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * tf.eye(num_total_points)

        return kernel
    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
          A `CNPRegressionDescription` namedtuple.
        """

        #print(num_context)
        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        # When testing, we want context points randomly sampled from training data
        # and target points randomly sampled from testing data
        if self._testing:
            num_testing_points = tf.Tensor.get_shape(self._y_test)[0]
            num_training_points = tf.Tensor.get_shape(self._y_train)[0]
            num_context = num_training_points
            # Select the targets (all testing data)
            indices_test = np.arange(int(num_testing_points))
            target_x_values = [tf.gather(self._x_test, tf.convert_to_tensor(indices_test))]
            target_y_values = [tf.gather(self._y_test, tf.convert_to_tensor(indices_test))]
            target_x = tf.stack(target_x_values)
            target_y_values = tf.stack(target_y_values)
            target_y = target_y_values
            #target_y = tf.expand_dims(target_y_values, axis=2)

            # Select the observations (all training data)
            indices_train = np.arange(int(num_training_points))
            context_x_values = [tf.gather(self._x_train, tf.convert_to_tensor(indices_train))]
            context_y_values = [tf.gather(self._y_train, tf.convert_to_tensor(indices_train))]
            context_x = tf.stack(context_x_values)
            context_y_values = tf.stack(context_y_values)
            context_y = context_y_values
            #context_y = tf.expand_dims(context_y_values, axis=2)


            #context_x = tf.gather(x_values, idx[:int(num_context)], axis=1)
            #context_y = tf.gather(y_values, idx[:int(num_context)], axis=1)
        # During training the number of target points and their x-positions are
        #selected at random
        else:
            n = int(tf.Tensor.get_shape(self._y_train)[0])
            acquired_indices = np.arange(n-self._acquired_number, n)
            #num_context = 100
            #num_target = 150
            #num_context = int(0.48*n)
            #num_target = int(0.64*n)
            num_target = int(0.64*(n-self._acquired_number))
            num_context = int(0.48*(n-self._acquired_number))
            #num_total_points = int(num_context + num_target)
            x_values_context = []
            y_values_context = []
            x_values_target = []
            y_values_target = []
            #num_training_points = tf.Tensor.get_shape(self._y_train)[0]
            #print(num_context)
            #print(num_target)
            #print(num_total_points)
            for batch_num in range(self._batch_size):
                #context_indices = np.random.choice(n, num_context, replace=False)
                #target_indices = np.arange(n)
                target_indices = np.concatenate([np.random.choice(n-self._acquired_number,num_target,replace=False), acquired_indices])
                context_indices = np.concatenate([target_indices[0:num_context], acquired_indices])
                #print('target indices: ' + str(target_indices))
                #print('context indices: ' + str(context_indices))
                #context_indices = np.array([i for i in range(n) if i not in target_indices])
                x_values_context_batch = tf.gather(self._x_train, tf.convert_to_tensor(context_indices))
                y_values_context_batch = tf.gather(self._y_train, tf.convert_to_tensor(context_indices))
                x_values_target_batch = tf.gather(self._x_train, tf.convert_to_tensor(target_indices))
                y_values_target_batch = tf.gather(self._y_train, tf.convert_to_tensor(target_indices))
                x_values_context.append(x_values_context_batch)
                y_values_context.append(y_values_context_batch)
                x_values_target.append(x_values_target_batch)
                y_values_target.append(y_values_target_batch)
            target_x = tf.stack(x_values_target)
            target_y = tf.stack(y_values_target)
            context_x = tf.stack(x_values_context)
            context_y = tf.stack(y_values_context)
            #y_values = tf.expand_dims(y_values, axis=2)
            #print(tf.shape(y_values))
            #x_values = tf.random_uniform(
            #  [self._batch_size, num_total_points, self._x_size], -2, 2)
            # Select the targets which will consist of the context points as well as
            # some new target points
            #target_x = x_values[:, :num_total_points, :]
            #target_y = y_values[:, :num_total_points, :]
            # Select the observations
            #context_x = x_values[:, :int(num_context), :]
            #context_y = y_values[:, :int(num_context), :]

        query = ((context_x, context_y), target_x)
        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=tf.shape(target_x)[1],
            num_context_points=num_context+self._acquired_number)

class DeterministicEncoder(object):
  """The Encoder."""

  def __init__(self, output_sizes):
    """CNP encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
    """
    self._output_sizes = output_sizes

  def __call__(self, context_x, context_y, num_context_points):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of size bs x observations x m_ch. For this 1D regression
          task this corresponds to the x-values.
      context_y: Tensor of size bs x observations x d_ch. For this 1D regression
          task this corresponds to the y-values.
      num_context_points: A tensor containing a single scalar that indicates the
          number of context_points provided in this iteration.

    Returns:
      representation: The encoded representation averaged over all context
          points.
    """
    # Concatenate x and y along the filter axes
    encoder_input = tf.concat([context_x, context_y], axis=-1)
    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = encoder_input.shape.as_list()
    hidden = tf.reshape(encoder_input, (batch_size * int(num_context_points), -1))
    hidden.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))
      # Last layer without a ReLu
      hidden = tf.layers.dense(
        hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))

    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, int(num_context_points), size))
    # Aggregator: take the mean over all points
    representation = tf.reduce_mean(hidden, axis=1)
    return representation

class DeterministicDecoder(object):
  """The Decoder."""

  def __init__(self, output_sizes):
    """CNP decoder.

    Args:
      output_sizes: An iterable containing the output sizes of the decoder MLP
          as defined in `basic.Linear`.
    """
    self._output_sizes = output_sizes

  def __call__(self, representation, target_x, num_total_points):
    """Decodes the individual targets.

    Args:
      representation: The encoded representation of the context
      target_x: The x locations for the target query
      num_total_points: The number of target points.

    Returns:
      dist: A multivariate Gaussian over the target points.
      mu: The mean of the multivariate Gaussian.
      sigma: The standard deviation of the multivariate Gaussian.
    """

    #Adjusting target_x shape such that each x is 1D: needed when Convolutional encoder and not disruptive otherwise

    target_x_shape = target_x.shape.as_list()
    try:
      target_x = tf.reshape(target_x, [target_x_shape[0], target_x_shape[1], target_x_shape[2]*target_x_shape[3]])
    except:
      pass

    # Concatenate the representation and the target_x
    representation = tf.tile(
        tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
    input = tf.concat([representation, target_x], axis=-1)
    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = input.shape.as_list()
    hidden = tf.reshape(input, (batch_size * num_total_points, -1))
    hidden.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for i, size in enumerate(self._output_sizes[:-1]):
        hidden = tf.nn.relu(
            tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

      # Last layer without a ReLu
      hidden = tf.layers.dense(
          hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

    # Bring back into original shape
    hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return dist, mu, sigma

class DeterministicModel(object):
  """The CNP model."""

  def __init__(self, encoder_output_sizes, decoder_output_sizes):
    """Initialises the model.

    Args:
      encoder_output_sizes: An iterable containing the sizes of hidden layers of
          the encoder. The last one is the size of the representation r.
      decoder_output_sizes: An iterable containing the sizes of hidden layers of
          the decoder. The last element should correspond to the dimension of
          the y * 2 (it encodes both mean and variance concatenated)
    """
    self._encoder = DeterministicEncoder(encoder_output_sizes)
    self._decoder = DeterministicDecoder(decoder_output_sizes)

  def __call__(self, query, num_total_points, num_contexts, target_y=None):
    """Returns the predicted mean and variance at the target points.

    Args:
      query: Array containing ((context_x, context_y), target_x) where:
          context_x: Array of shape batch_size x num_context x 1 contains the
              x values of the context points.
          context_y: Array of shape batch_size x num_context x 1 contains the
              y values of the context points.
          target_x: Array of shape batch_size x num_target x 1 contains the
              x values of the target points.
      target_y: The ground truth y values of the target y. An array of
          shape batchsize x num_targets x 1.
      num_total_points: Number of target points.

    Returns:
      log_p: The log_probability of the target_y given the predicted
      distribution.
      mu: The mean of the predicted distribution.
      sigma: The variance of the predicted distribution.
    """

    (context_x, context_y), target_x = query

    # Pass query through the encoder and the decoder
    representation = self._encoder(context_x, context_y, num_contexts)
    dist, mu, sigma = self._decoder(representation, target_x, num_total_points)

    # If we want to calculate the log_prob for training we will make use of the
    # target_y. At test time the target_y is not available so we return None
    if target_y is not None:
      log_p = dist.log_prob(target_y)
    else:
      log_p = None

    return log_p, mu, sigma

#Resetting Graphs
tf.reset_default_graph()

def rand_acq_step(X_train, X_test, y_train, y_test, X_pool, y_pool, model, acq_number, sample_size, num_batches, tr_iterations):
    #Resetting Graphs
    tf.reset_default_graph()

    X_train_lst = np.array([data_point.flatten() for data_point in X_train])
    X_train_1 = tf.constant(X_train_lst)
    X_test_lst = np.array([test_point.flatten() for test_point in X_test])
    X_test_1 = tf.constant(X_test_lst)
    n = 265137
    pool_size = X_pool.shape[0]
    sample_indices = np.random.choice(pool_size, sample_size, replace=False)
    X_pool = X_pool[sample_indices]
    y_pool = y_pool[sample_indices]
    X_pool_1 = tf.constant(X_pool)
    y_pool_1 = tf.constant(y_pool)
    y_train_lst = np.array([[training_y] for training_y in y_train])
    y_train_1 = tf.constant(y_train_lst)
    y_test_1 = tf.convert_to_tensor(value=y_test, dtype=tf.float64)

    #Train and Test Datasets
    dataset_train_5_prime = GPCurvesReader(batch_size=num_batches, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1,
        acquired_number=acq_number)
    data_train = dataset_train_5_prime.generate_curves()
    dataset_test_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1,acquired_number=0, testing=True)
    data_test = dataset_test_5_prime.generate_curves()
    dataset_pool_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_pool_1, y_train=y_train_1, y_test=y_pool_1,acquired_number=0, testing=True)
    data_pool = dataset_pool_5_prime.generate_curves()
    # Define the loss
    log_prob, _, _ = model(data_train.query, data_train.num_total_points,
         data_train.num_context_points, data_train.target_y)

    loss = -tf.reduce_mean(input_tensor=log_prob)

    # Get the predicted mean and variance at the target points for the testing set
    _, mu, sigma = model(data_test.query, data_test.num_total_points,
         data_test.num_context_points)
    _, pool_mu, pool_sigma = model(data_pool.query, data_pool.num_total_points,
         data_test.num_context_points)
    # Set up the optimizer and train step
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    init = tf.compat.v1.initialize_all_variables()
    mse = 100
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        it = 0
        sigma_std = 1
        while (sigma_std > 0.02 and it < tr_iterations):
            sess.run([train_step])
            #if (it % print_after==0):
            pred_y, sigma_pred_pool = sess.run(
                [mu, pool_sigma])
            pred_y = pred_y.flatten()
            sigma_pred_pool = sigma_pred_pool.flatten()
            mse = np.mean((y_test-pred_y)**2)
            sigma_std = np.std(sigma_pred_pool)
            it += 1
        sess.close()

    #Returning MSE
    return mse

def maxvar_acq_step (X_train, X_test, y_train, y_test, X_pool, y_pool, model, acq_number, sample_size, num_batches, tr_iterations):
    #Resetting Graphs
    tf.reset_default_graph()
    X_train_1 = tf.constant(X_train)
    X_test_1 = tf.constant(X_test)
    n = 265137
    pool_size = X_pool.shape[0]
    sample_indices = np.random.choice(pool_size, sample_size, replace=False)
    X_pool = X_pool[sample_indices]
    y_pool = y_pool[sample_indices]
    X_pool_1 = tf.constant(X_pool)
    y_pool_1 = tf.constant(y_pool)
    y_train_lst = np.array([[training_y] for training_y in y_train])
    y_train_1 = tf.constant(y_train_lst)
    y_test_1 = tf.convert_to_tensor(y_test)
    #Train and Test Datasets
    dataset_train_5_prime = GPCurvesReader(batch_size=num_batches, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1,
        acquired_number=acq_number)
    data_train = dataset_train_5_prime.generate_curves()
    dataset_test_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1, acquired_number=0, testing=True)
    data_test = dataset_test_5_prime.generate_curves()
    dataset_pool_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_pool_1, y_train=y_train_1, y_test=y_pool_1, acquired_number=0, testing=True)
    data_pool = dataset_pool_5_prime.generate_curves()
    # Sizes of the layers of the MLPs for the encoder and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    #encoder_output_sizes = encodersizes
    #encoder_output_sizes = [64, 64, 64, 64]
    #decoder_output_sizes = decodersizes
    #decoder_output_sizes = [64, 64, 2]

    # Define the model
    #model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)
    # Define the loss
    log_prob, _, _ = model(data_train.query, data_train.num_total_points,
         data_train.num_context_points, data_train.target_y)

    loss = -tf.reduce_mean(input_tensor=log_prob)

    # Get the predicted mean and variance at the target points for the testing set
    _, mu, sigma = model(data_test.query, data_test.num_total_points,
         data_test.num_context_points)

    _, pool_mu, pool_sigma = model(data_pool.query, data_pool.num_total_points,
         data_test.num_context_points)
    # Set up the optimizer and train step
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    init = tf.compat.v1.initialize_all_variables()
    mse = 100
    max_var_ind = 0
    loss_fn_arr = []
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        it = 0
        sigma_std = 1
        while (sigma_std > 0.02 and it < tr_iterations):
            sess.run([train_step])
            pred_y, sigma_pred_pool = sess.run(
              [mu, pool_sigma])
            sigma_pred_pool = sigma_pred_pool.flatten()
            mse = np.mean((y_test-pred_y.flatten())**2)
            max_var_ind = np.argmax(sigma_pred_pool)
            it += 1
            sigma_std = np.std(sigma_pred_pool)
    #print('elapsed iterations: ' + str(it) + ', MSE: ' + str(mse) + ',loss: ' + str(loss_fn_arr[-1]))
    sess.close()

    #Returning MSE
    return [mse, sample_indices[max_var_ind]]

#TESTING ON REAL DATA
#Imports
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
#import matplotlib.pyplot as plt
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
#from matplotlib.colors import LogNorm
#from sklearn.model_selection import KFold
#import matplotlib.pyplot as plt
from keras.layers import Dense, Conv1D, SpatialDropout1D, MaxPooling1D, MaxPooling2D,Input, Flatten
from keras import models
from keras import optimizers
import numpy as np
import keras
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

dataset_path = os.path.expanduser("~/5SS_compressed.txt")
seq_len = 101
n = 265137
inputs = np.zeros((n,seq_len, 4))
prob_s1 = np.zeros(n)
with open(dataset_path) as f:
    ind = 0
    for line in f:
        mod_line = line.split('\t')
        inputs[ind] = seq_to_arr(mod_line[1])
        prob_s1[ind] = prob_SD1(float(mod_line[2]), float(mod_line[3][:-1]))
        ind += 1

#Experiment Settings
train_size = 100
num_acquisitions = 5000
training_iterations = 300
variance_sample_size = 10000
num_batches = 2

#Getting training and testing sets
X = np.array(inputs)
y = np.array(prob_s1)
X = np.array([pt.flatten() for pt in X])
n = len(X)
all_train_indices = np.load(os.path.expanduser("~/trainindices2.npy"))
train_data_indices = all_train_indices[0:train_size].tolist()
test_indices = np.load(os.path.expanduser("~/testindices2.npy"))
test_data_indices = test_indices.tolist()
X_train = X[train_data_indices]
y_train = y[train_data_indices]
X_test = X[test_data_indices]
y_test = y[test_data_indices]
pool_indices_rand = [i for i in range(n) if i not in train_data_indices and i not in test_indices]
pool_indices_maxvar = [i for i in range(n) if i not in train_data_indices and i not in test_indices]
train_indices_rand = all_train_indices[0:train_size].tolist()
train_indices_maxvar = all_train_indices[0:train_size].tolist()
#Initialize model
encoder_output_sizes = [300,300]
decoder_output_sizes = [300,2]
model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)
#Testing acq_step
#pool_indices = [i for i in range(n) if i not in train_data_indices and i not in test_indices]
X_pool_init = X[pool_indices_rand]
y_pool_init = y[pool_indices_rand]
#print('calling mavar acq step')
#print(maxvar_acq_step(X_train, X_test, y_train, y_test, X_pool, y_pool, model, acq_num, variance_sample_size, num_batches, training_iterations, print_after))
#print('calling rand acq step')
#print(rand_acq_step(X_train, X_test, y_train, y_test, X_pool, y_pool, model, variance_sample_size, num_batches, training_iterations, print_after))

#Active Learning Experiment

filepath_mse_rand = os.path.expanduser("~/CNPMSERandModExp1.npy")
filepath_mse_maxvar = os.path.expanduser("~/CNPMSEMaxVarModExp1.npy")
filepath_dataind_rand = os.path.expanduser("~/CNPDataIndRandModExp1.npy")
filepath_dataind_maxvar = os.path.expanduser("~/CNPDataIndMaxVarModExp1.npy")

exp_mse_rand = []
exp_mse_maxvar = []

#First Acquisition
mse = 1
maxvar_acq_ind = -1
[mse, maxvar_acq_ind] = maxvar_acq_step(X_train, X_test, y_train, y_test, X_pool_init, y_pool_init, model, 0, variance_sample_size, num_batches,
    training_iterations)
exp_mse_rand = [mse]
exp_mse_maxvar = [mse]

#print(exp_mse_rand)
#print(exp_mse_maxvar)
#print(maxvar_acq_ind)

#Acquisitions

for acq_itr in range(num_acquisitions):
    #Maxvar Acquisition
    train_indices_maxvar.append(pool_indices_maxvar[maxvar_acq_ind])
    del pool_indices_maxvar[maxvar_acq_ind]
    #Setting acq_num (parameter after model) to 0 means context sampled randomly; setting to acq_itr+1 means attaching all acquired points
    #to all context sets; same for rand_acq_step
    [maxvar_mse, maxvar_acq_ind] = maxvar_acq_step(X[train_indices_maxvar], X_test, y[train_indices_maxvar], y_test, X[pool_indices_maxvar],
        y[pool_indices_maxvar], model, 0, variance_sample_size, num_batches, training_iterations)
    exp_mse_maxvar.append(maxvar_mse)
    pool_ind_ind_rand = np.random.choice(len(pool_indices_rand))
    train_indices_rand.append(pool_indices_rand[pool_ind_ind_rand])
    del pool_indices_rand[pool_ind_ind_rand]
    exp_mse_rand.append(rand_acq_step(X[train_indices_rand], X_test, y[train_indices_rand], y_test, X[pool_indices_rand], y[pool_indices_rand],
        model, acq_itr+1, variance_sample_size, num_batches, training_iterations))
    print('iteration ' + str(acq_itr+1) + ', RandMSE: ' + str(exp_mse_rand[-1]) + ', MaxVarMSE: ' + str(exp_mse_maxvar[-1]))
    np.save(filepath_mse_rand, np.array(exp_mse_rand))
    np.save(filepath_mse_maxvar, np.array(exp_mse_maxvar))
    np.save(filepath_dataind_rand, np.array(train_indices_rand))
    np.save(filepath_dataind_maxvar, np.array(train_indices_maxvar))
