import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import math
import tensorflow_probability as tfp
# Imports and functions from 5' Model
from matplotlib.colors import LogNorm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,Input, Flatten
from keras import models
from keras import optimizers
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
tf.compat.v1.disable_eager_execution()
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
reads_path = os.path.expanduser("~") + "/GSM1911086_A5SS_spliced_reads.txt"
seq_path = os.path.expanduser("~") + "/GSM1911085_A5SS_seq.txt"
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

for i in range(n):
    prob_s1[i] = prob_SD1(float(read_lines[i][1]), float(read_lines[i][2]))
    inputs[i] = seq_to_arr(seq_lines[i][1])

# The CNP takes as input a `CNPRegressionDescription` namedtuple with fields:
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
                max_num_context,
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
        self._max_num_context = max_num_context
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

        num_total_points = tf.shape(input=xdata)[1]

        # Expand and take the difference
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = tf.reduce_sum(
            input_tensor=norm, axis=-1)  # [B, data_size, num_total_points, num_total_points]

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
            # Select the targets (all training data)
            indices_test = np.arange(int(num_testing_points))
            target_x_values = [tf.gather(self._x_test, tf.convert_to_tensor(value=indices_test))]
            target_y_values = [tf.gather(self._y_test, tf.convert_to_tensor(value=indices_test))]
            target_x = tf.stack(target_x_values)
            target_y_values = tf.stack(target_y_values)
            target_y = target_y_values
            #target_y = tf.expand_dims(target_y_values, axis=2)

            # Select the observations (all testing data)
            indices_train = np.arange(int(num_training_points))
            context_x_values = [tf.gather(self._x_train, tf.convert_to_tensor(value=indices_train))]
            context_y_values = [tf.gather(self._y_train, tf.convert_to_tensor(value=indices_train))]
            context_x = tf.stack(context_x_values)
            context_y_values = tf.stack(context_y_values)
            context_y = context_y_values
        #context_y = tf.expand_dims(context_y_values, axis=2)
        #context_x = tf.gather(x_values, idx[:int(num_context)], axis=1)
        #context_y = tf.gather(y_values, idx[:int(num_context)], axis=1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_context = self._max_num_context
            x_values_cont = []
            y_values_cont = []
            x_values_target = []
            y_values_target = []
            num_training_points = tf.Tensor.get_shape(self._y_train)[0]
            #print(num_context)
            #print(num_target)
            #print(num_total_points)
            for batch_num in range(self._batch_size):
                context_indices = np.array([i for i in range(num_context*batch_num, num_context*(batch_num+1))]
                    + [j for j in range(self._batch_size*num_context, num_training_points)])
                #indices = np.random.choice(num_training_points,num_total_points,replace=False)
                x_values_context = tf.gather(self._x_train, tf.convert_to_tensor(value=context_indices))
                y_values_context = tf.gather(self._y_train, tf.convert_to_tensor(value=context_indices))
                x_values_cont.append(x_values_context)
                y_values_cont.append(y_values_context)
                x_values_target.append(self._x_train)
                y_values_target.append(self._y_train)
            context_x = tf.stack(x_values_cont)
            context_y = tf.stack(y_values_cont)
            target_x= tf.stack(x_values_target)
            target_y = tf.stack(y_values_target)
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
            num_total_points=tf.shape(input=target_x)[1],
            num_context_points=num_context)

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
        #print('context x shape: ' + str(context_x.get_shape().as_list()))
        #print('context y shape: ' + str(context_y.get_shape().as_list()))
        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)
        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_context_points, filter_size = encoder_input.shape.as_list()
        hidden = tf.reshape(encoder_input, (batch_size * int(num_context_points), -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.compat.v1.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.compat.v1.layers.dense(
                hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, int(num_context_points), size))

        # Aggregator: take the mean over all points
        representation = tf.reduce_mean(input_tensor=hidden, axis=1)

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

        # Concatenate the representation and the target_x
        representation = tf.tile(
            tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
        input = tf.concat([representation, target_x], axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = input.shape.as_list()
        hidden = tf.reshape(input, (batch_size * num_total_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.compat.v1.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.compat.v1.layers.dense(
                hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        #dist = tf.contrib.distributions.MultivariateNormalDiag(
        #loc=mu, scale_diag=sigma)

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
tf.compat.v1.reset_default_graph()

#Active Learning Experiments

def calculate_mse (X_train_1, X_test_1, y_train_1, y_test_1):
    #Resetting Graphs
    tf.compat.v1.reset_default_graph()

    X_train_lst = np.array([data_point.flatten() for data_point in X_train_1])
    X_train_1 = tf.constant(X_train_lst)
    X_test_lst = np.array([test_point.flatten() for test_point in X_test_1])
    X_test_1 = tf.constant(X_test_lst)
    y_train_lst = np.array([[training_y] for training_y in y_train_1])
    y_train_1 = tf.constant(y_train_lst)
    y_test_1 = tf.convert_to_tensor(value=y_test_1, dtype=tf.float64)

    #Train and Test Datasets
    TRAINING_ITERATIONS = 2500
    MAX_CONTEXT_POINTS = 500
    #MAX_CONTEXT_POINTS = 300
    dataset_train_5_prime = GPCurvesReader(batch_size=2, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1, max_num_context=MAX_CONTEXT_POINTS)
    data_train = dataset_train_5_prime.generate_curves()
    dataset_test_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_test_1, y_train=y_train_1, y_test=y_test_1, max_num_context=MAX_CONTEXT_POINTS,testing=True)
    data_test = dataset_test_5_prime.generate_curves()

    # Sizes of the layers of the MLPs for the encoder and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    encoder_output_sizes = [64, 64, 64, 64]
    decoder_output_sizes = [64, 64, 2]

    # Define the model
    model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)

    # Define the loss
    log_prob, _, _ = model(data_train.query, data_train.num_total_points,
         data_train.num_context_points, data_train.target_y)

    loss = -tf.reduce_mean(input_tensor=log_prob)

    # Get the predicted mean and variance at the target points for the testing set
    _, mu, sigma = model(data_test.query, data_test.num_total_points,
         data_test.num_context_points)

    # Set up the optimizer and train step
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    init = tf.compat.v1.initialize_all_variables()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for it in range(TRAINING_ITERATIONS):
            sess.run([train_step])

    _, mu, sigma = model(data_test.query, data_test.num_total_points,
        data_test.num_context_points)
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.initializers.global_variables().run()
    predictions = mu[0,:,0].eval()
    true_y = y_test_1.eval()
    sess.close()

    #Returning MSE
    return [mean_squared_error(true_y, predictions), model]

def test_synthetic_sigmoid_data():
    x = np.random.uniform(-1,1,1000)
    y = [1/(1+math.exp(-k)) for k in x]
    x_train = x[0:600]
    y_train = y[0:600]
    x_test = x[600:999]
    y_test = y[600:999]
    results = calculate_mse(x_train, x_test,y_train, y_test)
    print("MSE: " + str(results[0]))

#test_synthetic_sigmoid_data()

def update_train_data(train_data_ind, acq_index, pool_indices):
    new_index = pool_indices[acq_index]
    train_data_ind.append(new_index)
    return train_data_ind

def maxvar_acquisition(trained_model, X_train_1, X_test_1, y_train_1, y_test_1, X_pool, y_pool):
    X_train_lst = np.array([data_point.flatten() for data_point in X_train_1])
    X_train_1 = tf.constant(X_train_lst)
    X_pool_lst = np.array([pool_point.flatten() for pool_point in X_pool])
    X_pool_1 = tf.constant(X_pool_lst)
    y_train_lst = np.array([[training_y] for training_y in y_train_1])
    y_train_1 = tf.constant(y_train_lst)
    y_pool_1 = tf.convert_to_tensor(value=y_pool, dtype=tf.float64)
    MAX_CONTEXT_POINTS = X_train_1.shape[0]
    dataset_pool_5_prime = GPCurvesReader(
    batch_size=1, x_train=X_train_1, x_test=X_pool_1, y_train=y_train_1, y_test=y_pool_1, max_num_context=MAX_CONTEXT_POINTS,testing=True)
    data_pool = dataset_pool_5_prime.generate_curves()
    _, mu, sigma = trained_model(data_pool.query, data_pool.num_total_points, data_pool.num_context_points)
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.initializers.global_variables().run()
    sigma_arr = sigma[0,:,0].eval()
    max_variance_ind = np.argmax(sigma_arr)
    sess.close()
    return max_variance_ind


train_indices_1 = np.load(os.path.expanduser(".") + "/trainindices1.npy")
test_indices_1 = np.load(os.path.expanduser(".") + "/testindices1.npy")
X_train_1 = inputs[train_indices_1]
X_test_1 = inputs[test_indices_1]
y_train_1 = prob_s1[train_indices_1]
y_test_1 = prob_s1[test_indices_1]
all_train_data_indices_rand = []
all_train_data_indices_maxvar = []

#Calculating MSE when predicting all 0's as a Benchmark
#test_zeroes = np.zeros(y_test_1.shape[0])
#print('mean squared error when predicting exclusively 0: ' + str(mean_squared_error(y_test_1, test_zeroes)))

all_mse_rand = []
all_mse_maxvar = []
num_experiments = 2
initial_points = 1000
num_acquisitions = 100
for exp in range(num_experiments):
    print("Starting experiment " + str(exp+1))
    #Obtaining starting train data, test data, and pool data
    all_train_indices = np.load(os.path.expanduser(".") + "/trainindices" + str(exp+1) + ".npy")
    train_data_indices = all_train_indices[0:initial_points].tolist()
    train_indices_maxvar = all_train_indices[0:initial_points].tolist()
    exp_mse_rand = []
    exp_mse_maxvar = []
    test_indices = np.load(os.path.expanduser(".") + "/testindices" + str(exp+1) + ".npy")
    pool_indices_rand = [i for i in range(265137) if i not in train_data_indices and i not in test_indices]
    pool_indices_maxvar = [i for i in range(265137) if i not in train_data_indices and i not in test_indices]
    X_train = inputs[train_data_indices]
    X_test = inputs[test_indices]
    y_train = prob_s1[train_data_indices]
    y_test = prob_s1[test_indices]
    X_train_maxvar = X_train
    y_train_maxvar = y_train
    #Training and storing MSE of initial model
    init_calculations = calculate_mse(X_train, X_test, y_train, y_test)
    maxvar_model = init_calculations[1]
    exp_mse_rand.append(init_calculations[0])
    exp_mse_maxvar.append(init_calculations[0])
    for acq_itr in range(num_acquisitions):
        print("Starting acquisition iteration " + str(acq_itr+1))
        #Max Variance Acquisition
        maxvar_acq_ind = maxvar_acquisition(maxvar_model, X_train_maxvar, X_test, y_train_maxvar, y_test, inputs[pool_indices_maxvar], prob_s1[pool_indices_maxvar])
        train_indices_maxvar = update_train_data(train_indices_maxvar, maxvar_acq_ind, pool_indices_maxvar)
        X_train_maxvar = inputs[train_indices_maxvar]
        y_train_maxvar = prob_s1[train_indices_maxvar]
        del pool_indices_maxvar[maxvar_acq_ind]
        #Training new Max Variance Model
        maxvar_results = calculate_mse(X_train_maxvar, X_test, y_train_maxvar, y_test)
        maxvar_model = maxvar_results[1]
        exp_mse_maxvar.append(maxvar_results[0])
        #Selecting and adding random pool point to training data
        acq_ind = np.random.randint(len(pool_indices_rand))
        train_data_indices = update_train_data(train_data_indices, acq_ind, pool_indices_rand)
        X_train = inputs[train_data_indices]
        y_train = prob_s1[train_data_indices]
        del pool_indices_rand[acq_ind]
        #Training New Model and adding new MSE
        exp_mse_rand.append(calculate_mse(X_train, X_test, y_train, y_test)[0])
        print('maxvar mse: ' + str(exp_mse_maxvar))
        print('rand mse: ' + str(exp_mse_rand))
    #Saving Training Ind and MSE
    all_mse_rand.append(exp_mse_rand)
    all_mse_maxvar.append(exp_mse_maxvar)
    all_train_data_indices_rand.append(train_data_indices)
    all_train_data_indices_maxvar.append(train_indices_maxvar)
np.save(os.path.expanduser(".") + "/CNP_MLP_Rand_TrInd.npy", np.array(all_train_data_indices_rand))
np.save(os.path.expanduser(".") + "/CNP_MLP_Rand_MSE.npy", np.array(all_mse_rand))
np.save(os.path.expanduser(".") + "/CNP_MLP_MaxVar_TrInd.npy", np.array(all_train_data_indices_maxvar))
np.save(os.path.expanduser(".") + "/CNP_MLP_MaxVar_MSE.npy", np.array(all_mse_maxvar))
