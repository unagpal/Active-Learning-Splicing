import numpy as np
import keras
import tensorflow as tf
from keras.layers import Dense, Conv1D, SpatialDropout1D, MaxPooling1D,Input, Flatten
from keras.layers.core import Lambda
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
import os

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

  
"""
    Calculating acquisition function for minimization of tr(Var Univ) = tr(univ_covariance)
    univ_covariance: [size_univ x size_univ]
    other inputs denote how many of the samples are in training vs. pool
"""
def ei_trace_acquisition_fn_cnn (univ_covariance, num_pool_samples, num_training_samples):
    all_acq_values = np.zeros(num_pool_samples)
    for new_pt_ind in range(num_pool_samples):
        covariance_vector = univ_covariance[num_training_samples+new_pt_ind,:]
        all_acq_values[new_pt_ind] = np.sum(np.square(covariance_vector))/univ_covariance[num_training_samples+new_pt_ind, num_training_samples+new_pt_ind]
    return [all_acq_values, np.sum(univ_covariance.diagonal())]

"""
    Calculating acquisition function for minimization of det(Var Univ) = det(univ_covariance)
    univ_covariance = Var Univ = [size_univ x size_univ]
    other inputs denote how many of the samples are in training vs. pool
"""
def ei_det_acquisition_fn_cnn (univ_covariance, num_pool_samples, num_training_samples):
    all_acq_values = np.zeros(num_pool_samples)
    (original_det_sign, original_log_det) = np.linalg.slogdet(univ_covariance)
    for new_pt_ind in range(num_pool_samples):
        current_index = num_training_samples + new_pt_ind
        covariance_vector = np.expand_dims(univ_covariance[current_index, :], 1)
        updated_univ_covariance = np.subtract(univ_covariance, np.matmul(covariance_vector, covariance_vector.T)/univ_covariance[current_index,current_index])
        (sign, logdet) = np.linalg.slogdet(updated_univ_covariance) 
        all_acq_values[new_pt_ind] = logdet
    return [all_acq_values, original_log_det] 

#univ_covariance = np.array([[1,3,0.5],[3,4,6],[0.5,6,9]])
#print("univ covariance: " + str(univ_covariance) + ", pool num: 2, train num: 1")
#print("All acq values: " + str(ei_det_acquisition_fn_cnn(univ_covariance, 2, 1)))
#pred_train= np.array([[2,3,4,3],[0,1,2,1],[5,6,7,8]])
#pred_new = np.array([[10,11,11,10],[2,3,4,5]])
 


def permanent_dropout_layer(rate):
    seed = np.random.randint(0,100)
    return Lambda(lambda x: K.dropout(x,level=rate,seed=seed))

#n is the number of dropout masks (inferred from previous layers in create_dropout_masks)
#dim is the full output shape of predictions for all points for all dropout masks
def create_dropout_masks(dim, drop_prob):
    if (len(dim)==4):
        return 1/(1-drop_prob)*np.random.choice(2, size=((dim[0], 1, 1, dim[3])), p=[drop_prob, 1-drop_prob])
    return 1/(1-drop_prob)*np.random.choice(2, size=((dim[0], 1, 1, dim[3], dim[4])), p=[drop_prob, 1-drop_prob])

def create_spatial_dropout_masks(n, dim, drop_prob):
    if (len(dim) == 4):
        return 1/(1-drop_prob)*np.random.choice(2, size=((n, 1, 1, dim[3])), p=[drop_prob, 1-drop_prob])
    return 1/(1-drop_prob)*np.random.choice(2, size=((dim[0], 1, 1, 1, dim[4])), p=[drop_prob, 1-drop_prob])
  
def apply_dropout_masks(a,b):
    return np.squeeze(np.multiply(a,b))

def multi_mask_predict(layer_fn, multi_mask_input):
    layer_output = []
    for mask_num in range(multi_mask_input.shape[0]):
        layer_output.append(layer_fn((multi_mask_input[mask_num], 1)))
        #layer_output.append(layer_fn([multi_mask_input[mask_num]]))
    return np.array(layer_output)


#Used for calculating test MSE (average of MC dropout predictions)
def predict_with_uncertainty(f, x, n_iter=100):
    result = np.zeros((n_iter,x.shape[0]))
    for i in range(n_iter):
        predictions = np.array(f((x, 1))[0])[:,0]
        result[i,:] = predictions
    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty

#Forward pass through network with num_masks number of fixed dropout masks
def fixed_mask_forward_pass(model, forward_pass_input, num_masks):

    # Functions to retrieve output of intermediate layers
    # Needed for manual implementation of fixed dropout masks 
    # across all data points
    conv_1 = K.function([model.layers[0].input, K.learning_phase()],
                    [model.layers[1].output])

    pool_1 = K.function([model.layers[3].input, K.learning_phase()],
                   [model.layers[3].output])

    conv_2 = K.function([model.layers[5].input, K.learning_phase()],
                   [model.layers[5].output])

    pool_2 = K.function([model.layers[7].input, K.learning_phase()],
                   [model.layers[8].output])

    dense_1 = K.function([model.layers[10].input, K.learning_phase()],
                   [model.layers[10].output])

    dense_2 = K.function([model.layers[12].input, K.learning_phase()],
                   [model.layers[12].output])
    conv_1_output = np.array(conv_1((forward_pass_input, 1)))
    conv_1_masks = create_spatial_dropout_masks(num_masks, conv_1_output.shape, dropout_prob)
    pool_1_input = apply_dropout_masks(conv_1_output, conv_1_masks)
    pool_1_output = multi_mask_predict(pool_1, pool_1_input)
    pool_1_masks = create_dropout_masks(pool_1_output.shape, dropout_prob)
    conv_2_input = apply_dropout_masks(pool_1_output, pool_1_masks)
    conv_2_output = multi_mask_predict(conv_2, conv_2_input)
    conv_2_masks = create_spatial_dropout_masks(num_masks, conv_2_output.shape, dropout_prob)
    pool_2_input = apply_dropout_masks(conv_2_output, conv_2_masks)
    pool_2_output = multi_mask_predict(pool_2, pool_2_input)
    pool_2_masks = create_dropout_masks(pool_2_output.shape, dropout_prob)
    dense_1_input = apply_dropout_masks(pool_2_output, pool_2_masks)
    dense_1_output = multi_mask_predict(dense_1, dense_1_input)
    dense_1_masks = create_dropout_masks(dense_1_output.shape, dropout_prob)
    dense_2_input = apply_dropout_masks(dense_1_output, dense_1_masks)
    all_output = np.squeeze(multi_mask_predict(dense_2, dense_2_input))
    return all_output
 
def get_acquisition_fn (model, X_train_sample, X_cand, num_masks):
    forward_pass_input = np.concatenate((X_train_sample, X_cand))
    forward_pass_output = fixed_mask_forward_pass(model, forward_pass_input, num_masks).T
    return ei_det_acquisition_fn_cnn(np.cov(forward_pass_output), len(X_cand), len(X_train_sample)) 
    #return ei_acquisition_fn(forward_pass_output[0:rep_sample_size,:], forward_pass_output[rep_sample_size:,:])
#a e get_acnucquisition_fn(forward_pass_output[0:num_train,:], forward_pass_output[num_train:, :])
#print(a.shape)
#print(a)

#Active Learning Experiment Settings
num_acquisitions = 10000
sample_size=500
acq_fn_dropout_iterations = 50
mse_dropout_iterations = 100
size_train = 100
batchsize=128
batch_size = 100

#Loading Data
X = np.array(inputs)
y = np.array(prob_s1)

#Model with Spatial Dropout
dropout_prob = 0.15
lengthscale = 3
Weight_Decay = 0.0005
inputlayer = Input(shape=(seq_len, len(nuc_arr)))
x = Conv1D(seq_len,(4), strides=1, input_shape=(seq_len,len(nuc_arr)), activation='relu')(inputlayer)
x = SpatialDropout1D(dropout_prob)(x)
x = MaxPooling1D(pool_size=3)(x)
x = permanent_dropout_layer(dropout_prob)(x)
x = Conv1D(seq_len//2, (4), strides=1, activation='relu')(x)
x = SpatialDropout1D(dropout_prob)(x)
x = MaxPooling1D(pool_size=3)(x)
x = Flatten()(x)
x = permanent_dropout_layer(dropout_prob)(x)
#x = Dense(50, activation='relu', kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.l2(Weight_Decay))(x)
x = Dense(50, W_regularizer=l2(Weight_Decay), init='normal',  activation='relu')(x)
x = permanent_dropout_layer(dropout_prob)(x)
#outputlayer = Dense(1, kernel_initializer='random_normal',kernel_regularizer = keras.regularizers.l2(Weight_Decay))(x)
outputlayer = Dense(1, W_regularizer=l2(Weight_Decay), init='normal')(x)

model_rand = Model(inputlayer, outputlayer)
model_rand.compile(loss='mean_squared_error', optimizer='rmsprop')
model_expimprovement = Model(inputlayer, outputlayer)
model_expimprovement.compile(loss='mean_squared_error', optimizer='rmsprop')

#Initializing datasets and filepaths
#Note: EI = Expected Improvement
exp_mse_rand = []
exp_mse_expimprovement = []
#filepath_rand = os.path.expanduser("~/RandMSEEIExp1.npy")
#filepath_rand_indices = os.path.expanduser("~/RandIndicesEIExp1.npy")
filepath_rand_mse = os.path.expanduser("~/BATCHRAND_BS100_start100_MSEInd2.npy") 
filepath_expimprovement_det_univ_original = os.path.expanduser("~/EICNNDetReduxTest_LogDetUnivOrig_Ind2.npy")
filepath_expimprovement_acq_val = os.path.expanduser("~/EICNNDetReduxTest_LogAcqValInd2.npy")
filepath_expimprovement_det_univ_new = os.path.expanduser("~/EICNNDetReduxTest_LogDetUnivNew_Ind2.npy")
all_train_indices = np.load(os.path.expanduser('~/trainindices2.npy'))
filepath_rand_indices = os.path.expanduser("~/BATCHRAND_BS100_start100_Ind2_TrInd.npy") 
filepath_det_errors = os.path.expanduser("~/EICNNDetReduxTest_OffByOneErrors.npy")
train_data_indices_rand = all_train_indices[0:size_train].tolist()
train_data_indices_expimprovement = all_train_indices[0:size_train].tolist()
test_indices = np.load(os.path.expanduser('~/testindices2.npy'))
#X_train_rand = X[train_data_indices_rand]
#y_train_rand = y[train_data_indices_rand]
X_train_expimprovement = X[train_data_indices_expimprovement]
y_train_expimprovement = y[train_data_indices_expimprovement]
X_test = X[test_indices]
y_test = y[test_indices]

#First iteration of active learning experiment
model_expimprovement.fit(X_train_expimprovement, y_train_expimprovement, epochs=50, batch_size=batchsize, verbose=0)
model_rand.fit(X_train_expimprovement, y_train_expimprovement, epochs=50, batch_size=batchsize, verbose=0)
f_expimprovement = K.function([model_expimprovement.layers[0].input, K.learning_phase()], 
               [model_expimprovement.layers[-1].output])
predictions_with_uncertainty = predict_with_uncertainty(f_expimprovement, X_test, n_iter=mse_dropout_iterations)
y_predicted = predictions_with_uncertainty[0]
mse = np.mean((y_test-y_predicted)**2)
exp_mse_rand.append(mse)
exp_mse_expimprovement.append(mse)

#Initializing pool indices
#pool_indices_rand = [ind for ind in all_train_indices if ind not in train_data_indices_rand]
pool_indices_rand = [ind for ind in all_train_indices if ind not in train_data_indices_rand]
ei_acquired_values = []
all_original_det = []
all_updated_det = []
det_errors = []
#Acquisition Loop
for acq in range(num_acquisitions):
    #Adding a point randomly to random acquisition training set
    pool_ind_ind_rand = np.random.choice(len(pool_indices_rand), size=batch_size, replace=False)
    data_ind_rand = np.array(pool_indices_rand)[pool_ind_ind_rand]
    train_data_indices_rand += data_ind_rand.tolist()
    X_train_rand = X[train_data_indices_rand]
    y_train_rand = y[train_data_indices_rand]
    pool_indices_rand = np.delete(pool_indices_rand, pool_ind_ind_rand).tolist()
    #Retraining RandAcquisition model and calculating MSE
    model_rand.fit(X_train_rand, y_train_rand, epochs=50, batch_size=batchsize, verbose=0)
    f_rand = K.function([model_rand.layers[0].input, K.learning_phase()], 
               [model_rand.layers[-1].output])
    predictions_with_uncertainty = predict_with_uncertainty(f_rand, X_test, n_iter=mse_dropout_iterations)
    y_predicted = predictions_with_uncertainty[0]
    mse_rand = np.mean((y_test-y_predicted)**2)
    exp_mse_rand.append(mse_rand)
    np.save(filepath_rand_indices, np.array(train_data_indices_rand))
    np.save(filepath_rand_mse, np.array(exp_mse_rand))
    print(exp_mse_rand)
    """
    #Expected Improvement acquisition
    sample_indices = np.random.choice(len(pool_indices_expimprovement)+len(train_data_indices_expimprovement), sample_size, replace=False)
    sample_indices_train = sample_indices[sample_indices < len(train_data_indices_expimprovement)]
    sample_indices_pool = sample_indices[sample_indices >= len(train_data_indices_expimprovement)]
    X_train_fn = X[train_data_indices_expimprovement][sample_indices_train]
    X_pool_fn = np.concatenate((X[train_data_indices_expimprovement], X[pool_indices_expimprovement]))[sample_indices_pool]
    acq_fn_results = get_acquisition_fn(model_expimprovement, X_train_fn, X_pool_fn, acq_fn_dropout_iterations)
    acq_fn_values = acq_fn_results[0]
    if (acq_fn_values.shape[0] != sample_size):
        det_errors.append(1)
    else:
        det_errors.append(0) 
    all_original_det.append(acq_fn_results[1])
    ei_acquired_values.append(np.amin(acq_fn_values))
    acq_ind_ind = sample_indices_pool[np.argmin(acq_fn_values)] - len(train_data_indices_expimprovement)
    acq_ind = pool_indices_expimprovement[acq_ind_ind]
    train_data_indices_expimprovement.append(acq_ind)
    X_train_expimprovement = X[train_data_indices_expimprovement]
    y_train_expimprovement = y[train_data_indices_expimprovement]
    del pool_indices_expimprovement[acq_ind_ind]
    #Retraining EI model and calculating MSE
    model_expimprovement.fit(X_train_expimprovement, y_train_expimprovement, epochs=50, batch_size=batchsize, verbose=0)
    f_expimprovement = K.function([model_expimprovement.layers[0].input, K.learning_phase()], 
               [model_expimprovement.layers[-1].output])
    predictions_with_uncertainty = predict_with_uncertainty(f_expimprovement, X_test, n_iter=mse_dropout_iterations)
    y_predicted = predictions_with_uncertainty[0]
    mse_ei = np.mean((y_test-y_predicted)**2)
    exp_mse_expimprovement.append(mse_ei)
    all_univ_pred_new = fixed_mask_forward_pass(model_expimprovement, np.concatenate((X_train_fn, X_pool_fn)), acq_fn_dropout_iterations).T
    all_updated_det.append(np.linalg.slogdet(np.cov(all_univ_pred_new))[1])
    print('Cov univ shape: ' + str(np.cov(all_univ_pred_new).shape))
    print('All acq values: ' + str(acq_fn_values))
    print('Acq fn output shape: ' + str(acq_fn_values.shape))
    print('Det orig: ' + str(all_original_det[-1]))
    print('Det new: ' + str(all_updated_det[-1]))
    print("Acq Itr " + str(acq) + ", ExpImprovement Univ MSE: " + str(exp_mse_expimprovement[-1]) + ', EI Value: ' + str(ei_acquired_values[-1]) + ', Det(Cov Univ): ' + str(all_updated_det[-1])+', Det Reduction: ' + str(all_original_det[-1]-all_updated_det[-1]) + 'Det Errors: ' + str(det_errors))
    np.save(filepath_expimprovement_mse, np.array(exp_mse_expimprovement))
    np.save(filepath_expimprovement_acq_val, np.array(ei_acquired_values)) 
    np.save(filepath_expimprovement_det_univ_original, np.array(all_original_det))
    np.save(filepath_expimprovement_det_univ_new, np.array(all_updated_det))
    np.save(filepath_det_errors, np.array(det_errors))
    """ 
