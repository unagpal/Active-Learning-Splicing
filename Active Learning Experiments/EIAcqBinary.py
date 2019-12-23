import numpy as np
import keras
from keras.datasets import mnist
import sys
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K

#Used for calculating test acc (average of MC dropout predictions)
def predict_with_uncertainty(f, x, n_iter=100):
    result = np.zeros((n_iter,x.shape[0], 2))
    for i in range(n_iter):
        predictions = np.array(f((x, 1))[0])
        result[i,:, :] = predictions
    prediction = result.mean(axis=0)
    return prediction

#dim is the full output shape of predictions for all points for all dropout masks
def create_dropout_masks(dim, num_masks, drop_prob):
    if (len(dim)==4):
        return 1/(1-drop_prob)*np.random.choice(2, size=((dim[0], 1, 1, dim[3])), p=[drop_prob, 1-drop_prob])
    return 1/(1-drop_prob)*np.random.choice(2, size=((num_masks, 1, dim[2], dim[3], dim[4])), p=[drop_prob, 1-drop_prob])

def apply_dropout_masks(a,b):
    return np.squeeze(np.multiply(a,b))

def multi_mask_predict(layer_fn, multi_mask_input):
    layer_output = []
    for mask_num in range(multi_mask_input.shape[0]):
        layer_output.append(layer_fn((multi_mask_input[mask_num], 1)))
        #layer_output.append(layer_fn([multi_mask_input[mask_num]]))
    return np.array(layer_output)

def fixed_mask_forward_pass(model, forward_pass_input, num_masks, dropout_prob):
  conv_1 = K.function([model.layers[0].input, K.learning_phase()],
                    [model.layers[1].output])
  conv_2 = K.function([model.layers[3].input, K.learning_phase()],
                    [model.layers[4].output])
  pool_1 = K.function([model.layers[5].input, K.learning_phase()],
                    [model.layers[6].output])
  dense_1 = K.function([model.layers[8].input, K.learning_phase()],
                    [model.layers[9].output])
  dense_2 = K.function([model.layers[11].input, K.learning_phase()],
                    [model.layers[12].output])
  conv_1_output = np.array(conv_1((forward_pass_input, 1)))
  dropout_masks_1 = create_dropout_masks(conv_1_output.shape, num_masks, dropout_prob)
  conv_2_input = apply_dropout_masks(conv_1_output, dropout_masks_1)
  conv_2_output = np.squeeze(multi_mask_predict(conv_2, conv_2_input))
  pool_1_output = multi_mask_predict(pool_1, np.squeeze(conv_2_output))
  dropout_masks_2 = create_dropout_masks(pool_1_output.shape, num_masks, dropout_prob)
  dense_1_input = apply_dropout_masks(pool_1_output, dropout_masks_2)
  dense_1_output = multi_mask_predict(dense_1, dense_1_input)
  dropout_masks_3 = create_dropout_masks(dense_1_output.shape, num_masks, dropout_prob)
  dense_2_input = apply_dropout_masks(dense_1_output, dropout_masks_3)
  output = np.squeeze(multi_mask_predict(dense_2, dense_2_input))
  return output[:,:,0] 

def run_model (train_data_indices):
  X_train = np.expand_dims(X_train_All[train_data_indices], axis=1)
  y_train = y_train_All[train_data_indices]
  y_train[y_train==7] = 0
  y_train = keras.utils.to_categorical(y_train, num_classes=2)
  train_size = y_train.shape[0]
  Weight_Decay = 2.5/train_size
  dropout_prob = 0.25
  batch_size=128
  nb_filters = 30
  nb_pool = 3
  nb_conv = 4
  img_rows = img_cols = 28
  nb_classes = 2
  model = Sequential()
  model.add(Convolution2D(nb_filters, nb_conv, strides=1, data_format="channels_first", input_shape=(1, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(Dropout(dropout_prob))
  model.add(Convolution2D(nb_filters, nb_conv, strides=2))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Flatten())
  model.add(Dropout(dropout_prob))
  model.add(Dense(100, W_regularizer=l2(Weight_Decay)))
  model.add(Activation('relu'))
  model.add(Dropout(dropout_prob))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  hist = model.fit(X_train, y_train, epochs=60, steps_per_epoch=100, verbose=0)
  f = K.function([model.layers[0].input,K.learning_phase()],[model.layers[-1].output])
  y_test_output = predict_with_uncertainty(f, X_test, n_iter=100)
  y_test_predictions = np.argmax(y_test_output, axis=1)
  return [np.sum(y_test_predictions==y_test_original)/(y_test_original.shape[0]), model]

def get_acquisition_fn(pool_forward_pass, num_tr_pts, tau_inv_prop):
  pool_cov = np.cov(pool_forward_pass.T)
  tau_inv = np.trace(pool_cov)/(pool_cov.shape[0])*tau_inv_prop
  pool_cov = pool_cov + tau_inv * np.identity(pool_cov.shape[0])
  num_pool_pts = int(pool_cov.shape[0])-num_tr_pts
  acq_values = np.zeros(num_pool_pts)
  for new_pt_ind in range(num_pool_pts):
    cov_vector = pool_cov[num_tr_pts+new_pt_ind,:]
    acq_values[new_pt_ind] = np.sum(np.square(cov_vector))/(pool_cov[num_tr_pts+new_pt_ind,num_tr_pts+new_pt_ind]) 
  return acq_values


if __name__ == "__main__":
  data_path = "/home/unagpal/mnist/"
  train_data = np.loadtxt(data_path + "mnist_train.csv", 
			  delimiter=",")
  test_data = np.loadtxt(data_path + "mnist_test.csv", 
			 delimiter=",") 
  y_train_All = train_data[:,0]
  y_test = test_data[:,0]
  X_train_All = train_data[:,1:].reshape((60000,28,28))
  X_test = test_data[:,1:].reshape((10000,28,28)) 
  #(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()
  train_ind = np.concatenate((np.argwhere(y_train_All==1), np.argwhere(y_train_All==7))).flatten()
  test_ind = np.concatenate((np.argwhere(y_test==1), np.argwhere(y_test==7))).flatten()
  y_test = y_test[test_ind]
  y_test[y_test==7] = 0
  y_test_original = y_test
  y_test = keras.utils.to_categorical(y_test, num_classes=2)
  X_test = np.expand_dims(X_test[test_ind], axis=1)
  folder_path = "/home/unagpal/mnist/"
  acq_file = sys.argv[2]
  ind_file = sys.argv[3]
  #AL Parameters
  #train_size_init = 10
  dropout_prob = 0.25
  num_experiments = 1
  num_acquisitions = 400 
  pool_sample_size = 2000
  num_masks = 50
  tau_inv_prop = float(sys.argv[1])
  #Keras Model Parameters
  num_classes = 2
  nb_filters = 30
  nb_pool = 3
  nb_conv = 4
  img_rows = img_cols = 28
  print('Running')	  
  all_tr_ind = []
  all_acc = []
  for e in range(num_experiments):
    exp_acc = []
    #Initial training/accuracy
    train_data_indices = list(np.load(folder_path + 'trainindices' + str(e+2) + '.npy'))
    pool_indices = [i for i in train_ind if i not in train_data_indices]
    model_results = run_model(train_data_indices)
    exp_acc.append(model_results[0])
    all_acc.append(exp_acc)
    all_tr_ind.append(train_data_indices)
    for acq in range(num_acquisitions):
      all_ind_ind = np.random.choice(len(pool_indices)+len(train_data_indices), pool_sample_size, replace=False)
      pool_ind_ind = all_ind_ind[all_ind_ind>=len(train_data_indices)] - len(train_data_indices)
      train_ind_ind = all_ind_ind[all_ind_ind<len(train_data_indices)]
      pool_ind_sample = np.array(pool_indices)[pool_ind_ind]
      train_ind_sample = np.array(train_data_indices)[train_ind_ind]
      X_sample = np.concatenate((np.expand_dims(X_train_All[train_ind_sample],axis=1), np.expand_dims(X_train_All[pool_ind_sample], axis=1)))
      pool_forward_pass_results = fixed_mask_forward_pass(model_results[1], X_sample, num_masks, dropout_prob)  
      acq_fn_values = get_acquisition_fn(pool_forward_pass_results, len(train_ind_sample), tau_inv_prop)
      acq_ind_ind = np.argmax(acq_fn_values)
      train_data_indices.append(pool_ind_sample[acq_ind_ind])
      pool_indices.remove(pool_ind_sample[acq_ind_ind])
      model_results = run_model(train_data_indices)
      exp_acc.append(model_results[0])
      all_acc[-1] = exp_acc
      all_tr_ind[-1] = train_data_indices
      np.save(folder_path+acq_file, np.array(all_acc))
      np.save(folder_path+ind_file, np.array(all_tr_ind))
      print('All Acc: ' + str(all_acc))
      #print('All Ind: ' + str(all_tr_ind))
  print('All Acc: ' + str(all_acc))
