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

#Used for making repeated pool predictions
def predict_pool_with_uncertainty(f, x, n_iter=100):
    result = np.zeros((n_iter,x.shape[0], 2))
    for i in range(n_iter):
        predictions = np.array(f((x, 1))[0])
        result[i,:, :] = predictions
    return result


def run_model (train_data_indices, pool_sample_indices):
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
  f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
  y_test_output = predict_with_uncertainty(f, X_test, n_iter=100)
  y_test_predictions = np.argmax(y_test_output, axis=1)
  pool_sample_predictions = predict_pool_with_uncertainty(f, np.expand_dims(X_train_All[pool_sample_indices], axis=1), n_iter=50)
  return [np.sum(y_test_predictions==y_test_original)/(y_test_original.shape[0]), pool_sample_predictions[:,:,0]]


def get_acquisition_fn(pool_forward_pass):
  #Calculating First Term
  print('pool forward pass shape: ' + str(pool_forward_pass.shape))
  num_pool_pts = pool_forward_pass.T.shape[0]
  num_dropout_predictions = pool_forward_pass.T.shape[1]
  all_prob_pred_class_1 = np.mean(pool_forward_pass.T, axis=1)
  all_prob_pred_class_2 = np.subtract(1, all_prob_pred_class_1)
  ent_term_1 = np.multiply(all_prob_pred_class_1, np.log2(all_prob_pred_class_1))
  ent_term_2 = np.multiply(all_prob_pred_class_2, np.log2(all_prob_pred_class_2))
  ent_term_1[np.isnan(ent_term_1)==True] = 0.0
  ent_term_2[np.isnan(ent_term_2)==True] = 0.0
  ent_acq_values = np.absolute(np.add(ent_term_1, ent_term_2))
  #Calculating Second Term
  all_entropies = np.zeros((num_pool_pts, num_dropout_predictions))
  all_ent_term_1 = np.multiply(pool_forward_pass.T, np.log2(pool_forward_pass.T))
  all_ent_term_2 = np.multiply(np.subtract(1, pool_forward_pass.T), np.log2(np.subtract(1, pool_forward_pass.T)))
  all_ent_term_1[np.isnan(all_ent_term_1)==True] = 0.0
  all_ent_term_2[np.isnan(all_ent_term_2)==True] = 0.0
  all_entropies = np.absolute(np.add(all_ent_term_1, all_ent_term_2))
  all_expected_entropies = np.mean(all_entropies, axis=1)
  bald_values = ent_acq_values - all_expected_entropies
  print('predicted mean probability of acquired pt: ' + str(all_prob_pred_class_1[np.argmax(bald_values)]))
  print('all predicted probabilities of acquired pt: ' + str(pool_forward_pass.T[np.argmax(bald_values),:]))
  return bald_values


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
  folder_path = "/home/unagpal/mnist/1_7_Binary_Results/"
  acq_file = sys.argv[1]
  ind_file = sys.argv[2]
  #AL Parameters
  #train_size_init = 10
  dropout_prob = 0.25
  num_experiments = 2
  num_acquisitions = 321
  pool_sample_size = 2000
  num_masks = 50
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
    train_data_indices = list(np.load(folder_path + 'trainindices' + str(e+1) + '.npy'))
    pool_indices = [i for i in train_ind if i not in train_data_indices]
    all_tr_ind.append(train_data_indices)
    all_acc.append([])
    for acq in range(num_acquisitions):
      all_ind_ind = np.random.choice(len(pool_indices)+len(train_data_indices), pool_sample_size, replace=False)
      pool_ind_ind = all_ind_ind[all_ind_ind>=len(train_data_indices)] - len(train_data_indices)
      train_ind_ind = all_ind_ind[all_ind_ind<len(train_data_indices)]
      pool_ind_sample = np.array(pool_indices)[pool_ind_ind]
      train_ind_sample = np.array(train_data_indices)[train_ind_ind]
      #X_sample = np.concatenate((np.expand_dims(X_train_All[train_ind_sample],axis=1), np.expand_dims(X_train_All[pool_ind_sample], axis=1)))
      model_results = run_model(train_data_indices, pool_ind_sample)
      exp_acc.append(model_results[0])
      all_acc[-1] = exp_acc
      pool_forward_pass_results = model_results[1]  
      acq_fn_values = get_acquisition_fn(pool_forward_pass_results) 
      acq_ind_ind = np.argmax(acq_fn_values)
      train_data_indices.append(pool_ind_sample[acq_ind_ind])
      all_tr_ind[-1] = train_data_indices
      pool_indices.remove(pool_ind_sample[acq_ind_ind])
      print('len train_data_indices: ' + str(len(train_data_indices)))
      np.save(folder_path+acq_file, np.array(all_acc))
      np.save(folder_path+ind_file, np.array(all_tr_ind))
      print('All Acc: ' + str(all_acc))
      print('All Ind: ' + str(all_tr_ind))
