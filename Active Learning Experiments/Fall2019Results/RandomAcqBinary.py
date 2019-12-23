import numpy as np
import keras
from keras.datasets import mnist
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K

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

#AL Parameters
#train_size_init = 6
dropout_prob = 0.25
num_experiments = 3
num_acquisitions = 400 

#Keras Model Parameters
num_classes = 2
nb_filters = 30
nb_pool = 3
nb_conv = 4
img_rows = img_cols = 28

#Used for calculating test acc (average of MC dropout predictions)
def predict_with_uncertainty(f, x, n_iter=100):
    result = np.zeros((n_iter,x.shape[0], 2))
    for i in range(n_iter):
        predictions = np.array(f((x, 1))[0])
        result[i,:, :] = predictions
    prediction = result.mean(axis=0)
    return prediction

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
  f_rand = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
  y_test_output = predict_with_uncertainty(f_rand, X_test, n_iter=100)
  y_test_predictions = np.argmax(y_test_output, axis=1)
  return np.sum(y_test_predictions==y_test_original)/(y_test_original.shape[0])

all_acc = []
all_tr_ind = []
for e in range(num_experiments):
  exp_acc = []
  all_acc.append(exp_acc)
  #Initial training/accuracy
  train_data_indices = list(np.load(folder_path + 'trainindices' + str(e+1) + '.npy'))
  all_tr_ind.append(train_data_indices)
  pool_indices = [i for i in train_ind if i not in train_data_indices]
  exp_acc.append(run_model(train_data_indices))
  for acq in range(num_acquisitions):
    new_ind_ind = np.random.choice(len(pool_indices))
    train_data_indices.append(pool_indices[new_ind_ind])
    del pool_indices[new_ind_ind]
    exp_acc.append(run_model(train_data_indices))
    all_acc[-1] = exp_acc
    all_tr_ind[-1] = train_data_indices
    print('all acc: ' + str(all_acc))
    print('all ind: ' + str(all_tr_ind))
    np.save(folder_path+'RandomAcqAcc.npy', np.array(all_acc))
    np.save(folder_path+'RandomAcqInd.npy', np.array(all_tr_ind))
print('All Acc: ' + str(all_acc))
