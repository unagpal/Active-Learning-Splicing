
import numpy as np
import keras
from keras.datasets import mnist
import sys
import scipy
from gurobipy import *
from sklearn import metrics
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K


#Robust KCenter MIP
def kcent_feasibility_check (b, size_train, pairwise_distances, delta, Xi):
  """Solving robust k-Center MIP in order to see if robust k-Center cost of delta is feasible
  #Arguments:
    b: batch size
    size_train: size of training set
    pairwise_distances: distances between all (i.e. both training and candidate) embeddings used in robust k-Center
    delta: k-Center cost for which feasibility is being determined
    Xi: total outlier tolerance for robust k-Center
  #Returns:
    Indices of points selected as centers and whether the resulting robust k-Center cost is <= delta
  """
  print("Beginning MIP with delta = " + str(delta))
  total_points = pairwise_distances.shape[0]
  #Creating all_outlier_tuples, a list of tuples (i,j) whre dist(embed(x_i), embed(x_j))>delta
  all_outlier_tuples = []
  for i in range(total_points):
    for j in range(total_points):
      if pairwise_distances[i,j] > delta:
        all_outlier_tuples.append((i,j))
  #Code begins for MIP
  model = Model("k-center")
  u = {}
  for point_ind in range(total_points):
    if point_ind == 0:
      u[point_ind] = model.addVar(lb=0.0, ub=1.0, obj=1.0, vtype=GRB.BINARY, name='u[%i]' % point_ind)
    else:
      u[point_ind] = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name='u[%i]' % point_ind)
  w = {}
  xi = {}
  for i in range(total_points):
    for j in range(total_points):
      w_var_name = 'w[' + str(i)+ ',' + str(j) + ']'
      xi_var_name = 'xi['+ str(i)+ ',' + str(j) + ']'
      w[i,j] = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name=w_var_name)
      xi[i,j] = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name=xi_var_name)
  #Setting constraint: total centers = b + size_train 
  total_centers_coeff = [1 for i in range(total_points)]
  total_centers_var_list = [u[j] for j in range(total_points)]
  model.addConstr(LinExpr(total_centers_coeff, total_centers_var_list), GRB.EQUAL, size_train + b)
  #Setting constraints that all training u_i = 1
  for train_ind in range(size_train):
    model.addConstr(u[train_ind], GRB.EQUAL, 1)
  #Setting constraint that total outliers xi <= Xi
  total_outliers_coeff = [1 for i in range(total_points * total_points)]
  all_xi_var_list = []
  for i in range(total_points):
    for j in range(total_points):
      all_xi_var_list.append(xi[i,j])
  model.addConstr(LinExpr(total_outliers_coeff, all_xi_var_list), GRB.LESS_EQUAL, Xi)
  #Setting constraints that each point is assigned to one center
  one_center_coeff = [1 for i in range(total_points)]
  for i in range(total_points):
    w_list = []
    for j in range(total_points):
      w_list.append(w[i,j]) 
    model.addConstr(LinExpr(one_center_coeff, w_list), GRB.EQUAL, 1)
  #Checking w_{i,j} has points only assigned to the corresponding center
  for i in range(total_points):
    for j in range(total_points):
      model.addConstr(w[i,j], GRB.LESS_EQUAL, u[j])
  #Constraints specifying xi as denoting outliers (i.e. xi must be 1 if dist > delta)
  for outlier_tuple in all_outlier_tuples:
    model.addConstr(w[outlier_tuple[0], outlier_tuple[1]], GRB.EQUAL, xi[outlier_tuple[0], outlier_tuple[1]])
  #print('done creating all model constraints') 
  model.update()
  print('done calling model update, now calling optimize')
  model.optimize()
  all_vars= {}
  returned_vars = {}
  all_center_indices = []
  true_kcent_cost = 1e10
  if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
    print("Infeasible")
  else:
    #print("Feasible; now saving solution")
    for v in model.getVars():
      all_vars[v.varName] = v.x
      if 'u' in v.varName:
        if v.x == 1.0:
          center_num = ""
          for char in list(v.varName):
            if char.isdigit():
              center_num += char
          all_center_indices.append(int(center_num)) 
        returned_vars[v.varName] = v.x
    #print("Outputs: " + str(all_vars))
    all_remaining_indices = np.array([i for i in range(total_points) if i not in all_center_indices])
    remaining_embedding_dist = np.zeros((len(all_center_indices), len(all_remaining_indices))) 
    center_axis_ind = 0
    remaining_axis_ind = 0
    for center_ind in all_center_indices:
      remaining_axis_ind = 0
      for remaining_ind in all_remaining_indices:
        remaining_embedding_dist[center_axis_ind, remaining_axis_ind] = pairwise_distances[center_ind, remaining_ind]
        remaining_axis_ind += 1
      center_axis_ind += 1
    true_kcent_distances = np.amin(remaining_embedding_dist, axis=0)
    true_kcent_cost = true_kcent_distances[np.flip(np.argsort(true_kcent_distances))[int(Xi)]]  
    #print(all_center_indices)
    #print(all_remaining_indices)
    print('true kcent cost: ' + str(true_kcent_cost))
  #print(all_center_indices)
  return [all_center_indices, true_kcent_cost<=delta]


def farthest_first_kcenters (labeled_embeddings, embedding_arr, k):
  """Initialization of robust k-Center via greedy farthest-first traversal
  #Arguments:
    labeled_embeddings: embeddings for training points used in robust k-Center (shape: [size_train, embedding_dim])
    embedding_arr: embeddings for pool candidate points used in robust k-Center (shape: [num_candidates, embedding_dim])
    k: batch size
  #Returns:
    k-Center cost and selected embeddings resulting from the farthest-first traversal
  """
  #Greedily selecting k embeddings
  selected_embeddings = []
  num_labeled_embeddings = len(labeled_embeddings)
  chosen_embeddings = list(labeled_embeddings)
  remaining_embeddings = list(embedding_arr)
  center_count = 0
  while center_count < k:
    all_distances = np.zeros((len(chosen_embeddings), len(remaining_embeddings)))
    for c in range(len(chosen_embeddings)):
      for remaining_ind in range(len(remaining_embeddings)):
        all_distances[c][remaining_ind] = np.linalg.norm(remaining_embeddings[remaining_ind]-chosen_embeddings[c])
    set_distances = np.amin(all_distances, axis=0)
    chosen_embedding = remaining_embeddings[np.argmax(set_distances)].tolist()
    del remaining_embeddings[np.argmax(set_distances)]
    chosen_embeddings.append(chosen_embedding)
    selected_embeddings.append(chosen_embedding)
    center_count += 1
  #Calculating cost
  all_final_distances = np.zeros((len(selected_embeddings), len(remaining_embeddings)))
  for c in range(len(selected_embeddings)):
    for remaining_ind in range(len(remaining_embeddings)):
      all_final_distances[c][remaining_ind] = np.linalg.norm(remaining_embeddings[remaining_ind]-selected_embeddings[c])
  all_center_dist = np.amin(all_final_distances, axis=0)
  greedy_cost = np.amax(all_center_dist)
  return [greedy_cost, selected_embeddings]

#Used for calculating test acc (average of MC dropout predictions)
def predict_with_uncertainty(f, x, n_iter=100):
    """Function generating non-deterministic predictions using MC dropout and returning the mean of these predictions
    Adapted from: https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras
    #Arguments:
        f: function mapping model input and Keras backend learning_phase flag to model output
        x: input
        n_iter: number of repreated MC dropout predictions per point
    #Returns:
        Mean of MC dropout predictions
    """
    result = np.zeros((n_iter,x.shape[0], 10))
    for i in range(n_iter):
        predictions = np.array(f((x, 1))[0])
        result[i,:, :] = predictions
    prediction = result.mean(axis=0)
    return prediction

def run_model (train_data_indices):
  """Trains a Keras model with the training points specified by train_data_indices and evaluates model on test data
  #Arguments:
      train_data_indices: indices of current training points within X_train_All
  #Returns:
      Test accuracy and Keras model
  """
  X_train = np.expand_dims(X_train_All[train_data_indices], axis=1)
  y_train = y_train_All[train_data_indices]
  y_train = keras.utils.to_categorical(y_train, num_classes=10)
  train_size = y_train.shape[0]
  Weight_Decay = 0.01/train_size
  dropout_prob = 0.25
  nb_filters = 40
  nb_pool = 3
  nb_conv = 4
  img_rows = img_cols = 28
  nb_classes = 10
  model = Sequential()
  model.add(Convolution2D(nb_filters, kernel_size=nb_conv, strides=1, data_format="channels_first", input_shape=(1, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(Convolution2D(nb_filters, kernel_size=nb_conv, strides=2, data_format = "channels_first"))
  model.add(Activation('relu'))
	
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format="channels_first"))
  model.add(Dropout(dropout_prob))

  model.add(Flatten())
  model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
  model.add(Activation('relu'))
  model.add(Dropout(dropout_prob))
  model.add(Dense(nb_classes, W_regularizer=l2(Weight_Decay)))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam')
  model.fit(X_train, y_train, epochs=300, batch_size=128, verbose=0)
  f = K.function([model.layers[0].input,K.learning_phase()],[model.layers[-1].output])
  y_test_output = predict_with_uncertainty(f, X_test, n_iter=100)
  y_test_predictions = np.argmax(y_test_output, axis=1)
  print('y test predictions shape: ' + str(y_test_predictions.shape))
  return [np.sum(y_test_predictions==y_test_original)/(y_test_original.shape[0]), model]

#Active learning parameters/settings
#Experiment starts with balance training set of 30 points

dropout_prob = 0.25
num_experiments = 3
#Proportion of unlabeled points set to be the robust k-Center outlier tolerance
outlier_proportion = 0.001

binary_search_itr = 5
num_acquisitions = 1000 
num_candidates = 2000
num_masks = 50
embedding_dim = 128
batch_size = 100

#Keras Model Parameters
num_classes = 10
nb_filters = 40
nb_pool = 2
nb_conv = 4
img_rows = img_cols = 28

#Loading data
data_path = "/gpfs/commons/home/unagpal/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
		  delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
		 delimiter=",") 
y_train_All = train_data[:,0]
y_test = test_data[:,0]
X_train_All = train_data[:,1:].reshape((60000,28,28))
X_test = test_data[:,1:].reshape((10000,28,28)) 
train_ind = np.arange(60000)
test_ind = np.arange(10000)

y_test_original = y_test
y_test = keras.utils.to_categorical(y_test, num_classes=10)
X_test = np.expand_dims(X_test[test_ind], axis=1)
folder_path = "/gpfs/commons/home/unagpal/CoreSetMIP/MNISTMultiClass/"
out_folder_path = folder_path

print('Running')	  
all_tr_ind = []
all_acc = []
#Iterating across active learning experiments, each of which starts with a different initial training set
for e in range(num_experiments):
  acc_file = "RobustKCenterAccBS"+str(batch_size)+"Ind"+str(e+1)+".npy"
  ind_file = "RobustKCenterIndBS"+str(batch_size)+"Ind"+str(e+1)+".npy"
  exp_acc = []
  #exp_acc = list(np.load(out_folder_path+acq_file))
  
  #Initial training/accuracy
  train_data_indices = list(np.load(folder_path + 'trainindices' + str(e+1) + '.npy'))
  print('Initial train size: ' + str(len(train_data_indices)))
  #train_data_indices = list(np.load(out_folder_path+ind_file))
  pool_indices = [i for i in train_ind if i not in train_data_indices]
  model_results = run_model(train_data_indices)
  print('trained model')
  exp_acc.append(model_results[0])
  num_acquisitions = num_acquisitions - batch_size * (len(exp_acc)-1)
  #all_acc.append(exp_acc)
  #all_tr_ind.append(train_data_indices)
  print('Initial Acc: ' + str(exp_acc))

  #Looping over active learning iterations
  for acq in range(num_acquisitions//batch_size):
    #Sampling pool candidates and obtaining current model performance
    pool_ind_sample = np.random.choice(pool_indices, num_candidates, replace=False)
    curr_size_train = len(train_data_indices)
    model = model_results[1]
    #Obtaining embeddings used in robust k-Center acquisition (activations of penultimate layer)
    X_k_center = np.expand_dims(np.concatenate((X_train_All[train_data_indices], X_train_All[pool_ind_sample])), axis=1)
    f_embed = K.function([model.layers[0].input, K.learning_phase()],
                    [model.layers[-4].output])
    all_embeddings = np.array(f_embed((X_k_center, 0))).reshape((X_k_center.shape[0], embedding_dim))
    #Initializing upper bound ub and lower bound ub for binary search on robust k-Center cost
    [cost, chosen_embeddings] = farthest_first_kcenters (all_embeddings[0:curr_size_train, :], all_embeddings[curr_size_train:, :], batch_size)
    
    all_acq_ind = []
    chosen_embeddings = np.array(chosen_embeddings)    
    for embedding in chosen_embeddings:
      for all_embed_ind in range(len(all_embeddings)):
        if np.all(embedding == all_embeddings[all_embed_ind]):
          all_acq_ind.append(pool_ind_sample[all_embed_ind - curr_size_train])
    ub = cost
    lb = cost/2
    pairwise_distances = metrics.pairwise_distances(all_embeddings)
    #Binary search on robust k-Center cost
    for bin_search_itr in range(binary_search_itr):
      [center_indices, validity] = kcent_feasibility_check(batch_size,curr_size_train, pairwise_distances, (ub+lb)/2.0, outlier_proportion*num_candidates)
      if validity==False and len(center_indices)>0:
        print('Gurobi solution found to be invalid')
      if validity == False:
        print('KCenter solution invalid or infeasible')
        lb = np.amin(pairwise_distances[pairwise_distances >= (ub+lb)/2.0])  
      else:
        print('Valid KCenter solution found')
        all_acq_ind = []
        ub = np.amax(pairwise_distances[pairwise_distances <= (ub+lb)/2.0])
        for center_num in center_indices:
          if center_num >= curr_size_train:
            all_acq_ind.append(pool_ind_sample[center_num - curr_size_train])
    print('len all_acq_ind: ' + str(len(all_acq_ind)))
    #Acquiring the selected points
    for acq_ind in all_acq_ind:
      train_data_indices.append(acq_ind)
      pool_indices.remove(acq_ind)
    #Training model and saving results
    model_results = run_model(train_data_indices)
    exp_acc.append(model_results[0])
    print('len train ind: ' + str(len(train_data_indices)))
    print('len pool ind: ' + str(len(pool_indices)))
    np.save(out_folder_path+acc_file, np.array(exp_acc))
    np.save(out_folder_path+ind_file, np.array(train_data_indices))
    print('All Acc: ' + str(all_acc))
    #print('All Ind: ' + str(all_tr_ind))
  print('All Acc: ' + str(all_acc))



