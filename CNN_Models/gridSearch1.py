import numpy as np
import tensorflow as tf
import csv
import pandas as pd
from xml.etree import ElementTree
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.callbacks import History
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

# loading data from CSV file and generating train-test sets
all_Data = pd.read_csv('all_Data.csv', header= None)
all_Data.head(5)
output_all = all_Data.iloc[:,-1]
train_Y = output_all.to_numpy()
#train_Y
input_all = all_Data.iloc[:,0:len(all_Data.columns)-1]
train_X = input_all.to_numpy()
#train_X
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20, random_state=42)
#X_train.shape
X_train = np.expand_dims(X_train, axis=2)
#X_train.shape
X_test = np.expand_dims(X_test, axis=2)
#X_test.shape

def create_model():
    #CNN Architecture - Model 7
    model = Sequential()
    model.add(Convolution1D(filters=10, kernel_size=12, activation="relu", kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=4, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=16, kernel_size=12, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=22, kernel_size=12, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=28, kernel_size=12, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=4, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=34, kernel_size=12, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=40, kernel_size=12, activation='relu', kernel_initializer="glorot_uniform"))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    #model.add(Dropout(0.35))
    model.add(Dense(130, activation='relu'))
    #model.add(Dropout(0.35))
    model.add(Dense(130, activation='relu'))
    model.add(Dense(1, activation='linear'))

    history = History()
    model.compile(loss='mean_squared_error',optimizer='Adam', metrics=[coeff_determination])
    #model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=400, batch_size=30, callbacks=[history])
    return model

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()))
# to reprduce the same results next time
seed = 7
np.random.seed(seed)
# Creating Keras model with Scikit learn wrap-up
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [64]
epochs = [100]
# Using make scorer to convert metric r_2 to a scorer
my_scorer = make_scorer(r2_score, greater_is_better=True)

# passing dictionaries of parameters to the GridSearchCV
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, scoring=my_scorer, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)
# summarizing the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

