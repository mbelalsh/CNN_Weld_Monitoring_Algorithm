# bad Optimization
import numpy as np
import pandas as pd
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
from kerastuner import RandomSearch
from kerastuner import Objective
from kerastuner.engine.hyperparameters import HyperParameters

# loading data from CSV file and generating train-test sets
all_Data = pd.read_csv('all_Data.csv', header= None)
#all_Data.head(5)
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

def build_model(hp):
    #CNN Architecture - Model 7
    model = Sequential()
    model.add(Convolution1D(filters=hp.Int('conv1_filter', min_value=5, max_value=50, step=5),
                            kernel_size=hp.Choice('conv1_kernel', values=[8,10,12,14]), activation="relu",
                            kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=4, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=hp.Int('conv2_filter', min_value=5, max_value=100, step=5),
                            kernel_size=hp.Choice('conv2_kernel', values=[8,10,12,14]), activation="relu",
                            kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Convolution1D(filters=hp.Int('conv3_filter', min_value=5, max_value=100, step=5),
                            kernel_size=hp.Choice('conv3_kernel', values=[8,10,12,14]), activation="relu",
                            kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    #model.add(Dropout(0.35))
    model.add(Dense(130, activation='relu'))
    #model.add(Dropout(0.35))
    model.add(Dense(130, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',optimizer= Adam(hp.Choice('lr', values=[0.001,0.0001,0.00001])), metrics=[coeff_determination])
    return model

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

#hp = HyperParameters()
#model = build_model(hp)
tuner_search = RandomSearch(build_model, objective=Objective("coeff_determination", direction="max"), max_trials=2)
tuner_search.search(X_train, y_train, epochs=100, validation_split=0.1)