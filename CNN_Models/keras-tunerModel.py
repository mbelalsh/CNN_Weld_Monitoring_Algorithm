import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#CNN Architecture - Model 13 Keras Optimized model hpTuning 4b after 100 trials
model13 = Sequential()
model13.add(Convolution1D(filters=15, kernel_size=10, activation="relu", kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
model13.add(MaxPooling1D(pool_size=7, strides=2))
model13.add(BatchNormalization())
model13.add(Convolution1D(filters=13, kernel_size=14, activation="relu", kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
model13.add(MaxPooling1D(pool_size=7, strides=2))
model13.add(BatchNormalization())
model13.add(Convolution1D(filters=9, kernel_size=10, activation="relu", kernel_initializer="glorot_uniform", input_shape=(X_train.shape[1],1)))
model13.add(MaxPooling1D(pool_size=4, strides=2))
model13.add(BatchNormalization())
model13.add(Flatten())
#model.add(Dropout(0.35))
model13.add(Dense(50, activation='relu', kernel_initializer="glorot_uniform"))
#model.add(Dropout(0.35))
model13.add(Dense(100, activation='relu', kernel_initializer="glorot_uniform"))
model13.add(Dense(1, activation='linear'))

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()))
history = History()
model13.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001), metrics=[coeff_determination])
model13.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=200, batch_size=64, callbacks=[history])

model13.save('keras-tunerModel.h5')

# Plot training & validation accuracy values
plt.plot(history.history['coeff_determination'])
plt.plot(history.history['val_coeff_determination'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()