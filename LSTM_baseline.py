# lstm model under



from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot




def lstm_input_reshape(data_transposed): 
    # data_transposed has the signal channels in raws and timesetps in columns
    # this for me a good way to reshape the data and inject it in the lstm network
    
    y_train= data_transposed.iloc[-1]
    x_train= data_transposed.iloc[0:len(data_transposed[1:]),]


    x_train_data= x_train.as_matrix(columns=None)
    y_train_data= y_train.as_matrix(columns=None)
    x_train_data= x_train_data.reshape(x_train_data.shape[0],x_train_data.shape[1],1)
    y_train_data= y_train_data.reshape((1,y_train_data.shape[0],1))
    
    return x_train_data, y_train_data
# lstm_input_reshape is not necessary to run the lstm


def lstm_baseline(x_train_data, y_train_data):
    
    
    #num_classes = 6
    #batch_size=128

    
    model = Sequential()
    model.add(LSTM(128, input_shape=x_train_data.shape[1:],
                   activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    model.fit(x_train_data,y_train_data, epochs= 15)
    
    return model

