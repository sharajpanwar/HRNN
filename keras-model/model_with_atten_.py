from AttentionWithContext import AttentionWithContext
import numpy as np
from keras.layers import Dense, Dropout, Lambda
from keras.layers import LSTM, TimeDistributed, Reshape
import h5py
from keras.engine import Input, Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

# uploading training data
X_train = h5py.File('tr_drive_hrnn_1.mat')['tr_drive_hrnn_1']
X_train = np.transpose(X_train) #0.125 sec data
print(X_train.shape)
# uploading test data
X_test = h5py.File('te_drive_hrnn_1.mat')['te_drive_hrnn_1']
X_test = np.transpose(X_test)
print(X_test.shape)
#creating labels for training and test data sets.
Y_train = np.zeros(X_train.shape[0])
Y_test = np.zeros(X_train.shape[0])
Y_train[0:500] = 1 # first 500 samples are target samples
Y_test[0:250] = 1 # first 250 samples are target samples

def hrnn_model():

    input_section = Input(shape=(4096, ), dtype='float32') # 0.5 second flattened
    epoch_section = Reshape((64, 64), input_shape=(4096, ))(input_section) # 0.5 second epoch section
    lstm1 = LSTM(32, return_sequences=True, name='recurrent_layer')(epoch_section)
    dense1 = TimeDistributed(Dense(32))(lstm1)
    # TimeDistributed to apply a Dense layer to each of the 32 timesteps, independently
    attention = AttentionWithContext()(dense1) #apply attention
    model_section = Model(inputs=input_section, outputs=attention)
    model_section.summary() # summary of LSTM model applied on each 0.5 second of a 5 sec epoch

    ##################################################################################################################

    model_input = Input(shape=(10, 4096), dtype='float32')
    section_output = TimeDistributed(model_section)(model_input)
    # "TimeDistributed layer"for sequentially feeding each 0.5 second of 5 second EEG epoch,(each section of 10 sections of 4096)
    lstm2 = LSTM(32, return_sequences=True)(section_output)
    lstm3 = TimeDistributed(Dense(32))(lstm2)
    attention_2 = AttentionWithContext()(lstm3)
    model_output = Dense(1, activation='sigmoid')(attention_2)

    model = Model(inputs=model_input, outputs=model_output)
    optimizer = Adam(lr=0.01)

    print('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("model fitting - Hierachical LSTM")
    model.summary()
    return model

if __name__ == '__main__':
    model = hrnn_model()
    model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_split=0.1, validation_data=(X_train, Y_train))
    score, acc = model.evaluate(X_test, Y_test, batch_size=128)
    print('Test accuracy:', acc)
    target = model.predict(X_test, batch_size=128)
    auc = roc_auc_score(Y_test, target)
    print('AUC', auc)
