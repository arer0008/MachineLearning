from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM
from keras.models import Sequential, Model
from keras.layers.merge import Concatenate
import numpy as np
import os
# To make sure results are reproducible
np.random.seed(0)

def cnn_model(embedding_weights, max_features, \
              X_train_pad, y_train, X_test_pad, y_test, vocabulary_inv):
    """
    NOTE: build a cnn model.
    Target: build a classifier to classify positive/negative review.
    Inputs:
        embedding_weights: pre-trained embedding weights
        max_features: number of features X_train_pad: input data after padding
        y_train: label of training data
        X_test_pad: testing data after padding
        y_test: label of testing data
        vocabulary_inv: inversed index of vocaburary
    Outputs:
        model: cnn model
    """
    model = None
    eval_loss, eval_acc = 0.0, 0.0
    # Your implementation starts from here

    maxlen = 400
    batch_size = 32
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 6



    model = Sequential()
    model.add(Embedding(max_features, 300, input_length=100))
    model.add(Dropout(0.2))
    model.add(Convolution1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)
    eval_loss, eval_acc = model.evaluate(X_test_pad, y_test, batch_size=32)

    # End of student implementation

    # Save model
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("trained_models/cnn_model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("trained_models/cnn_model.h5")
    print("Saved CNN model to file")
    return eval_loss, eval_acc, model


def lstm_model(max_features, X_train_pad, y_train, X_test_pad, y_test):
    """
    NOTE: build LSTM model
    Inputs:
        max_features: number of features
        X_train_pad: input data after padding
        y_train: label of training data
        X_test_pad: testing data after padding
        y_test: label of testing data
    Outputs:
        eval_score: evaluated score
        eval_acc: evaluated accuracy
        model: lstm model
    """

    model = Sequential()
    model.add(Embedding(40000, 300))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_pad, y_train, batch_size=32, epochs=3, validation_data=(X_test_pad, y_test))
    eval_loss, eval_acc = model.evaluate(X_test_pad, y_test, batch_size=32)


    # Save model
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("trained_models/lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("trained_models/lstm_model.h5")
    print("Saved LSTM model to file")

    return eval_loss, eval_acc, model
