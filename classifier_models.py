from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM
from keras.models import Sequential, Model
from keras.layers.merge import Concatenate
import numpy as np
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
    # Your implementation starts here
    
    # end
    
    # Save model
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("trained_models/lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("trained_models/lstm_model.h5")
    print("Saved LSTM model to file")
    
    return eval_loss, eval_acc, model
