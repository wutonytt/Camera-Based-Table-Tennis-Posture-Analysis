from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping


def lstm(shape_1):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape_1,1), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    model.summary()
    return model


def splitdata(train):
    X = train.iloc[:, 2:-1]
    y = train.iloc[:, -1].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    return X_train, X_val, y_train, y_val


def fit(model_type, kernel_type, X_train, y_train):
    if model_type == 'LSTM':
        X_train_np = X_train.to_numpy()
        X_train_np = np.reshape(X_train_np, (X_train_np.shape[0], X_train_np.shape[1], 1))
        model = lstm(X_train_np.shape[1])
        train_history = model.fit(X_train_np, y_train, epochs=10, batch_size=10)
        return model
    
    elif model_type == 'SVM':
        svm = SVC(kernel = kernel_type, probability = True)
        svm.fit(X_train, y_train)
        return svm


def prediction(model_type, model, X_val, y_val):
    if model_type == 'LSTM':
        X_val_np = X_val.to_numpy()
        X_val_np = np.reshape(X_val_np, (X_val_np.shape[0], X_val_np.shape[1], 1))
        y_pred = model.predict_classes(X_val_np)
        
    elif model_type == 'SVM':
        y_pred = model.predict(X_val)
        
    print(classification_report(y_val, y_pred))
    return


def train(model_type, kernel_type, train):
    X_train, X_val, y_train, y_val = splitdata(train)
    model = fit(model_type, kernel_type, X_train, y_train)
    prediction(model_type, model, X_val, y_val)
    return model