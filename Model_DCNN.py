import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Sequential
from Evaluation import evaluation, evaluate_error


def Model_DCNN(trainX, trainy, testX, testy):
    dropout = 0.5
    hN = 100
    train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(hN, activation='relu'))
    model.add(Dense(trainy.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.summary()
    model.fit(train, trainy, epochs=50, batch_size=4)
    pred = model.predict(test)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(testy, pred)
    return Eval, pred


def Model_IDCNNPred(trainX, trainy, testX, testy, StepPerEpochs):
    dropout = 0.5
    hN = 100
    train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(hN, activation='relu'))
    model.add(Dense(trainy.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.summary()
    model.fit(train, trainy, epochs=100, steps_per_epoch=StepPerEpochs, batch_size=4)
    pred = model.predict(test)
    Eval = evaluate_error(pred, testy)
    return Eval, pred

