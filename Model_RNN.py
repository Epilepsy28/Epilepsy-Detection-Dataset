import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

from Evaluation import evaluate_error


def Model_RNN(train_data, train_target, test_data, test_target, Step_per_Epoch):
    pred, model = RNN_train(train_data, train_target, test_data, test_target, Step_per_Epoch)

    Eval = evaluate_error(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX, testY, Step_per_Epoch):
    train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, train.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    TrainX = np.zeros((train.shape[0], train.shape[1], train.shape[2]))
    for k in range(train.shape[0]):
        for j in range(train.shape[2]):
            val = train[k, :, j][0]
            TrainX[k, :, j] = np.asarray(val).astype(np.float32)
    TestX = np.zeros((test.shape[0], test.shape[1], test.shape[2]))
    for k in range(test.shape[0]):
        for j in range(test.shape[2]):
            val = test[k, :, j][0]
            TestX[k, :, j] = np.asarray(val).astype(np.float32)
    model.fit(TrainX, trainY, epochs=100, steps_per_epoch=Step_per_Epoch)
    testPredict = model.predict(TestX)
    return testPredict, model


