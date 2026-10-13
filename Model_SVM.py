import numpy as np
from sklearn.svm import SVR

from Evaluation import evaluation


def Model_SVM(train_data, train_target, test_data, test_target):
    model = SVR(kernel='linear', epsilon=100)
    model.fit(train_data, train_target[:, 0])
    pred = model.predict(test_data)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(test_target[:, 0], pred)
    return Eval, pred


