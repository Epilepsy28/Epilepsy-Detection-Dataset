import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from Evaluation import evaluate_error
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Dense, Dropout, Flatten, Input, concatenate
)
from tensorflow.keras.models import Model

from Evaluation import evaluation


def inception_module(x, f1, f3_in, f3, f5_in, f5, pool_proj):
    # 1x1 conv branch
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    # 1x1 -> 3x3 conv branch
    conv3 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(x)
    conv3 = Conv2D(f3, (3, 3), padding='same', activation='relu')(conv3)

    # 1x1 -> 5x5 conv branch
    conv5 = Conv2D(f5_in, (1, 1), padding='same', activation='relu')(x)
    conv5 = Conv2D(f5, (5, 5), padding='same', activation='relu')(conv5)

    # pooling -> 1x1 conv branch
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = Conv2D(pool_proj, (1, 1), padding='same', activation='relu')(pool)

    # Concatenate filters
    output = concatenate([conv1, conv3, conv5, pool], axis=3)
    return output


def GoogLeNet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Stage 2
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Stage 3 - Inception
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Stage 4
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Stage 5
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    # Global Average Pooling
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def Model_GoogleNet_CNN(Train_Data, Train_Target, Test_Data, Test_Target):
    train = np.reshape(Train_Data, (Train_Data.shape[0], 1, 1, Train_Data.shape[1]))
    test = np.reshape(Test_Data, (Test_Data.shape[0], 1, 1, Test_Data.shape[1]))
    model = GoogLeNet((train.shape[1], train.shape[2], train.shape[3]), Train_Target.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.summary()
    model.fit(train, Train_Target, epochs=100, batch_size=4)
    pred = model.predict(test)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred


def Model_CNN_Pred(Train_Data, Train_Target, Test_Data, Test_Target, Step_Per_Epochs):

    X_train = Train_Data.reshape((Train_Data.shape[0], Train_Data.shape[1], 1))
    X_test = Test_Data.reshape((Test_Data.shape[0], Test_Data.shape[1], 1))

    # Define the CNN model
    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(Train_Target.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()
    TrainX = np.asarray(X_train).astype(np.float32)
    TestX = np.asarray(X_test).astype(np.float32)
    # Train the model
    model.fit(TrainX, Train_Target, epochs=100, steps_per_epoch=Step_Per_Epochs, batch_size=4)
    pred = model.predict(TestX)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))
    Eval = evaluate_error(pred, Test_Target)

    return Eval, pred


