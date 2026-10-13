import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv1D, Activation, Multiply, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from Evaluation import evaluation, evaluate_error


def region_attention(x):
    """
    Applies region attention over the time dimension of Conv1D output.
    x: Input tensor of shape (batch, time_steps, channels)
    """
    # Compute channel-wise attention
    attention = GlobalAveragePooling1D()(x)  # shape -> (batch, channels)
    attention = tf.expand_dims(attention, axis=1)  # shape -> (batch, 1, channels)
    attention = tf.keras.activations.sigmoid(attention)  # attention weights between 0 and 1

    # Apply attention
    out = Multiply()([x, attention])  # element-wise multiplication
    return out


def build_tcn_model(Train_Data, num_classes, sol):
    TrainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    input_layer = Input(shape=(1, TrainX.shape[2]))

    # TCN Convolution
    x = Conv1D(sol[0], kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(input_layer)

    # Apply Region Attention
    x = region_attention(x)

    # Output layers
    x = Conv1D(num_classes.shape[1], kernel_size=1, padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    output_layer = Activation('sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, StepPerEpochs, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    model = build_tcn_model(Train_Data, Train_Target, sol)
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    TrainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    TestX = np.reshape(Test_Data, (Test_Data.shape[0], 1, Test_Data.shape[1]))
    trainX = np.asarray(TrainX).astype(np.float32)
    testX = np.asarray(TestX).astype(np.float32)

    model.fit(trainX, Train_Target, epochs=sol[2], steps_per_epoch=StepPerEpochs, batch_size=64)
    pred = model.predict(testX)
    Eval = evaluate_error(pred, Test_Target)
    return Eval, pred


