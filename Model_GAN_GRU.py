import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Lambda
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation


def NDAGAT_layer(X, sol, dilation_rate=2, num_heads=4):
    _, _, feat = X.shape

    w_query = [tf.Variable(tf.random.normal([feat, feat]), trainable=True, name=f"w_query_{h}")
               for h in range(sol[0])]
    w_key = [tf.Variable(tf.random.normal([feat, feat]), trainable=True, name=f"w_key_{h}")
             for h in range(num_heads)]
    w_val = [tf.Variable(tf.random.normal([feat, feat]), trainable=True, name=f"w_val_{h}")
             for h in range(num_heads)]

    def ndagat(x):
        outputs = []
        for h in range(num_heads):
            Q = tf.tensordot(x, w_query[h], axes=1)
            K = tf.tensordot(x, w_key[h], axes=1)
            V = tf.tensordot(x, w_val[h], axes=1)

            K_shift = tf.roll(K, shift=dilation_rate * h, axis=1)
            attn = tf.nn.softmax(tf.matmul(Q, K_shift, transpose_b=True) / tf.sqrt(float(feat)))
            out = tf.matmul(attn, V)
            outputs.append(out)

        return tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)

    return Lambda(ndagat)(X)


def grad_cam(model, input_data, target_class_index):
    last_gru_layer = None
    for layer in model.layers:
        if isinstance(layer, GRU):
            last_gru_layer = layer

    grad_model = Model([model.inputs],
                       [last_gru_layer.output, model.output])

    with tf.GradientTape() as tape:
        gru_out, preds = grad_model(input_data)
        loss = preds[:, :, target_class_index]

    grads = tape.gradient(loss, gru_out)
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(tf.multiply(gru_out, tf.expand_dims(weights, 1)), axis=2)

    cam = tf.nn.relu(cam)
    cam /= tf.reduce_max(cam) + 1e-10

    return cam.numpy()


def Model_NDAGAT_GRU(X_train, y_train, X_test, Y_test, sol=None):
    if sol is None:
        sol = [5, 0.01, 5, 0.01]

    trainX = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    testX = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    inp = Input(shape=(1, trainX.shape[2]))
    x = NDAGAT_layer(inp, sol, dilation_rate=2, num_heads=4)
    x = Dropout(0.2)(x)
    x = GRU(units=sol[2], return_sequences=True, activation='relu', name="gradcam_gru")(x)
    x = Dropout(0.2)(x)
    # ----- Dense output with GRU -----
    x = GRU(units=y_train.shape[1], return_sequences=True, activation="sigmoid",
            name="final_output_gru")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(learning_rate=sol[3]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    model.fit(trainX.astype(np.float32), y_train,
              epochs=100, batch_size=64, verbose=1)
    pred = model.predict(testX.astype(np.float32))
    pred = pred.reshape(pred.shape[0], pred.shape[2])
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Y_test, pred)

    # ---- Grad-CAM ----
    gradcam_maps = []
    for i in range(testX.shape[0]):
        cam = grad_cam(model, testX[i:i + 1], target_class_index=np.argmax(pred[i]))
        gradcam_maps.append(cam)

    return Eval, pred, np.array(gradcam_maps)

