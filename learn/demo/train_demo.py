import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from fundamental_domain_projections.dirichlet.dirichlet_dataset import DirichletDataset


def get_nn():
    inp = tf.keras.layers.Input(shape=(4, 26,))
    prep = tf.keras.layers.Reshape((4 * 26,))(inp)
    h1 = tf.keras.layers.Dense(100, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(50, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(50, activation='relu')(h2)
    out = tf.keras.layers.Dense(43, activation='softmax')(h3)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


def main():
    model = get_nn()

    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/raw')
    Xpath = os.path.join(datapath, 'v26_Xpermuted.npy')
    ypath = os.path.join(datapath, 'v26_y.npy')
    with open(f'{Xpath}', 'rb') as fx, open(f'{ypath}', 'rb') as fy:
        X = np.load(fx, allow_pickle=True)
        y = np.load(fy, allow_pickle=True)

    # if you want to take Dirichlet dataset
    diric_proj = DirichletDataset(X=X, matrix_dim=(4, 26), x0='Daniel', seeded_ascent=False, from_permuted=True,
                                  # save_proj=True, file_name='dirichlet_from_permuted',
                                  cutoff=-1)
    X = diric_proj.X_proj
    y = y[:-1]

    X_new = {}
    y_new = {}
    X_new['train'], X_new['test'], y_new['train'], y_new['test'] = train_test_split(X, y, test_size=0.5)
    model.fit(
        X_new['train'], y_new['train'],
        epochs=200,
        validation_data=(X_new['test'], y_new['test']),
    )


if __name__ == '__main__':
    main()

# output on permuted dataset
# Epoch 1/20
# 1227/1227 [==============================] - 2s 1ms/step - loss: 2.5478 - accuracy: 0.1487 - val_loss: 2.1064 - val_accuracy: 0.1907
# Epoch 2/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 2.0245 - accuracy: 0.2171 - val_loss: 2.0369 - val_accuracy: 0.2020
# Epoch 3/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.9275 - accuracy: 0.2404 - val_loss: 2.0002 - val_accuracy: 0.2156
# Epoch 4/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.8579 - accuracy: 0.2612 - val_loss: 2.0138 - val_accuracy: 0.2139
# Epoch 5/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.8168 - accuracy: 0.2713 - val_loss: 2.0184 - val_accuracy: 0.2156
# Epoch 6/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.7826 - accuracy: 0.2836 - val_loss: 1.9968 - val_accuracy: 0.2189
# Epoch 7/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.7414 - accuracy: 0.2983 - val_loss: 1.9994 - val_accuracy: 0.2206
# Epoch 8/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.7065 - accuracy: 0.3088 - val_loss: 2.0252 - val_accuracy: 0.2155
# Epoch 9/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6927 - accuracy: 0.3150 - val_loss: 2.0239 - val_accuracy: 0.2218
# Epoch 10/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6711 - accuracy: 0.3250 - val_loss: 2.0380 - val_accuracy: 0.2200
# Epoch 11/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6628 - accuracy: 0.3303 - val_loss: 2.0615 - val_accuracy: 0.2165
# Epoch 12/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6488 - accuracy: 0.3359 - val_loss: 2.0645 - val_accuracy: 0.2207
# Epoch 13/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6273 - accuracy: 0.3445 - val_loss: 2.0607 - val_accuracy: 0.2203
# Epoch 14/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.6106 - accuracy: 0.3557 - val_loss: 2.0900 - val_accuracy: 0.2226
# Epoch 15/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5919 - accuracy: 0.3604 - val_loss: 2.1125 - val_accuracy: 0.2212
# Epoch 16/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5794 - accuracy: 0.3647 - val_loss: 2.0992 - val_accuracy: 0.2223
# Epoch 17/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5664 - accuracy: 0.3696 - val_loss: 2.1226 - val_accuracy: 0.2191
# Epoch 18/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5597 - accuracy: 0.3710 - val_loss: 2.1356 - val_accuracy: 0.2153
# Epoch 19/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5612 - accuracy: 0.3718 - val_loss: 2.1457 - val_accuracy: 0.2189
# Epoch 20/20
# 1227/1227 [==============================] - 1s 1ms/step - loss: 1.5388 - accuracy: 0.3783 - val_loss: 2.1443 - val_accuracy: 0.2184


# output on unpermmuted  dataset:
# Epoch 1/20
# 1091/1091 [==============================] - 2s 1ms/step - loss: 2.2385 - accuracy: 0.1769 - val_loss: 1.7860 - val_accuracy: 0.2596
# Epoch 2/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.6828 - accuracy: 0.2987 - val_loss: 1.6647 - val_accuracy: 0.3042
# Epoch 3/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.5747 - accuracy: 0.3334 - val_loss: 1.6231 - val_accuracy: 0.3214
# Epoch 4/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.5049 - accuracy: 0.3626 - val_loss: 1.6255 - val_accuracy: 0.3167
# Epoch 5/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.4721 - accuracy: 0.3669 - val_loss: 1.6057 - val_accuracy: 0.3260
# Epoch 6/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.4412 - accuracy: 0.3889 - val_loss: 1.5877 - val_accuracy: 0.3367
# Epoch 7/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.4064 - accuracy: 0.3980 - val_loss: 1.6235 - val_accuracy: 0.3287
# Epoch 8/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.3791 - accuracy: 0.4078 - val_loss: 1.5937 - val_accuracy: 0.3385
# Epoch 9/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.3619 - accuracy: 0.4098 - val_loss: 1.5893 - val_accuracy: 0.3370
# Epoch 10/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.3426 - accuracy: 0.4217 - val_loss: 1.5819 - val_accuracy: 0.3421
# Epoch 11/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.3195 - accuracy: 0.4320 - val_loss: 1.5848 - val_accuracy: 0.3413
# Epoch 12/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.3081 - accuracy: 0.4433 - val_loss: 1.5928 - val_accuracy: 0.3432
# Epoch 13/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2821 - accuracy: 0.4508 - val_loss: 1.5896 - val_accuracy: 0.3464
# Epoch 14/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2686 - accuracy: 0.4560 - val_loss: 1.6062 - val_accuracy: 0.3467
# Epoch 15/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2661 - accuracy: 0.4531 - val_loss: 1.6193 - val_accuracy: 0.3494
# Epoch 16/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2504 - accuracy: 0.4588 - val_loss: 1.6206 - val_accuracy: 0.3462
# Epoch 17/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2289 - accuracy: 0.4721 - val_loss: 1.6167 - val_accuracy: 0.3500
# Epoch 18/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.2130 - accuracy: 0.4772 - val_loss: 1.6467 - val_accuracy: 0.3480
# Epoch 19/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.1963 - accuracy: 0.4884 - val_loss: 1.6493 - val_accuracy: 0.3492
# Epoch 20/20
# 1091/1091 [==============================] - 1s 1ms/step - loss: 1.1878 - accuracy: 0.4887 - val_loss: 1.6420 - val_accuracy: 0.3525