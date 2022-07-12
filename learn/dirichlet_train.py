import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.parse_data import *
from fundamental_domain_projections.dirichlet.dirichlet_dataset import *



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
    X, Y = parse_txt_file()
    X, Y = X[:2000], Y[:2000]  # taking a subset for quick test run
    dirc_proj = DirichletDataset(X, Y, (4, 26))
    X_new = {}
    Y_new = {}
    X_new['train'], X_new['test'], Y_new['train'], Y_new['test'] = train_test_split(dirc_proj.X_proj, dirc_proj.Y, test_size=0.5)
    model.fit(
        X_new['train'], Y_new['train'],
        epochs=20,
        validation_data=(X_new['test'], Y_new['test']),
    )


if __name__ == '__main__':
    main()