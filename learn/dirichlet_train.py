import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.parse_data import *
from pathlib import Path
from fundamental_domain_projections.dirichlet.dirichlet_dataset import *

MATRIX_DIM = (4, 26)
NUM_EPOCHS = 200

SAVE_MODEL = True
LOAD_MODEL = False
SAVE_RESULT = True
MODEL_NAME = 'model_simple_nn_seeded_ascent_200_epoch'

SAVE_PROJ = False
LOAD_PROJ = True
PROJ_FILE_NAME = 'dirichlet_proj_seeded_ascent'

def save_results_as_csv(results_dict, file_name):
    base_dir = Path(__file__).parents[1]
    rawpath = base_dir.joinpath('data/saved_results/' + file_name + '.csv')
    print('Saving results as csv in  %s' % rawpath)
    df = pd.DataFrame(results_dict)
    df.to_csv(rawpath)

def load_model(model_name):
    base_dir = Path(__file__).parents[1]
    rawpath = base_dir.joinpath('data/saved_models/' + model_name + '.h5')
    print('Loading model from %s' % rawpath)
    model = tf.keras.models.load_model(rawpath)
    print(model.summary())
    return model

def saved_model(model, model_name):
    base_dir = Path(__file__).parents[1]
    rawpath = base_dir.joinpath('data/saved_models/' + model_name + '.h5')
    print('Saving model to %s' % rawpath)
    tf.keras.models.save_model(model, rawpath)

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
    if LOAD_MODEL:
        model = load_model(MODEL_NAME)
    else:
        model = get_nn()

    if LOAD_PROJ:
        diric_proj = DirichletDataset(load_proj=True, file_name=PROJ_FILE_NAME)
        X_proj, Y = diric_proj.X_proj, diric_proj.Y
    else:
        X, Y = parse_txt_file()
        # X, Y = X[:200], Y[:200]  # taking a subset for quick test run
        diric_proj = DirichletDataset(X=X, Y=Y, matrix_dim=MATRIX_DIM, seeded_ascent=True, save_proj=SAVE_PROJ, file_name=PROJ_FILE_NAME)
        X_proj = diric_proj.X_proj

    X_new, Y_new = {}, {}
    X_new['train'], X_new['test'], Y_new['train'], Y_new['test'] = train_test_split(X_proj, Y, test_size=0.5)
    result = model.fit(
        X_new['train'], Y_new['train'],
        epochs=NUM_EPOCHS,
        validation_data=(X_new['test'], Y_new['test']),
    )

    if SAVE_MODEL:
        saved_model(model, MODEL_NAME)

    if SAVE_RESULT:
        save_results_as_csv(result.history, MODEL_NAME)


if __name__ == '__main__':
    main()