from src.dataset import KreuzerSkarkeDataset

from keras import layers, models, optimizers, callbacks
from keras import backend as K

from pathlib import Path


class mini_model:
    def __init__(self, dataset, one_hot_encoded = False):
        self.one_hot_encoded = one_hot_encoded
        self.dataset = dataset
        self.batch_size = 32

        self.X = {}
        self.y = {}
        self.X['train'], self.y['train'] = self.dataset.X_train, self.dataset.Y_train
        self.X['test'], self.y['test'] = self.dataset.X_test, self.dataset.Y_test
        self.initialise_model()

    def initialise_model(self):
        inp = layers.Input(shape=(4, 24, 1))
        p2 = layers.Reshape((4*24,))(inp)
        fc1 = layers.Dense(256, activation='relu')(p2)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        fc3 = layers.Dense(64, activation='relu')(fc2)
        out = layers.Dense(35, activation='linear')(fc3)

        self.model = models.Model(inputs=inp, outputs=out)
        if self.one_hot_encoded:
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizers.Adam(0.001),
                metrics=['categorical_accuracy'],
            )
        else:
            self.model.compile(
                loss='mean_squared_error',
                optimizer=optimizers.Adam(0.001),
                metrics=[self.soft_acc],
            )

    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def get_model(self):
        return self.model

    def train(self, save_csv: bool = True,
            num_epochs: int = 300):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param num_epochs: Train for how many epochs
        :return: None
        '''

        # if self.output_tag is None:
        #     self.output_tag = 'hartford_' + self.dataset.projections_file
        # log_dir = TENSORBOARD_DIR.joinpath(self.output_tag)
        history = self.model.fit(
            x=self.X['train'],
            y=self.y['train'],
            batch_size=self.batch_size,
            epochs=num_epochs,
            validation_data=(self.X['test'], self.y['test']),
            callbacks=[#callbacks.TensorBoard(log_dir=log_dir),
                       callbacks.EarlyStopping(monitor="loss", patience=12, verbose=1, restore_best_weights=True)],
        )
        # if self.output_tag is not None:
        #     saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.h5')
        #     print('Saving final model checkpoint to %s for %d epochs ' % (saved_model_path, num_epochs))
        #     self.model.save_weights(saved_model_path)  # creates a HDF5 file 'my_model.h5'
        #
        # if save_csv:
        #     saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')
        #     results_df = pd.DataFrame(history.history)
        #     print('Saving results as a csv in  %s' % saved_results_path)
        #     results_df.to_csv(saved_results_path)

        return history

    def get_accuracy(self):
        return self.model.evaluate(self.X['test'], self.y['test'], batch_size=128)[1]
        # evaluate gives [L2 loss, soft acc]

def create_different_number_vertices_dataset():
    BASE_DIR = Path(__file__).parents[1]

    this_input = BASE_DIR.joinpath('data/raw/v24')
    dataset = KreuzerSkarkeDataset(input_file=this_input, save_projections=True, projections_file='v24original')
    dataset = KreuzerSkarkeDataset(input_file=this_input, apply_random_permutation=True, save_projections=True, projections_file='v24original_permuted')
    dataset = KreuzerSkarkeDataset(input_file=this_input, projection='dirichlet', save_projections=True, projections_file='v24dirichlet')
    dataset = KreuzerSkarkeDataset(input_file=this_input, projection='dirichlet', apply_random_permutation=True, save_projections=True, projections_file='v24dirichlet_permuted')


def main():
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='v24dirichlet')
    X_proj, Y = dataset.X_proj, dataset.Y
    X_train, Y_train = dataset.X_train, dataset.Y_train
    X_test, Y_test = dataset.X_test, dataset.Y_test
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    my_model = mini_model(dataset)
    my_model.train(num_epochs=100)
    # original: loss: 0.4821 - soft_acc: 0.5520 - val_loss: 0.5272 - val_soft_acc: 0.5389
    # original permuted: loss: 2.3401 - soft_acc: 0.2680 - val_loss: 2.7440 - val_soft_acc: 0.2497
    # dirichlet: loss: 0.4203 - soft_acc: 0.5910 - val_loss: 0.4933 - val_soft_acc: 0.5620

if __name__ == '__main__':
    main()
