from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import layers, Model, optimizers
import tensorflow as tf

from src.dataset import *


class Vanilla_nn:
    def __init__(self, dataset: any, load_saved_model: bool = False, output_tag: str = None, one_hot_encoded=False):
        """
        Arguments:
        ----------
        dataset : any
            The KreuzerSkarkeDataset object containing the type of dataset used. Should be amongst 'original',
            'original_permuted', 'combinatorial', 'combinatorial_permuted', 'dirichlet', and 'dirichlet_permuted'.

        load_saved_model : bool
            Flag specifying if a pre-saved model is to be used. If 'True', loads the state_dict of the model saved
            under 'output_tag' in SAVED_MODELS_DIR folder. Defaults to 'None'. TODO: IMPLEMENT THIS

        output_tag : str
            Tag used to save model, model results, and tensorboard logs. Alternatively, if loading a pre-saved model,
            the tag used to fetch model's state_dict. Defaults to 'None'. TODO: IMPLEMENT THIS
        """
        self.dataset = dataset
        self.output_tag = output_tag

        self.one_hot_encoded=one_hot_encoded
        self.initialise_model()

    def initialise_model(self):
        
        inp = layers.Input(shape=(4, 26, 1))


        inp = layers.Input((4,26,1))     ##
        x = layers.Conv2D(5, (2,2))(inp) ##  Convolution part
        x = layers.Conv2D(4, (2,2))(x)   ##


        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(50)(x)
        if not self.one_hot_encoded:
            out = layers.Dense(1)(out)
        self.model = Model(inp, out)

        
        
        if self.one_hot_encoded:
            self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        else:
            self.model.compile(
                loss='mean_squared_error',
                optimizer=optimizers.Adam(0.001),
                metrics=[self.soft_acc],
            )
        print(self.model.summary())

    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

   
    def get_model(self):
        return self.model

    def train(self, num_epochs):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param num_epochs: Train for how many epochs
        :return: None
        '''

        self.X = {}
        self.y = {}
        self.X['train'], self.y['train'] = self.dataset.X_train, self.dataset.Y_train
        self.X['test'], self.y['test'] = self.dataset.X_test, self.dataset.Y_test

        self.model.fit(
            self.X['train'], self.y['train'],
            epochs=num_epochs,
            validation_data=(self.X['test'], self.y['test']),
            batch_size=1 # TODO: ADD CALLBACK THAT SAVES DATA FOR TENSORBOARD HERE
        )

    def get_accuracy(self):
        print(self.model.evaluate(self.X['test'], self.y['test'], batch_size=128))


if __name__ == '__main__':
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    vanillann = Vanilla_nn(dataset)
    vanillann.train(num_epochs=1)
    print(vanillann.get_accuracy())
