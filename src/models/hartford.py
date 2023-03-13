from keras import layers, models, optimizers, callbacks
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
from kormos.models import BatchOptimizedSequentialModel
import pandas as pd
from src.dataset import *


class Hartford:
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
        self.batch_size = 32

        self.X = {}
        self.y = {}
        self.X['train'], self.y['train'] = self.dataset.X_train, self.dataset.Y_train
        self.X['test'], self.y['test'] = self.dataset.X_test, self.dataset.Y_test
        self.one_hot_encoded=one_hot_encoded
        self.initialise_model(self,load_saved_model=load_saved_model)
        if load_saved_model:
            assert output_tag is not None
            #del self.model  # deletes the existing model
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.h5')
            self.model.load_weights(saved_model_path)

    def initialise_model(self, pooling='sum', load_saved_model:bool = True):
        number_of_channels = 256
        inp = layers.Input(shape=(4, 26, 1))
        inp_list = [inp for _ in range(number_of_channels)]
        inp_duplicated = layers.Concatenate(axis=3)(inp_list)
        e1 = self.apply_equivariant_layer(inp_duplicated, number_of_channels)
        e2 = self.apply_equivariant_layer(e1, number_of_channels)
        e3 = self.apply_equivariant_layer(e2, number_of_channels)

        if pooling == 'sum':
            p1 = layers.AveragePooling2D((4, 26), strides=(1, 1), padding='valid')(e3)
        else:
            p1 = layers.MaxPooling2D((4, 26), strides=(1, 1), padding='valid')(e3)
        p2 = layers.Reshape((number_of_channels,))(p1)
        fc1 = layers.Dense(256, activation='relu')(p2)
        fc2 = layers.Dense(256, activation='relu')(fc1)
        fc3 = layers.Dense(256, activation='relu')(fc2)
        fc4 = layers.Dense(256, activation='relu')(fc3)
        out = layers.Dense(35, activation='linear')(fc4)

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
        if not load_saved_model:
          print(self.model.summary())

    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def apply_equivariant_layer(self, inp, number_of_channels_out):
        '''
        Implementation of the equivariant layer from Hartford et al.: Deep Models of Interactions Across Sets

        Each channel of this layer has 5 parameters. Explanation of the parameters:
        (1) Multiply every element of the matrix by a parameter
        (2) Take the average of every row, which gives a 4x1 matrix. Write that 26 times next to each other to get a 4x26 matrix. Multiply the result by a parameter
        (3) Same for columns
        (4) Take the average of all matrix elements, which gives a 1x1 matrix. Repeat that number to get a 4x26 matrix. Multiply the result by a parameter
        (5) The bias, which is programmatically included in (4) but could have been applied extra

        :param inp: an input tensor
        :param number_of_channels_out: number of output channels, each channel in this layer has 4 learnable parameters
        :return: the result of a Hartford layer applied to the input tensor
        '''

        # ---(1)---
        out1 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(inp)

        # ---(2)---
        out2 = layers.AveragePooling2D((1, 26), strides=(1, 1), padding='valid')(inp)
        repeated2 = [out2 for _ in range(26)]
        out2 = layers.Concatenate(axis=2)(repeated2)
        out2 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(out2)

        # ---(3)---
        out3 = layers.AveragePooling2D((4, 1), strides=(1, 1), padding='valid')(inp)
        repeated3 = [out3 for _ in range(4)]
        out3 = layers.Concatenate(axis=1)(repeated3)
        out3 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=False, activation='relu')(out3)

        # ---(4)---
        out4 = layers.AveragePooling2D((4, 26), strides=(1, 1), padding='valid')(inp)
        repeated4 = [out4 for _ in range(4)]
        out4 = layers.Concatenate(axis=1)(repeated4)
        repeated4 = [out4 for _ in range(26)]
        out4 = layers.Concatenate(axis=2)(repeated4)
        out4 = layers.Conv2D(number_of_channels_out, (1,1), strides=(1, 1), padding='valid', use_bias=True, activation='relu')(out4)

        return layers.Add()([out1,out2,out3,out4])

    def get_model(self):
        return self.model

    def train(self, save_csv: bool = True,
            num_epochs: int = 20):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param num_epochs: Train for how many epochs
        :return: None
        '''

        if self.output_tag is None:
            self.output_tag = 'hartford_' + self.dataset.projections_file
        log_dir =  TENSORBOARD_DIR.joinpath(self.output_tag)      
        history = self.model.fit(
            x=self.X['train'],
            y=self.y['train'],
            batch_size=self.batch_size,
            epochs=num_epochs,
            validation_data=(self.X['test'], self.y['test']),
            callbacks=[callbacks.TensorBoard(log_dir=log_dir)],#tensorboard_callback
        )
        if self.output_tag is not None:
            saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.h5')
            print('Saving final model checkpoint to %s for %d epochs ' % (saved_model_path, num_epochs))
            self.model.save_weights(saved_model_path)  # creates a HDF5 file 'my_model.h5'

        if save_csv:
            saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')
            results_df = pd.DataFrame(history.history)
            print('Saving results as a csv in  %s' % saved_results_path)
            results_df.to_csv(saved_results_path)


        return history

    def get_accuracy(self):
        print(self.model.evaluate(self.X['test'], self.y['test'], batch_size=128))


if __name__ == '__main__':
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    hardy = Hartford(dataset)
    hardy.train(num_epochs=1)
    print(hardy.get_accuracy())
