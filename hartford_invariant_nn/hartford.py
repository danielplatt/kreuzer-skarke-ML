from keras import layers, models, optimizers
from keras import backend as K
from data.get_data import get_data
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Hartford:
    def __init__(self):
        self.initalise_model(self)

    def initalise_model(self, pooling='sum'):
        number_of_channels = 256
        inp = layers.Input(shape=(4, 26, 1))
        inp_list = [inp for _ in range(number_of_channels)]
        inp_duplicated = layers.Concatenate(axis=3)(inp_list)
        e1 = self.apply_equivariant_layer(inp_duplicated, number_of_channels)
        # e1 = layers.Dropout(0.1)(e1)
        e2 = self.apply_equivariant_layer(e1, number_of_channels)
        # e2 = layers.Dropout(0.1)(e2)
        e3 = self.apply_equivariant_layer(e2, number_of_channels)
        # e3 = layers.Dropout(0.1)(e3)
        # e4 = self.apply_equivariant_layer(e3, number_of_channels)
        # e4 = layers.Dropout(0.1)(e4)
        # e5 = self.apply_equivariant_layer(e4, number_of_channels)
        # e5 = layers.Dropout(0.1)(e5)

        # e6 = self.apply_equivariant_layer(e5, number_of_channels)
        # e6 = layers.Dropout(0.1)(e6)
        # e7 = equivariant_layer(e6, number_of_channels, number_of_channels)
        # # e7 = layers.Dropout(0.5)(e7)
        # e8 = equivariant_layer(e7, number_of_channels, number_of_channels)
        # e9 = equivariant_layer(e8, number_of_channels, number_of_channels)

        if pooling == 'sum':
            p1 = layers.AveragePooling2D((4, 26), strides=(1, 1), padding='valid')(e3)
        else:
            p1 = layers.MaxPooling2D((4, 26), strides=(1, 1), padding='valid')(e3)
        p2 = layers.Reshape((number_of_channels,))(p1)
        fc1 = layers.Dense(256, activation='relu')(p2)
        fc1 = tf.keras.layers.BatchNormalization()(fc1)
        fc2 = layers.Dense(256, activation='relu')(fc1)
        fc2 = tf.keras.layers.BatchNormalization()(fc2)
        fc3 = layers.Dense(256, activation='relu')(fc2)
        fc3 = tf.keras.layers.BatchNormalization()(fc3)
        fc4 = layers.Dense(256, activation='relu')(fc3)
        fc4 = tf.keras.layers.BatchNormalization()(fc4)
        out = layers.Dense(35, activation='linear')(fc4)

        self.model = models.Model(inputs=inp, outputs=out)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(0.01),
            metrics=['categorical_accuracy'],
        )
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

    def train(self, datasetname='original'):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param datasetname:'original.npz' or 'dirichlet_permuted.npz' or 'dirichlet.npz' etc
        :return: None
        '''

        loaded_data = get_data(datasetname=datasetname, one_hot=True)
        X_loaded = loaded_data['x_proj']
        y_loaded = loaded_data['y']

        self.X = {}
        self.y = {}
        self.X['train'], self.X['test'], self.y['train'], self.y['test'] = train_test_split(X_loaded, y_loaded, test_size=0.5)
        self.model.fit(
            self.X['train'], self.y['train'],
            epochs=200,
            validation_data=(self.X['test'], self.y['test']),
        )

    def get_accuracy(self):
        print(self.model.evaluate(self.X['test'], self.y['test'], batch_size=128))


if __name__ == '__main__':
    hardy = Hartford()
    hardy.train(datasetname='dirichlet.npz')
    print(hardy.get_accuracy())
