from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import layers, Model, optimizers
import tensorflow as tf
import datetime, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from src.dataset import *
import random
from itertools import permutations

class Patches(layers.Layer):
    def get_config(self):
      config = super().get_config()
      return config
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1,self.patch_size, self.patch_size,1],
            strides=[1,self.patch_size, self.patch_size,1],
            rates=[1,1, 1,1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def get_config(self):
       config = super().get_config()
       return config
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


class VisionT:
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
        self.learning_rate = 0.001
        self.weight_decay = 0.00001
        self.batch_size = 256
        self.num_epochs = 5
        self.image_size = 52  # We'll resize input images to this size
        self.patch_size = 4  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 2
        #self.mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
        self.mlp_head_units = [4,2]
        perm_col_index = [i for i in range(26)]
        random.seed(4)
        random.shuffle(perm_col_index)
        self.perm_col_index = perm_col_index
        self.num_classes = 45
        self.input_shape = (52, 52 , 1)

        self.model = self.initialise_model()

    def permute_row(self,input_matrix, index):
        permute_index = index % 12
        l = list(permutations(range(1, 5)))
        output_matrix = np.zeros(input_matrix.shape)
        for i in range(4):
            output_matrix[i,:] = input_matrix[l[permute_index][i]-1,:] 
        return output_matrix

    def permute_column(self,input_matrix):
        output_matrix = np.zeros(input_matrix.shape)
        for i in range(26):
            output_matrix[:,i] = input_matrix[:,self.perm_col_index[i]] 
        return output_matrix


    def create_vit_classifier(self):
        inputs = layers.Input(shape= self.input_shape)
        # # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        # inputs = tf.convert_to_tensor(inputs.reshape((inputs.shape[0],inputs.shape[1],inputs.shape[2],1)))
        print(inputs.shape)
        patches = Patches(self.patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def initialise_model(self):
        
        

        ## TO DO: Handle one hot encoding case
        #if not self.one_hot_encoded:
        #out = layers.Dense(1)(out)
        model = self.create_vit_classifier()
        #configure the training parameters
        optimizer = tfa.optimizers.AdamW(
        learning_rate=self.learning_rate, weight_decay=self.weight_decay, clipnorm=5
    )

        
        
        
        if self.one_hot_encoded:
            model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
            )
        else:
            model.compile(
                loss='mean_squared_error',
                optimizer=optimizers.Adam(0.001),
                metrics=[self.soft_acc],
            )
        print(model.summary())
        return model

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
        x_train, self.y['train'] = self.dataset.X_train, self.dataset.Y_train
        x_test, self.y['test'] = self.dataset.X_test, self.dataset.Y_test
        
        num_train = np.shape(x_train)[0]
        num_test = np.shape(x_test)[0]
        self.X['train'] = np.zeros((num_train,52,52))
        self.X['test'] = np.zeros((num_test,52,52))
        for i in range(np.shape(x_train)[0]):
            for j in range(13):
                self.X['train'][i,4*j:4*j+4,:26] = np.copy(self.permute_row(x_train[i,:,:],j))
                # not permuted column stacked
                self.X['train'][i,4*j:4*j+4,26:] = np.copy(self.permute_column(self.permute_row(x_train[i,:,:],j)))
        for i in range(np.shape(x_test)[0]):
            for j in range(13):

                self.X['test'][i,4*j:4*j+4,:26] = np.copy(self.permute_row(x_test[i,:,:],j))
                # not permuted column stacked
                self.X['test'][i,4*j:4*j+4,26:] = np.copy(self.permute_column(self.permute_row(x_test[i,:,:],j)))
        
     
        
        
        if self.output_tag is None:
            self.output_tag = 'vision_transformer_' + self.dataset.projections_file

        self.saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.pt')
        

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            self.saved_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )




        # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        
        history = self.model.fit(
            x=self.X['train'],
            y=self.y['train'],
            batch_size=self.batch_size,
            epochs=1,
            validation_split=0.1,
            callbacks=[checkpoint_callback],#tensorboard_callback
        )

        

        return history

        # self.model.fit(
        #     self.X['train'], self.y['train'],
        #     epochs=num_epochs,
        #     validation_data=(self.X['test'], self.y['test']),
        #     batch_size=1 # TODO: ADD CALLBACK THAT SAVES DATA FOR TENSORBOARD HERE
        # )

    def get_accuracy(self):
        #self.model.load_weights(self.saved_model_path)
        _, accuracy, top_5_accuracy = self.model.evaluate(self.X['test'], self.y['test'])
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        #print(self.model.evaluate(self.X['test'], self.y['test'], batch_size=128))


if __name__ == '__main__':
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    visiontrans = VisionT(dataset)
    visiontrans.train(num_epochs=1)
    print(visiontrans.get_accuracy())




