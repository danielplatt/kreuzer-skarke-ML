import xgboost as xgb

from config.constants import *
from src.dataset import *
from sklearn.model_selection import train_test_split

class XGboost:
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
        # TODO: We need to let users input params

        super().__init__()
        self.dataset = dataset
        params = {'n_estimators': 500, "objective":"reg:tweedie",'colsample_bytree': 0.5,'learning_rate': 0.01,
            'max_depth': 6, 'alpha': 1}
        
        self.model = xgb.XGBRegressor(**params)

        self.output_tag = output_tag
        self.X = {}
        self.y = {}

        num_entries_test = self.dataset.X_test.shape[0]
        self.X['test'] = self.dataset.X_test.reshape(num_entries_test,4*26)
        self.y['test'] = self.dataset.Y_test.reshape(num_entries_test)

        x_train, x_val, y_train, y_val =  train_test_split(self.dataset.X_train, 
            self.dataset.Y_train, test_size=0.2,random_state=3)
        num_entries_train = x_train.shape[0]
        self.X['train'] = x_train.reshape(num_entries_train,4*26)
        self.y['train'] = y_train.reshape(num_entries_train)

        num_entries_validation = x_val.shape[0]
        self.X['validation'] = x_val.reshape(num_entries_validation,4*26)
        self.y['validation'] = y_val.reshape(num_entries_validation)

        assert not one_hot_encoded, "one hot encoding is not currently supported for XGboost."
        if load_saved_model:
            assert output_tag is not None
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.json')
            print(saved_model_path)
            self.model = xgb.Booster()
            self.model.load_model(saved_model_path)

    
    def train(self, num_epochs):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param num_epochs: Train for how many epochs
        :return: None
        '''
        # num_epochs as number of early stopping rounds
        # training, we set the early stopping rounds parameter to be the number of epochs
        self.model.fit(self.X['train'], self.y['train'], eval_set=[(self.X['train'], self.y['train']),
                (self.X['validation'], self.y['validation'])], early_stopping_rounds=num_epochs)
    
    def get_accuracy(self):
        
        y_predicted = self.model.predict(xgb.DMatrix(self.X['test']))
        print("The accuracy on the test set is ",(np.sum(np.round(y_predicted) == self.y['test']))/len(self.y['test']))
        

if __name__ == '__main__':
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    xgbb = XGboost(dataset)
    xgbb.train(num_epochs=20)
    xgbb.get_accuracy()
