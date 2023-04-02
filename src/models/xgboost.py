import pandas as pd
import numpy as np
import xgboost as xgb

import tensorflow as tf
from config.constants import *
from src.dataset import *
from sklearn.model_selection import train_test_split
from typing import List

from tensorboardX import SummaryWriter
import datetime, os

class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, experiment: str = None, data_name: str = None):
        self.projection_file_name = experiment
        #self.datetime_ = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir_train = TENSORBOARD_DIR.joinpath(str(self.projection_file_name + "_train"))
        self.log_dir_test = TENSORBOARD_DIR.joinpath(str(self.projection_file_name + "_test"))
        self.train_writer = SummaryWriter(log_dir=self.log_dir_train)
        self.test_writer = SummaryWriter(log_dir=self.log_dir_test)

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.test_writer.add_scalar(metric_name, score, epoch)        
        return False



def reg_accuracy(predt: np.ndarray, dtrain: xgb.DMatrix) -> List[Tuple[str, float]]:
    ''' Root mean squared log error metric.'''
    y_true = dtrain.get_label()
    acc = float((np.sum(np.round(predt) == y_true))/ len(y_true))
    #default p in XGBoost as 1.5
    p = 1.5
    loss = - y_true * tf.pow(predt, 1 - p) / (1 - p) + \
               tf.pow(predt, 2 - p) / (2 - p)
    return [('Regression-accuracy', acc),('Tweedie-Loss', tf.reduce_mean(loss))]

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
            under 'output_tag' in SAVED_MODELS_DIR folder. Defaults to 'None'. 

        output_tag : str
            Tag used to save model, model results, and tensorboard logs. Alternatively, if loading a pre-saved model,
            the tag used to fetch model's state_dict. Defaults to 'None'. TODO: IMPLEMENT THIS
        """
        # TODO: We need to let users input params

        super().__init__()
        self.dataset = dataset
        # params = {'n_estimators': 500, "objective":"reg:tweedie",'colsample_bytree': 0.5,'learning_rate': 0.01,
        #     'max_depth': 6, 'alpha': 1}
        
        self.output_tag = output_tag
        self.X = {}
        self.y = {}
        self.model = xgb.Booster()
        num_entries_test = self.dataset.X_test.shape[0]
        self.X['test'] = self.dataset.X_test.reshape(num_entries_test,4*26)
        self.y['test'] = self.dataset.Y_test.reshape(num_entries_test)

        num_entries_train = self.dataset.X_train.shape[0]
        self.X['train'] = self.dataset.X_train.reshape(num_entries_train,4*26)
        self.y['train'] = self.dataset.Y_train.reshape(num_entries_train)
        
        assert not one_hot_encoded, "one hot encoding is not currently supported for XGboost."
        if load_saved_model:
            assert output_tag is not None
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.json')
            self.model.load_model(saved_model_path)

    
    def train(self, save_csv: bool = True,
            num_epochs: int = 20):
        '''
        Trains the neural network with "Hartford et al" architecture and prescribed hyperparameters

        :param num_epochs: Train for how many epochs
        :return: None
        '''
        # num_epochs as number of early stopping rounds
        # training, we set the early stopping rounds parameter to be the number of epochs
        dtrain = xgb.DMatrix(self.X['train'], label=self.y['train'])
        dtest = xgb.DMatrix(self.X['test'], label=self.y['test'])
        params = {"objective":"reg:tweedie",'colsample_bytree': 0.5,'learning_rate': 0.01,
             'max_depth': 6, 'alpha': 1, 'disable_default_eval_metric':1}
        results: Dict[str, Dict[str, List[float]]] = {}
        if self.output_tag is None:
            self.output_tag = 'xgb_' + self.dataset.projections_file
        
        self.model= xgb.train(params, dtrain, num_boost_round=num_epochs, custom_metric=reg_accuracy, evals=[(dtrain,'train'),
                (dtest,'dtest')], callbacks = [TensorBoardCallback(self.output_tag)],evals_result=results
                )
                
        if self.output_tag is not None:
          saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.json')
          print('Saving final model checkpoint to %s for %d boosting rounds(number of estimators) ' % (saved_model_path, num_epochs))
          self.model.save_model(saved_model_path)
        if save_csv:
          saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')
          results_df = pd.DataFrame(results)
          print('Saving results as a csv in  %s' % saved_results_path)
          results_df.to_csv(saved_results_path)

    def get_accuracy(self):
        
        y_predicted = self.model.predict(xgb.DMatrix(self.X['test']))
        this_accuracy = (np.sum(np.round(y_predicted) == self.y['test']))/len(self.y['test'])
        print("The accuracy on the test set is ", this_accuracy)
        return this_accuracy
        

if __name__ == '__main__':
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    xgbb = XGboost(dataset)
    xgbb.train(num_epochs=20)
    xgbb.get_accuracy()
