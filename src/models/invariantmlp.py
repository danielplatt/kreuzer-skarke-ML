import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from config.constants import *
from src.dataset import *

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

        # Encoder branch on columns
        encoder1 = nn.ModuleList()
        encoder1_dims = [26, 256, 256, 256]
        for layer_id in range(len(encoder1_dims) - 1):
            encoder1.append(nn.Conv1d(encoder1_dims[layer_id], encoder1_dims[layer_id + 1], 1, padding=0))
            # encoder1.append(nn.BatchNorm1d(encoder1_dims[layer_id+1]))
            encoder1.append(self.activation)
        self.encoder1 = nn.Sequential(*encoder1)

        # Encoder branch on rows
        encoder2 = nn.ModuleList()
        encoder2_dims = [4, 256, 256, 256]
        for layer_id in range(len(encoder2_dims) - 1):
            encoder2.append(nn.Conv1d(encoder2_dims[layer_id], encoder2_dims[layer_id + 1], 1, padding=0))
            # encoder2.append(nn.BatchNorm1d(encoder2_dims[layer_id+1]))
            encoder2.append(self.activation)
        self.encoder2 = nn.Sequential(*encoder2)

        # Decoder for joined features
        decoder = nn.ModuleList()
        decoder.append(nn.Flatten())
        decoder_dims = [256, 256, 256, 256, 43]
        for layer_id in range(len(decoder_dims) - 2):
            decoder.append(nn.Linear(in_features=decoder_dims[layer_id], out_features=decoder_dims[layer_id + 1]))
            decoder.append(nn.BatchNorm1d(decoder_dims[layer_id + 1]))
            decoder.append(self.activation)
        decoder.append(nn.Linear(in_features=decoder_dims[len(decoder_dims) - 2],
                                 out_features=decoder_dims[len(decoder_dims) - 1]))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, matrix):
        feat1, _ = torch.max(self.encoder1(torch.transpose(matrix, 1, 2)), dim=2)
        feat2, _ = torch.max(self.encoder2(matrix), dim=2)
        return self.decoder(feat1 + feat2)


class InvariantMLP():
    def __init__(
            self,
            dataset: any,
            load_saved_model: bool = False,
            output_tag: str = None,
    ):
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
            the tag used to fetch model's state_dict. Defaults to 'None'.
        """

        super().__init__()
        self.dataset = dataset
        self.model = ConvModel()
        self.output_tag = output_tag

        if load_saved_model:
            assert output_tag is not None
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.pt')
            self.model.load_state_dict(torch.load(saved_model_path))

        self.train_dataloader, self.val_dataloader = self._create_dataloaders()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _create_dataloaders(
            self,
            batch_size: int = 32,
    ) -> list:
        """
        Creates dataloaders for training and validation

        :param batch_size: batch size of the dataloaders
        :type batch_size: int
        :returns: list of training and validation dataloaders
        :rtype: list
        """
        split_idx = [self.dataset.train_idx, self.dataset.test_idx]
        samplers = [SubsetRandomSampler(x) for x in split_idx]
        dataloaders = [DataLoader(self.dataset, batch_size=batch_size, sampler=x) for x in samplers]
        return dataloaders

    def run_epoch(
            self,
            dataloader: any,
            train: bool = False,
            optimizer: any = None,
            print_result: bool = False,
    ) -> tuple:
        """
        Runs model on a single epoch of data either in 'train' or 'eval' mode

        :param dataloader: dataloader used for running epoch
        :type dataloader: pytorch dataloader object
        :param train: flag specifying if model needs to run in 'train' or 'eval' mode
        :type train: bool
        :param optimizer: optimizer used when running model in 'train' mode
        :type optimizer: pytorch optimizer object
        :param print_result: flag specifying whether to print results after running epoch
        :type print_result: bool
        :return: tuple of epoch loss and epoch accuracy
        :rtype: tuple
        """
        correct_pred, losses, num_samples, batch_counter = 0, 0, 0, 0

        for (X, y) in dataloader:
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            pred = np.argmax(pred.detach().numpy(), axis=1)
            y = y.detach().numpy()
            correct_pred += (pred == y).sum()
            losses += loss.item()
            num_samples += len(X)
            batch_counter += 1

            if train:
                loss.backward()
                optimizer.step()

        epoch_loss, epoch_acc = losses / batch_counter, correct_pred / num_samples
        if print_result:
            print('Epoch Loss: %.2f\t Epoch Accuracy: %.2f' % (epoch_loss, epoch_acc))
        return losses / batch_counter, correct_pred / num_samples

    def get_accuracy(self) -> float:
        """
        Returns accuracy on validation dataset

        :return: accuracy on validation dataset
        :rtype: float
        """
        self.model.eval()
        _, acc = self.run_epoch(self.val_dataloader, print_result=True)
        self.model.train()
        return acc

    def save_checkpoint(self, model, path):
        print('Saving model checkpoint to %s' % (path))
        torch.save(model.state_dict(), path)

    def train(
            self,
            save_csv: bool = False,
            num_epochs: int = 20,
            learning_rate: float = 0.001,
            patience: int = 12,
    ):
        """
        Trains model on given dataset. Also, saves model and logs data for tensorboard visualization

        :param save_csv: flag specifying if results need to be saved in a csv
        :type save_csv: bool
        :param num_epochs: num of epochs used while training the model
        :type num_epochs: int
        :param learning_rate: learning rate used in the optimizer
        :type learning_rate: float
        :param patience: number of epochs with no improvemtent after which training stops
        :type patience: int
        """
        if self.output_tag is None:
            self.output_tag = 'invariantmlp_' + self.dataset.projections_file

        saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.pt')
        saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')

        writer = SummaryWriter(TENSORBOARD_DIR)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        max_acc, counter = None, 0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self.run_epoch(dataloader=self.train_dataloader, train=True, optimizer=optimizer)
            val_loss, val_acc = self.run_epoch(dataloader=self.val_dataloader)

            print('Epoch: %d\n\t Training Loss: %.2f\t Training Accuracy: %.2f' % (epoch, train_loss, train_acc ))
            print('\t Validation Loss: %.2f\t Validation Accuracy: %.2f' % (val_loss, val_acc))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            writer.add_scalars('loss/train', {self.output_tag: train_loss}, epoch)
            writer.add_scalars('accuracy/train', {self.output_tag: train_acc}, epoch)
            writer.add_scalars('loss/val', {self.output_tag: val_loss}, epoch)
            writer.add_scalars('accuracy/val', {self.output_tag: val_acc}, epoch)
            writer.flush()

            if max_acc is None:
                max_acc = train_acc
                self.save_checkpoint(self.model, saved_model_path)
            elif train_acc < max_acc:
                counter += 1
                if counter >= patience:
                    print('No improvement in training accuracy for %d epochs. Exiting training.'%(patience))
                    break
            else:
                max_acc = train_acc
                counter = 0
                self.save_checkpoint(self.model, saved_model_path)

            # if max_acc <= val_acc:
            #     max_acc = val_acc
            #     print('Saving model checkpoint to %s for epoch %d' % (saved_model_path, epoch))
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #         'train_accuracy': train_acc,
            #         'val_accuracy': val_acc,
            #     }, saved_model_path)

        if save_csv:
            results_df = pd.DataFrame({
                'epoch': list(range(num_epochs)),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
            })

            print('Saving results as a csv in  %s' % saved_results_path)
            results_df.to_csv(saved_results_path)

        writer.close()

if __name__ == "__main__":
    dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='original')
    invariantmlp = InvariantMLP(dataset)
    invariantmlp.train(num_epochs=1)
    print(invariantmlp.get_accuracy())

