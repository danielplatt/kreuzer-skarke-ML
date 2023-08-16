from src.dataset import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from torchsummary import summary

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
class VanillaCNN(torch.nn.Module):
    def __init__(self):
        """Initialize a network of 361,608 parameters"""
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Flatten(),
            nn.Linear(2688, 128),
            nn.ReLU(),
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, 50))
       

    def forward(self, batch):
        
        return self.network(batch)

class Vanilla_nn():
    def __init__(
            self,
            dataset: any,
            load_saved_model: bool = False,
            output_tag: str = None,one_hot_encoded=False
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VanillaCNN().float()
        
        self.model.to(self.device)
        # Uncomment to see the full network parameters
        #summary(self.model,(1,4,26), batch_size=64)
        self.dataset = dataset
        self.output_tag = output_tag
        
        if load_saved_model:
            assert output_tag is not None
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.pt')
            self.model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_dataloader, self.val_dataloader = self._create_dataloaders()
        def build_loader(x_, y_):
            loaders = []
            for X, y in zip(x_, y_):
                #print(matrix.shape)
                X = np.reshape(X,(1,4,26))
                X = torch.from_numpy(X).float().to(self.device)
                
                y = torch.tensor(y).to(self.device)
                loaders.append((X,y))
            return loaders

        points_train = build_loader(self.dataset.X_train, self.dataset.Y_train)
        points_test  = build_loader(self.dataset.X_test,  self.dataset.Y_test)

        self.train_dataloader = DataLoader(points_train, batch_size=64)
        self.val_dataloader   = DataLoader(points_test,  batch_size=64)
        #print(self.train_dataloader.dataset[0].shape)
        

    def _create_dataloaders(
            self,
            batch_size: int = 64,
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
            #print(pred.shape)
            #print(y.shape)
            loss = self.loss_fn(pred, y)

            pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            y = y.detach().cpu().numpy()
            correct_pred += (pred == y).sum()
            losses += loss.item()
            num_samples += len(y)
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
        return acc

   

    def train(
            self,
            save_csv: bool = False,
            num_epochs: int = 100,
            learning_rate: float = 0.001,
            patience: int = 100,
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
            self.output_tag = 'vanillann_' + self.dataset.projections_file

        saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.pt')
        saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')

        writer = SummaryWriter(TENSORBOARD_DIR)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        min_loss = 1000
        counter = 0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        self.model.train()

        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self.run_epoch(dataloader=self.train_dataloader, train=True, optimizer=optimizer)
            val_loss, val_acc     = self.run_epoch(dataloader=self.val_dataloader)

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

            if min_loss is None:
                min_loss = train_loss
                self.save_checkpoint(self.model, saved_model_path)
            elif train_loss > min_loss:
                counter += 1
                if counter >= patience:
                    print('No improvement in training accuracy for %d epochs. Exiting training.'%(patience))
                    break
            else:
                min_loss = train_loss
                counter = 0
                torch.save({
                     'epoch': epoch,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'train_loss': train_loss,
                     'val_loss': val_loss,
                     'train_accuracy': train_acc,
                     'val_accuracy': val_acc,
                 }, saved_model_path)

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
    vanillann = Vanilla_nn(dataset)
    vanillann.train(num_epochs=1)
    print(vanillann.get_accuracy())



