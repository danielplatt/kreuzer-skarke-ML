from src.dataset import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 4, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)
            
        return self.mlp(input)  # Apply our final MLP.
    
class PointNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(4, 50)
        self.conv2 = PointNetLayer(50, 100)
        self.conv3 = PointNetLayer(100, 100)
        self.conv4 = PointNetLayer(100, 43)
        self.classifier = Linear(43, 43)
        
    def forward(self, pos, batch, edge_index):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        #edge_index = knn_graph(pos, k=26, batch=batch, loop=False)
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        
        # 5. Classifier.
        return self.classifier(h)
    
class PointNet():
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PointNetModel().float() 
        self.model.to(self.device)
        self.dataset = dataset
        self.output_tag = output_tag

        if load_saved_model:
            assert output_tag is not None
            saved_model_path = SAVED_MODELS_DIR.joinpath(output_tag + '.pt')
            self.model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])

        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        #self.train_dataloader, self.val_dataloader = self._create_dataloaders()
        def build_points(x_, y_):
            graphs = []
            for matrix, y in zip(x_, y_):
                #print(matrix.shape)
                matrix_ = torch.from_numpy(matrix).float().to(self.device)
                data = Data()
                data.y = torch.tensor(y).to(self.device)
                data.pos = torch.transpose(matrix_, 0, 1)
                graphs.append(data)
            return graphs

        points_train = build_points(self.dataset.X_train, self.dataset.Y_train)
        points_test  = build_points(self.dataset.X_test,  self.dataset.Y_test)
        
        for data in points_train:
            data.edge_index = knn_graph(data.pos, k=26)
        for data in points_test:
            data.edge_index = knn_graph(data.pos, k=26)
            
        self.train_dataloader = DataLoader(points_train, batch_size=64)
        self.val_dataloader   = DataLoader(points_test,  batch_size=64)

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

        for data in dataloader:
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
            pred = self.model(data.pos, data.batch, data.edge_index)
            loss = self.loss_fn(pred, data.y)

            pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            y = data.y.detach().cpu().numpy()
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
    
    def get_predictions(self):
        """
        Returns predictions and truth vectors from validation dataset
        """
        self.model.eval()
        preds = []
        truth = []
        
        for data in self.val_dataloader:
            pred = self.model(data.pos, data.batch, data.edge_index)
            pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            preds.append(pred)
            truth.append(data.y.detach().cpu().numpy())
            
        if self.output_tag is None:
            self.output_tag = 'pointnet_' + self.dataset.projections_file

        saved_confm_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '_confmatrix.png')
            
        return preds, truth, saved_confm_path

    def train(
            self,
            save_csv: bool = False,
            num_epochs: int = 100,
            learning_rate: float = 0.001,
    ):
        """
        Trains model on given dataset. Also, saves model and logs data for tensorboard visualization

        :param save_csv: flag specifying if results need to be saved in a csv
        :type save_csv: bool
        :param num_epochs: num of epochs used while training the model
        :type num_epochs: int
        :param learning_rate: learning rate used in the optimizer
        :type learning_rate: float
        """
        if self.output_tag is None:
            self.output_tag = 'pointnet_' + self.dataset.projections_file

        saved_model_path = SAVED_MODELS_DIR.joinpath(self.output_tag + '.pt')
        saved_results_path = SAVED_RESULTS_DIR.joinpath(self.output_tag + '.csv')

        writer = SummaryWriter(TENSORBOARD_DIR)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        max_acc = 0
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

            if max_acc <= val_acc:
                max_acc = val_acc
                print('Saving model checkpoint to %s for epoch %d' % (saved_model_path, epoch))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                }, saved_model_path)

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
    pointnet = PointNet(dataset)
    pointnet.train(num_epochs=1)
    print(pointnet.get_accuracy())