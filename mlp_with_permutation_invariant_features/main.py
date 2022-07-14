import torch
import numpy as np
from torch.nn import ReLU
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os.path import join

def get_accuracy(prediction, target):
    return (prediction == target).sum() / target.shape[0]

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.activation = ReLU()
        
        # Encoder branch on columns
        encoder1 = nn.ModuleList()
        encoder1_dims = [26,256,256,256]
        for layer_id in range(len(encoder1_dims)-1):
            encoder1.append(nn.Conv1d(encoder1_dims[layer_id], encoder1_dims[layer_id + 1], 1, padding=0))
            #encoder1.append(nn.BatchNorm1d(encoder1_dims[layer_id+1]))
            encoder1.append(self.activation)
        self.encoder1 = nn.Sequential(*encoder1)
        
        # Encoder branch on rows
        encoder2 = nn.ModuleList()
        encoder2_dims = [4,256,256,256]
        for layer_id in range(len(encoder2_dims)-1):
            encoder2.append(nn.Conv1d(encoder2_dims[layer_id], encoder2_dims[layer_id + 1], 1, padding=0))
            #encoder2.append(nn.BatchNorm1d(encoder2_dims[layer_id+1]))
            encoder2.append(self.activation)
        self.encoder2 = nn.Sequential(*encoder2)
	
	# Decoder for joined features 
        decoder = nn.ModuleList()
        decoder.append(nn.Flatten())
        decoder_dims = [256,256,256,256,43]
        for layer_id in range(len(decoder_dims)-2):
            decoder.append(nn.Linear(in_features=decoder_dims[layer_id], out_features=decoder_dims[layer_id + 1]))
            decoder.append(nn.BatchNorm1d(decoder_dims[layer_id+1]))
            decoder.append(self.activation)
        decoder.append(nn.Linear(in_features=decoder_dims[len(decoder_dims)-2], out_features=decoder_dims[len(decoder_dims)-1]))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, matrix):
        feat1,_ = torch.max(self.encoder1(matrix), dim=2)
        feat2,_ = torch.max(self.encoder2(torch.transpose(matrix,1,2)), dim=2)
        return self.decoder(feat1+feat2)

class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.count = self.y.shape[0]

    def __getitem__(self, index):
        d = dict()
        d["x"] = self.x[index]
        d["y"] = self.y[index]
        return d

    def __len__(self):
        return self.count

run_path = "mainrun"
#os.makedirs(run_path)

# Load data
X_data = np.load("v26_X.npy")
y_data = np.load("v26_y.npy")
X_data = np.transpose(X_data, axes=(0,2,1))
device = "cuda:0"
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5)
model = Model().to(device)
sm = nn.Softmax(dim=1)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
dataset_train = Dataset(x=X_train, y=y_train)
train_dataloader = DataLoader(dataset=dataset_train, batch_size=10000, shuffle=True)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=50, min_lr=0.000001)
dataset_validation = Dataset(x=X_test, y=y_test)
validation_dataloader = DataLoader(dataset=dataset_validation, batch_size=40000, shuffle=False)
validation_losses = []
max_acc = 0

for epoch in range(200):

    #print("Epoch: {}, LR {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    train_loss = 0
    batch_counter = 0
	
    # Training loop
    for batch_idx, sample_batch in enumerate(train_dataloader):

        optimizer.zero_grad()
        x = sample_batch["x"].float().to(device)
        y = sample_batch["y"].to(device)

        prediction = model.forward(x)

        loss_value = loss_function(sm(prediction), F.one_hot(y, num_classes=43).float())
        loss_value.backward()

        optimizer.step()
        train_loss += loss_value.item()
        batch_counter += 1

    train_loss /= batch_counter
    #print("Epoch {}: Training-Loss {}".format(epoch, train_loss))

    # Validation loop
    with torch.no_grad():

        model.eval()
        validation_loss = 0
        batch_counter = 0

        for batch_idx, sample_batch in enumerate(validation_dataloader):

            x = sample_batch["x"].float().to(device)
            y = sample_batch["y"].to(device)

            prediction = model.forward(x)

            loss_value = loss_function(sm(prediction), F.one_hot(y, num_classes=43).float())
            validation_loss += loss_value.item()
            batch_counter += 1

        validation_loss /= batch_counter

        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        target = y.detach().cpu().numpy()
        accuracy = get_accuracy(prediction, target)
        validation_losses.append(accuracy)
        scheduler.step(accuracy)

        #print("Epoch {}: Validation-Loss {}".format(epoch, accuracy))

        if accuracy > max_acc:
            max_acc = accuracy
            print("Epoch {}: {}".format(epoch, max_acc))
            save_name = join(run_path, "epoch_{}_best.pkl".format(str(epoch).zfill(3)))
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch}
            torch.save(checkpoint, save_name)

print("Best loss {} ...".format(np.max(np.array(validation_losses))))
