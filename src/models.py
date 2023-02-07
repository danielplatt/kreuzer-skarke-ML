import torch
from torch.nn import ReLU
from torch import nn

class InvariantMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()

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