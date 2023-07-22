
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


class AutoEncoder(nn.Module):

    def __init__(self, encoder_layers, decoder_layers):
        super(AutoEncoder, self).__init__()

        # Save bottleneck size
        self.bottleneck = encoder_layers[-1]

        # Create Encoder
        encoder_layers_torch = []
        enc_len = len(encoder_layers)
        
        # Loop through encoder_layers and add layers
        for i in range(enc_len - 1):
            encoder_layers_torch.append(nn.Linear(encoder_layers[i], encoder_layers[i+1]))
            encoder_layers_torch.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers_torch)

        # Create Decoder
        decoder_layers_torch = []
        dec_len = len(decoder_layers)

        # Loop through decoder_layers and add layers
        for i in range(dec_len - 1):
            decoder_layers_torch.append(nn.Linear(decoder_layers[i], decoder_layers[i+1]))
            # Last one does not have activation function
            if i != dec_len - 2:
                decoder_layers_torch.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers_torch)
        
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output

    def train(self, dataset, criterion = 'mse', optimizer = 'sgd', lr = 0.01, batch_size = 1, epochs = 5, verbose = 5):
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Criterion
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr = lr)

        losses = []
        start_time = time.time()
        for i in range(epochs):
            batch_n = 0
            for data in data_loader:

                # Reset Gradients
                optimizer.zero_grad()

                # Forward Pass
                pred =  self.forward(data)


                # Loss Function
                loss = criterion(pred, data)
                loss.backward()

                # Upgrade Weights and Biases
                optimizer.step()

                batch_n += 1
                if verbose != 0 and batch_n % verbose == 0:
                    print('Epoch:', i + 1, '| Batch number:', batch_n , '| Loss:', loss.item(), '| Time:',   round(time.time() - start_time, 2), 's')
                    start_time = time.time()


            losses.append(loss.item())

        return losses

    def input_gen(self, output, epochs = 100, optimizer = 'sgd', criterion = 'mse', lr = 0.01):

        input = torch.rand(1, self.bottleneck)


        if optimizer == 'sgd':
            optimizer = optim.SGD([input], lr = lr)
        
        if criterion == 'mse':
            criterion = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()

            pred = self.decoder(input)

            loss = criterion(pred, output)
            loss.backward()

            optimizer.step()

        return input