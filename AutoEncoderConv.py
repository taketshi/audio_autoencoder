
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


class AutoEncoderConv(nn.Module):


    def __init__(self, encoder_layers, decoder_layers):
        super(AutoEncoderConv, self).__init__()

        # encoder_layers is dictionary of the form
        # {channels: [1, 40, 20, 10],
        #  kernel_conv: [3, 3, 1, 2],
        #  padding: [1, 1, 1, 3],
        #  stride_conv: [1,1,1],
        #  kernel_pool: [2, 2, 2],
        #  stride_pool: [2, 3, 5, 2]}


        # Create Encoder
        encoder_layers_torch = []
        enc_len = len(encoder_layers['channels'])
        
        # Loop through encoder_layers and add layers
        for i in range(enc_len - 1):
            encoder_layers_torch.append(nn.Conv2d(in_channels = encoder_layers['channels'][i], out_channels = encoder_layers['channels'][i+1],
                                        kernel_size = encoder_layers['kernel_conv'][i], padding = encoder_layers['padding'][i]))
            encoder_layers_torch.append(nn.LeakyReLU())
            encoder_layers_torch.append(nn.MaxPool2d(kernel_size=encoder_layers['kernel_pool'][i], stride=encoder_layers['stride_pool'][i]))

        self.encoder = nn.Sequential(*encoder_layers_torch)

        # decoder_layers is dictionary of the form
        # {channels: [1, 40, 20, 10],
        #  kernel: [3, 3, 1, 2],
        #  padding: [1, 1, 1, 3],
        #  stride: [1, 1, 1, 3]}

        # Create Decoder
        decoder_layers_torch = []
        dec_len = len(decoder_layers['channels'])

        # Loop through decoder_layers and add layers
        for i in range(dec_len - 1):
            decoder_layers_torch.append(nn.ConvTranspose2d(in_channels = decoder_layers['channels'][i], out_channels = decoder_layers['channels'][i+1],
                                        kernel_size = decoder_layers['kernel'][i], padding = decoder_layers['padding'][i],
                                        stride =decoder_layers['stride'][i] ))
            # Last one does not have activation function
            if i != dec_len - 2:
                decoder_layers_torch.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers_torch)
        
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output

    def train(self, dataset, criterion = 'mse', optimizer = 'sgd', lr = 0.01, batch_size = 1, epochs = 5, reg = None, verbose = 5):
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Criterion
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'mae':
            criterion = nn.L1Loss()

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

                # Regularization
                if reg != None:
                    if reg == 'l1':
                        p = 1
                    elif reg == 'l2':
                        p = 2
                    regularization = 0.0
                    for param in self.parameters():
                        regularization += torch.norm(param.detach(), p)
                    loss += 0.0005 * regularization

                
                loss.backward()

                # Upgrade Weights and Biases
                optimizer.step()

                batch_n += 1
                if verbose != 0 and batch_n % verbose == 0:
                    print('Epoch:', i + 1, '| Batch number:', batch_n , '| Loss:', loss.item(), '| Time:',   round(time.time() - start_time, 2), 's')
                    start_time = time.time()


            losses.append(loss.item())

        return losses

    def input_gen(self, output, rand_input, epochs = 10, optimizer = 'sgd', criterion = 'mse', lr = 0.01):
        
        # Making sure it has gradients on
        rand_input.requires_grad = True

        if optimizer == 'sgd':
            optimizer = optim.SGD([rand_input], lr = lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam([rand_input], lr = lr)
        
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'mae':
            criterion = nn.L1Loss()

        losses = []

        for _ in range(epochs):
            # Reset optimizer
            optimizer.zero_grad()

            # Forward pass
            pred = self.decoder(rand_input)

            # Calculate loss
            loss = criterion(pred, output)
            loss.backward()

            losses.append(loss.item())

            # Change gradients
            optimizer.step()

        return rand_input, losses