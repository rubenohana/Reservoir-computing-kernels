import numpy as np
import torch
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ESN(torch.nn.Module):
    """
    Implements an Echo State Network.
    Parameters:
      - input_size: size of the input
      - reservoir_size: number of units in the reservoir
      - scale_in: scaling of the input-to-reservoir matrix
      - f: activation function for the state transition function
    """

    def __init__(self, input_size, reservoir_size, bias =0, leak_rate=1, 
                 scale_in=1.0, scale_res=1.0, f='erf', redraw=False):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.scale_in = scale_in
        self.scale_res = scale_res
        self.bias = bias
        self.leak_rate = leak_rate
        if f == 'erf':
            self.f = torch.erf
        if f == 'cos_rbf':
            self.f = lambda x : np.sqrt(2)*torch.cos(x + self.bias).to(device)
        if f == 'heaviside':
            self.f = lambda x: 1 * (x > 0)
        if f == 'sign':
            self.f = torch.sign
                
        #self.f = f
        
        self.redraw = redraw
        
        torch.manual_seed(1)
        self.W_in = torch.randn(reservoir_size, input_size).to(device)
        self.W_res = torch.randn(reservoir_size, reservoir_size).to(device)
        
    def forward(self, input_data, initial_state=None):
        """
        Compute the reservoir states for the given sequence.
        Parameters:
          - input: Input sequence of shape (seq_len, input_size), i.e. (t,d)
        
        Returns: a tensor of shape (seq_len, reservoir_size)
        """
        x = torch.zeros((input_data.shape[0], self.reservoir_size), device=device)

        if initial_state is not None:
            x[0,:] = self.f( self.scale_in * torch.matmul(self.W_in, input_data[0,:]) +
                            self.scale_res * torch.matmul(self.W_res, initial_state) ) / np.sqrt(self.reservoir_size)
        else:
            x[0,:] = self.f( self.scale_in * torch.matmul(self.W_in, input_data[0,:]) ) / np.sqrt(self.reservoir_size)
        
        # I made an important change here, i needs to start at 1 since the first step has been computed already
        for i in range(1, input_data.shape[0]):
            if self.redraw == True:
                torch.manual_seed(i)
                W_inn = self.scale_in*torch.randn((self.reservoir_size, self.input_size)).to(device)
                W_ress = self.scale_res*torch.randn((self.reservoir_size, self.reservoir_size)).to(device)/np.sqrt(self.reservoir_size)
                x[i,:] = (1 - self.leak_rate) * x[i-1, :] + \
                	self.leak_rate * self.f( self.scale_in * torch.matmul(W_inn, input_data[i,:]) + 
                		self.scale_res * torch.matmul(W_ress, x[i-1]) ) / np.sqrt(self.reservoir_size)
            else:
                x[i,:] = (1 - self.leak_rate) * x[i-1, :] + \
                	self.leak_rate * self.f( self.scale_in * torch.matmul(self.W_in, input_data[i,:]) + 
                		self.scale_res * torch.matmul(self.W_res, x[i-1]) ) / np.sqrt(self.reservoir_size)
        
        return x
