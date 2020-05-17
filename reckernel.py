import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class RecKernel():
    """
    Implements a recurrent kernel
    """

    def __init__(self, function='arcsin', 
                 res_scale=1, input_scale=1):
        self.function = function
        self.res_scale = res_scale
        self.input_scale = input_scale
        
    def forward(self, input_data):
        n_input = input_data.shape[0]
        input_len = input_data.shape[1]
        input_dim = input_data.shape[2]

        K = torch.ones((n_input,n_input)).to(device)  # the norm of initial_state has been removed
        # TODO: merge the different functions
        if self.function == 'arcsin':
            for t in range(input_len):
                current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T

                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)
            return K.to(device)  # TOCHECK: maybe to(device) is not necessary
        elif self.function == 'rbf':
            for t in range(input_len):
                current_input = input_data[:,t,:].reshape(n_input,input_dim)
                input_gram = torch.matmul(current_input, current_input.t())
                
                diag_res = torch.diag(K) #x**2
                diag_in = torch.diag(input_gram) #i**2

                K = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
            return K.to(device)
        elif self.function == 'acos heaviside':
            for t in range(input_len):
                current_input = input_data[:,t,:].reshape(n_input,input_dim)
                input_gram = torch.matmul(current_input, current_input.t())
                
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = 0.5 - torch.acos(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor) /(2*np.pi) 
            return K.to(device)
        elif self.function == 'asin sign':
            for t in range(input_len):
                current_input = input_data[:,t,:].reshape(n_input,input_dim)
                input_gram = torch.matmul(current_input, current_input.t())
                
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = (2/np.pi)*torch.asin(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor)  
            return K.to(device)
