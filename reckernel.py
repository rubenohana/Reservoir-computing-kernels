import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class RecKernel():
    """
    Implements a recurrent kernel
    """

    def __init__(self, function='arcsin', 
                 res_scale=1, input_scale=1,
                 memory_efficient=True, n_iter=50, leak_rate=1):
        self.function = function
        self.res_scale = res_scale
        self.input_scale = input_scale
        self.leak_rate = leak_rate

        self.memory_efficient = memory_efficient
        self.n_iter = n_iter
        
    def forward(self, input_data, initial_K=None):
        if self.memory_efficient:
            input_len, input_dim = input_data.shape
            n_iter = self.n_iter
            n_input = input_len - n_iter
        else:
            n_input, n_iter, input_dim = input_data.shape

        if initial_K is None:
            K = torch.ones((n_input, n_input)).to(device)
        else:
            K = initial_K
        # TODO: merge the different functions
        if self.function == 'arcsin':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
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
        if self.function == 'arcsin leak':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T

                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K = (1 - self.leak_rate**2) * K + self.leak_rate**2 *  2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)
            return K.to(device)  # TOCHECK: maybe to(device) is not necessary
        elif self.function == 'rbf':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                
                diag_res = torch.diag(K) #x**2
                diag_in = torch.diag(input_gram) #i**2

                K = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
            return K.to(device)
        elif self.function == 'acos heaviside':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = 0.5 - torch.acos(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor) /(2*np.pi) 
            return K.to(device)
        elif self.function == 'asin sign':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = (2/np.pi)*torch.asin(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor)  
            return K.to(device)

    def forward_test(self, train_data, test_data):
        if self.memory_efficient:
            pass
        else:
            input_data = torch.cat((train_data, test_data), dim=0)
            n_input, n_iter, input_dim = input_data.shape
            n_train = train_data.shape[0]
            n_test = test_data.shape[0]

            K = self.forward(input_data)
            return K[-n_test:, :n_train].to(device)

    def stability_test(self, input_data):
        if self.memory_efficient:
            input_len, input_dim = input_data.shape
            n_iter = self.n_iter
            n_input = input_len - n_iter
        else:
            n_input, n_iter, input_dim = input_data.shape

        res = torch.zeros(n_iter).to(device)
        K0 = torch.zeros((n_input, n_input)).to(device)
        K1 = torch.ones((n_input, n_input)).to(device)

        # TODO: merge the different functions
        if self.function == 'arcsin':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                diag_in = torch.diag(input_gram)

                diag_res = torch.diag(K1)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K1 = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K1 + 2 * self.input_scale**2 * input_gram) * renorm_factor)

                diag_res = torch.diag(K0)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K0 = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K0 + 2 * self.input_scale**2 * input_gram) * renorm_factor)

                res[t] = torch.mean(torch.abs(K0-K1)**2)
        elif self.function == 'rbf':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                diag_in = torch.diag(input_gram) #i**2
                
                diag_res = torch.diag(K1) #x**2
                K1 = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K1*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
                diag_res = torch.diag(K0) #x**2
                K0 = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K1*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)

                res[t] = torch.mean(torch.abs(K0-K1)**2)
        elif self.function == 'acos heaviside':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                diag_in = torch.diag(input_gram)
                
                diag_res = torch.diag(K1)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K1 = 0.5 - torch.acos(((self.res_scale**2)*K1 + (self.input_scale**2) *input_gram) * renorm_factor) /(2*np.pi) 
                diag_res = torch.diag(K0)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K0 = 0.5 - torch.acos(((self.res_scale**2)*K0 + (self.input_scale**2) *input_gram) * renorm_factor) /(2*np.pi) 

                res[t] = torch.mean(torch.abs(K0-K1)**2)
        elif self.function == 'asin sign':
            for t in range(n_iter):
                if self.memory_efficient:
                    current_input = input_data[t:t+input_len-n_iter, :]
                else:
                    current_input = input_data[:, t, :].reshape(n_input, input_dim)
                input_gram = current_input @ current_input.T
                diag_in = torch.diag(input_gram)
                
                diag_res = torch.diag(K1)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K1 = (2/np.pi)*torch.asin(((self.res_scale**2)*K1 + (self.input_scale**2) *input_gram) * renorm_factor)  
                diag_res = torch.diag(K0)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K0 = (2/np.pi)*torch.asin(((self.res_scale**2)*K0 + (self.input_scale**2) *input_gram) * renorm_factor)  

                res[t] = torch.mean(torch.abs(K0-K1)**2)
        return res