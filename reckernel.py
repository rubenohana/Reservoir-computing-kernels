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

        for t in range(n_iter):
            if self.memory_efficient:
                current_input = input_data[t:t+input_len-n_iter, :]
            else:
                current_input = input_data[:, t, :].reshape(n_input, input_dim)
            input_gram = current_input @ current_input.T

            if self.function == 'linear':
                K = self.res_scale**2 * K + self.input_scale**2 * input_gram
            elif self.function == 'arcsin':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)
            elif self.function == 'rbf':
                diag_res = torch.diag(K) #x**2
                diag_in = torch.diag(input_gram) #i**2

                K = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
            elif self.function == 'acos heaviside':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = 0.5 - torch.acos(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor) /(2*np.pi) 
            elif self.function == 'asin sign':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = (2/np.pi)*torch.asin(((self.res_scale**2)*K + (self.input_scale**2) *input_gram) * renorm_factor)
            elif self.function == 'arcsin leak':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K = (1 - self.leak_rate)**2 * K + self.leak_rate**2 *  2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)
        return K.to(device)


    def forward_test(self, train_data, test_data):
        if self.memory_efficient:
            n_iter = self.n_iter
            train_len, input_dim = train_data.shape
            n_train = train_len - n_iter
            test_len = test_data.shape[0]
            n_test = test_len - n_iter

            diag_res_train = torch.ones(n_train).to(device)
            diag_res_test = torch.ones(n_test).to(device)
            K = torch.ones(n_test, n_train).to(device)

            for t in range(n_iter):
                current_train = train_data[t:t+n_train, :]
                current_test = test_data[t:t+n_test, :]
                input_gram = current_test @ current_train.T

                if self.function == 'arcsin':
                    renorm_factor = 1 / torch.sqrt(
                        (1 + 2*self.res_scale**2*diag_res_test+2*self.input_scale**2*torch.sum((current_test)**2, dim=1)).reshape(n_test, 1) @
                        (1 + 2*self.res_scale**2*diag_res_train+2*self.input_scale**2*torch.sum(current_train**2, dim=1)).reshape(1, n_train)
                        )
                    K = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)

                    diag_res_train = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * diag_res_train + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1)) /
                        (1 + 2 * self.res_scale**2 * diag_res_train + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1))
                        )
                    diag_res_test = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1)) /
                        (1 + 2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1))
                        )
                elif self.function == 'linear':
                    K = self.res_scale**2 * K + self.input_scale**2 * input_gram

            return K.to(device)

        else:
            input_data = torch.cat((train_data, test_data), dim=0)
            n_input, n_iter, input_dim = input_data.shape
            n_train = train_data.shape[0]
            n_test = test_data.shape[0]

            K = self.forward(input_data)
            return K[-n_test:, :n_train].to(device)

    def train(self, K, y, method='cholesky', alpha=1e-3):
        if method == 'cholesky':
            # This technique uses the Cholesky decomposition
            K.view(-1)[::len(K)+1] += alpha  # add elements on the diagonal inplace
            L = torch.cholesky(K, upper=False)
            return torch.cholesky_solve(y, L, upper=False)
        elif method == 'sklearn ridge':
            from sklearn.linear_model import Ridge
            clf = Ridge(fit_intercept=False, alpha=alpha)
            clf.fit(K.cpu().numpy(), y.cpu().numpy())
            return torch.from_numpy(clf.coef_.T).to(device)
