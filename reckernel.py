import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class RecKernel():
    """
    Implements a recurrent kernel.
    Parameters:
    - function: kernel function of the recurrent kernel
    - res_scale: scale of the corresponding reservoir weights
    - input_scale: scale of the corresponding input-to-reservoir weights
    - bias_scale: scale of the corresponding random additive bias
    - memory_efficient: if True, uses more efficient schemes for time series prediction
    - n_iter: number of iterations (stability => only a finite number of iterations is necessary)
    - step: if > 1, allows to subsample the time series to save memory
    """
    def __init__(self, function='arcsin', 
                 res_scale=1, input_scale=1, bias_scale=0, 
                 memory_efficient=True, n_iter=50, step=1, device=device):
        self.function = function
        self.res_scale = res_scale
        self.input_scale = input_scale
        self.bias_scale = bias_scale
        self.device = device

        self.memory_efficient = memory_efficient
        self.n_iter = n_iter
        self.step = step
        
    def forward(self, input_data, initial_K=None, bypass=None, return_diag=False):
        """
        Computes the forward pass of a recurrent kernel.
        Parameters:
        - input_data: data of shape (input_len, input_dim) or (n_input, n_iter, input_dim)
        - initial_K: initial value of the kernel matrix if provided
        - bypass: if True, uses the memory-inefficient alternative (see the first lines)
        - return_diag: if True, returns the diagonal norm matrix for renormalization

        Returns:
        - K: kernel matrix of size (n_input, n_input)
        - diag_res: if return_diag==True, returns this diagonal matrix for later uses
        """
        if self.memory_efficient and bypass is None:
            input_len, input_dim = input_data.shape
            n_iter = self.n_iter
            n_input = input_len - n_iter + 1
        else:
            n_input, n_iter, input_dim = input_data.shape

        if initial_K is None:
            K = torch.ones((n_input, n_input)).to(self.device)
        else:
            K = initial_K

        for t in range(n_iter):
            if self.memory_efficient and bypass is None:
                current_input = input_data[t:t+n_input, :]
            else:
                current_input = input_data[:, t, :].reshape(n_input, input_dim)
            input_gram = current_input @ current_input.T

            if self.function == 'linear':
                K = self.res_scale**2 * K + self.input_scale**2 * input_gram + self.bias_scale**2
            elif self.function == 'arcsin':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in + 2 * self.bias_scale**2)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram + 2 * self.bias_scale**2) * renorm_factor)
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
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + 2 * self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = 0.5 - torch.acos(((self.res_scale**2)*K + (self.input_scale**2) *input_gram + 2 * self.bias_scale**2) * renorm_factor) /(2*np.pi) 
            elif self.function == 'asin sign':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + 2 * self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K = (2/np.pi)*torch.asin(((self.res_scale**2)*K + (self.input_scale**2) *input_gram + 2 * self.bias_scale**2) * renorm_factor)
            elif self.function == 'acos relu':
                diag_res = torch.diag(K)
                diag_in = torch.diag(input_gram)
                
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_diag_inv = ((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_factor_inv = torch.sqrt(torch.matmul(renorm_diag_inv.reshape(n_input, 1), renorm_diag_inv.reshape(1, n_input)))
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                correl = (self.res_scale**2)*K + (self.input_scale**2) *input_gram + self.bias_scale**2
                correl_norm = correl*renorm_factor
                
                K = (1/20*np.pi) * renorm_factor_inv *(correl_norm*torch.acos(-correl_norm) + torch.sqrt(1-correl_norm**2))
        if return_diag:
            return K.to(self.device), diag_res
        else:
            return K.to(self.device)

    def forward_test(self, train_data, test_data, initial_K=None, bypass=None):
        """
        Computes the kernel matrix between train and test data.
        Parameters:
        - train_data: data of shape (train_len, input_dim) or (n_train, n_iter, input_dim)
        - test_data: data of shape (test_len, input_dim) or (n_test, n_iter, input_dim)
        - initial_K: initial value of the kernel matrix if provided
        - bypass: if True, uses the memory-inefficient alternative

        Returns:
        - K: kernel matrix of size (n_input, n_input)
        - The other outputs should be cleaned
        """
        if self.memory_efficient and bypass is None:
            n_iter = self.n_iter
            train_len, input_dim = train_data.shape
            n_train = train_len - n_iter + 1
            test_len = test_data.shape[0]
            n_test = test_len - n_iter + 1

            diag_res_train = torch.ones(n_train).to(self.device)
            diag_res_test = torch.ones(n_test).to(self.device)
            if initial_K is None:
                K = torch.ones((n_test, n_train)).to(self.device)
            else:
                K = initial_K

            for t in range(n_iter):
                current_train = train_data[t:t+n_train, :]
                current_test = test_data[t:t+n_test, :]
                input_gram = current_test @ current_train.T

                if self.function == 'arcsin':
                    renorm_factor = 1 / torch.sqrt(
                        (1 + 2*self.res_scale**2*diag_res_test+2*self.input_scale**2*torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2).reshape(n_test, 1) @
                        (1 + 2*self.res_scale**2*diag_res_train+2*self.input_scale**2*torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2).reshape(1, n_train)
                        )
                    K = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram + 2 * self.bias_scale**2) * renorm_factor)

                    diag_res_train = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * diag_res_train + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2) /
                        (1 + 2 * self.res_scale**2 * diag_res_train + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2)
                        )
                    diag_res_test = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2) /
                        (1 + 2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2)
                        )
                elif self.function == 'linear':
                    K = self.res_scale**2 * K + self.input_scale**2 * input_gram + self.bias_scale**2

            if self.function == 'arcsin':
                return K.to(self.device), diag_res_train, diag_res_test
            else:
                return K.to(self.device)
        elif bypass is True:
            input_data = torch.cat((train_data, test_data), dim=0)
            n_input, n_iter, input_dim = input_data.shape
            n_train = train_data.shape[0]
            n_test = test_data.shape[0]

            K, diag_res = self.forward(input_data, bypass=True, return_diag=True)
            return K[-n_test:, :n_train].to(self.device), diag_res[:n_train], diag_res[n_train:]
        else:
            input_data = torch.cat((train_data, test_data), dim=0)
            n_input, n_iter, input_dim = input_data.shape
            n_train = train_data.shape[0]
            n_test = test_data.shape[0]

            K = self.forward(input_data, bypass=True)
            return K[-n_test:, :n_train].to(self.device)

    def train(self, K, y, method='cholesky', alpha=1e-3):
        """
        Compute the output weights with a linear regression.
        Parameters:
          - K: input sequence of shape (n_input, n_input)
          - y: target output (n_input, out_dim)
          - method: "cholesky" or "sklearn ridge"
          - alpha: L2-regularization parameter
        
        Returns: a tensor of shape (n_input, out_dim)
        """
        if method == 'cholesky':
            # This technique uses the Cholesky decomposition
            K.view(-1)[::len(K)+1] += alpha  # add elements on the diagonal inplace
            L = torch.cholesky(K, upper=False)
            return torch.cholesky_solve(y, L, upper=False)
        elif method == 'sklearn ridge':
            from sklearn.linear_model import Ridge
            clf = Ridge(fit_intercept=False, alpha=alpha)
            clf.fit(K.cpu().numpy(), y.cpu().numpy())
            return torch.from_numpy(clf.coef_.T).to(self.device)

    def rec_pred(self, K, train_data, test_data, output_w, n_rec, diag_res_train, diag_res_test, concat):
        # Retrieve parameters
        input_len, input_dim = train_data.shape
        n_points = K.shape[0]
        out_len = output_w.shape[1]
        single_pred_length = out_len // input_dim

        concat_train = torch.zeros(input_len-self.n_iter+1, self.n_iter, input_dim).to(self.device)
        for i in range(input_len-self.n_iter+1):
            concat_train[i, :, :] = train_data[i:i+self.n_iter, :]
        concat_test = torch.zeros(n_points, self.n_iter, input_dim).to(self.device)
        for i in range(n_points):
            concat_test[i, :, :] = test_data[i:i+self.n_iter, :]

        # Prediction
        total_pred = torch.zeros(n_points, out_len*(n_rec+1))
        if concat != 0:
            final_train_data = train_data[self.n_iter-1:, :]
            final_test_data = test_data[self.n_iter-1:, :]
            gram_mat = final_test_data @ final_train_data.T

            pred_data = (K + concat * gram_mat) @ output_w
        else:
            pred_data = (K @ output_w)
        total_pred[:, :out_len] = pred_data
        pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)
        for i_rec in range(n_rec):
            concat_test = torch.cat((concat_test, pred_data), dim=1)
            concat_test = concat_test[:, single_pred_length:, :]
            K = self.forward_test(concat_train, concat_test, bypass=True)

            if concat != 0:
                final_train_data = train_data[self.n_iter-1:, :]
                final_test_data = pred_data[:, single_pred_length-1, :].reshape(n_points, input_dim)
                gram_mat = final_test_data @ final_train_data.T

                pred_data = (K + concat * gram_mat) @ output_w
            else:
                pred_data = (K @ output_w)
            total_pred[:, (i_rec+1)*out_len:(i_rec+2)*out_len] = pred_data
            pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)

        return total_pred

    def rec_pred_fast(self, K, train_data, test_data, output_w, n_rec, diag_res_train, diag_res_test, concat):
        # Retrieve parameters
        input_len, input_dim = train_data.shape
        n_points = K.shape[0]
        out_len = output_w.shape[1]
        single_pred_length = out_len // input_dim

        # Create matrices for the first values of K, computed using forward_test
        small_train_data = train_data[:self.n_iter+single_pred_length, :]
        concat_train = torch.zeros(single_pred_length, self.n_iter, input_dim).to(self.device)
        for i in range(single_pred_length):
            concat_train[i, :, :] = small_train_data[i:i+self.n_iter, :]
        concat_test = torch.zeros(n_points, self.n_iter, input_dim).to(self.device)
        for i in range(n_points):
            concat_test[i, :, :] = test_data[i:i+self.n_iter, :]

        # Prediction
        total_pred = torch.zeros(n_points, out_len*(n_rec+1))
        if concat != 0:
            final_train_data = train_data[self.n_iter-1:, :]
            final_test_data = test_data[self.n_iter-1:, :]
            gram_mat = final_test_data @ final_train_data.T

            pred_data = (K + concat * gram_mat) @ output_w
        else:
            pred_data = (K @ output_w)
        total_pred[:, :out_len] = pred_data
        pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)
        for i_rec in range(n_rec):
            # We first compute the last elements using an online method
            shrinking_K = K
            shrinking_diag_res_train = diag_res_train
            for t in range(single_pred_length):
                current_train = train_data[self.n_iter+t:, :]  # we remove one element at each iteration
                current_test = pred_data[:, t, :].reshape(n_points, input_dim)
                input_gram = current_test @ current_train.T

                if self.function == 'arcsin':
                    # print(shrinking_diag_res_train.shape)
                    # print(current_train.shape)
                    renorm_factor = 1 / torch.sqrt(
                        (1 + 2*self.res_scale**2*diag_res_test+2*self.input_scale**2*torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2).reshape(-1, 1) @
                        (1 + 2*self.res_scale**2*shrinking_diag_res_train[:-1]+2*self.input_scale**2*torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2).reshape(1, -1)
                        )
                    shrinking_K = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * shrinking_K[:, :-1] + 2 * self.input_scale**2 * input_gram + 2 * self.bias_scale**2) * renorm_factor)  # we remove the last element of K

                    shrinking_diag_res_train = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * shrinking_diag_res_train[:-1] + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2) /
                        (1 + 2 * self.res_scale**2 * shrinking_diag_res_train[:-1] + 2 * self.input_scale**2 * torch.sum(current_train**2, dim=1) + 2 * self.bias_scale**2)
                        )
                    # shrinking_diag_res_train = shrinking_diag_res_train[:-1]
                    diag_res_test = 2 / np.pi * torch.asin(
                        (2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2) /
                        (1 + 2 * self.res_scale**2 * diag_res_test + 2 * self.input_scale**2 * torch.sum((current_test)**2, dim=1) + 2 * self.bias_scale**2)
                        )
                elif self.function == 'linear':
                    shrinking_K = self.res_scale**2 * shrinking_K[:, :-1] + self.input_scale**2 * input_gram + self.bias_scale**2
            K[:, single_pred_length:] = shrinking_K

            # We complete the Gram matrix by using the forward_pred function
            concat_test = torch.cat((concat_test, pred_data), dim=1)
            concat_test = concat_test[:, single_pred_length:, :]
            K[:, :single_pred_length], _, _ = self.forward_test(concat_train, concat_test, bypass=True)

            if concat != 0:
                final_train_data = train_data[self.n_iter-1:, :]
                final_test_data = pred_data[:, single_pred_length-1, :].reshape(n_points, input_dim)
                gram_mat = final_test_data @ final_train_data.T

                pred_data = (K + concat * gram_mat) @ output_w
            else:
                pred_data = (K @ output_w)
            total_pred[:, (i_rec+1)*out_len:(i_rec+2)*out_len] = pred_data
            pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)

        return total_pred

    def rec_pred_concat(self, K, concat_train, concat_test, output_w, n_rec, diag_res_train, diag_res_test, concat):
        # Retrieve parameters
        input_len, _, input_dim = concat_train.shape
        n_points = K.shape[0]
        out_len = output_w.shape[1]
        single_pred_length = out_len // input_dim

        # Prediction
        total_pred = torch.zeros(n_points, out_len*(n_rec+1))
        if concat != 0:
            final_train_data = concat_train[:, self.n_iter-1, :]
            final_test_data = concat_test[:, self.n_iter-1, :]
            gram_mat = final_test_data @ final_train_data.T

            pred_data = (K + concat * gram_mat) @ output_w
        else:
            pred_data = (K @ output_w)
        total_pred[:, :out_len] = pred_data
        pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)
        for i_rec in range(n_rec):
            concat_test = torch.cat((concat_test, pred_data), dim=1)
            concat_test = concat_test[:, single_pred_length:, :]
            K, _, _ = self.forward_test(concat_train, concat_test, bypass=True)

            if concat != 0:
                final_train_data = concat_train[:, self.n_iter-1, :]
                final_test_data = pred_data[:, single_pred_length-1, :].reshape(n_points, input_dim)
                gram_mat = final_test_data @ final_train_data.T

                pred_data = (K + concat * gram_mat) @ output_w
            else:
                pred_data = (K @ output_w)
            total_pred[:, (i_rec+1)*out_len:(i_rec+2)*out_len] = pred_data
            pred_data = pred_data.reshape(n_points, single_pred_length, input_dim)

        return total_pred

    def test_stability(self, input_data, initial_K1=None, initial_K2=None, bypass=None):
        """ 
        Computes the distance between two different kernel matrices with the same input.
        Parameters:
        - input_data: 
        - initial_K1, initial_K2: initial matrices
        """
        if self.memory_efficient and bypass is None:
            input_len, input_dim = input_data.shape
            n_iter = self.n_iter
            n_input = input_len - n_iter + 1
        else:
            n_input, n_iter, input_dim = input_data.shape

        if initial_K1 is None:
            K1 = torch.ones((n_input, n_input)).to(self.device)
        else:
            K1 = initial_K1
        if initial_K2 is None:
            K2 = torch.zeros((n_input, n_input)).to(self.device)
        else:
            K2 = initial_K2

        res = torch.zeros(n_iter).to(self.device)
        for t in range(n_iter):
            if self.memory_efficient and bypass is None:
                current_input = input_data[t:t+n_input, :]
            else:
                current_input = input_data[:, t, :].reshape(n_input, input_dim)
            input_gram = current_input @ current_input.T

            if self.function == 'linear':
                K1 = self.res_scale**2 * K1 + self.input_scale**2 * input_gram + self.bias_scale**2
                K0 = self.res_scale**2 * K0 + self.input_scale**2 * input_gram + self.bias_scale**2
            elif self.function == 'arcsin':
                diag_in = torch.diag(input_gram)

                diag_res = torch.diag(K1)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K1 = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K1 + 2 * self.input_scale**2 * input_gram + self.bias_scale**2) * renorm_factor)
                diag_res = torch.diag(K2)
                renorm_diag = 1 / (
                    1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(
                    renorm_diag.reshape(n_input, 1) @ renorm_diag.reshape(1, n_input))
                K2 = 2 / np.pi * torch.asin(
                    (2 * self.res_scale**2 * K2 + 2 * self.input_scale**2 * input_gram + self.bias_scale**2) * renorm_factor)
            elif self.function == 'rbf':
                diag_in = torch.diag(input_gram)

                diag_res = torch.diag(K1)
                K1 = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K1*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
                diag_res = torch.diag(K2)
                K2 = torch.exp(-0.5*(diag_res.reshape(n_input,1))*self.res_scale**2)\
                    *torch.exp(-0.5*(diag_res.reshape(1,n_input))*self.res_scale**2)\
                        *torch.exp(K2*self.res_scale**2)\
                            * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *self.input_scale**2)
            elif self.function == 'acos heaviside':
                diag_in = torch.diag(input_gram)

                diag_res = torch.diag(K1)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K1 = 0.5 - torch.acos(((self.res_scale**2)*K1 + (self.input_scale**2) *input_gram + self.bias_scale**2) * renorm_factor) /(2*np.pi)
                diag_res = torch.diag(K2)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K2 = 0.5 - torch.acos(((self.res_scale**2)*K2 + (self.input_scale**2) *input_gram + self.bias_scale**2) * renorm_factor) /(2*np.pi)
            elif self.function == 'asin sign':
                diag_in = torch.diag(input_gram)

                diag_res = torch.diag(K1)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K1 = (2/np.pi)*torch.asin(((self.res_scale**2)*K1 + (self.input_scale**2) *input_gram + self.bias_scale**2) * renorm_factor)
                diag_res = torch.diag(K2)
                renorm_diag = 1 /((self.res_scale**2)*diag_res + (self.input_scale)**2 * diag_in + self.bias_scale**2)
                renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
                K2 = (2/np.pi)*torch.asin(((self.res_scale**2)*K2 + (self.input_scale**2) *input_gram + self.bias_scale**2) * renorm_factor)
            res[t] = torch.mean(torch.abs(K2-K1)**2)
        return res

