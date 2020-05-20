import numpy as np
import torch
from tqdm import tqdm

from hadamard_cuda import hadamard_transform

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ESN(torch.nn.Module):
    """
    Implements an Echo State Network.
    Parameters:
      - input_size: size of the input
      - res_size: number of units in the reservoir
      - input_scale: scaling of the input-to-reservoir matrix
      - f: activation function for the state transition function
    """

    def __init__(self, input_size, res_size, 
                 random_projection='gaussian', 
                 input_scale=1.0, res_scale=1.0, 
                 bias=0, leak_rate=1, f='erf', 
                 redraw=False, 
                 seed=1):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.res_size = res_size
        self.input_scale = input_scale
        self.res_scale = res_scale
        self.bias = bias
        self.leak_rate = leak_rate
        self.redraw = redraw
        self.random_projection = random_projection
        self.seed = seed
        
        #self.f = f
        if f == 'erf':
            self.f = torch.erf
        if f == 'cos_rbf':
            torch.manual_seed(1)
            self.bias = 2 * np.pi * torch.rand(self.res_size).to(device)
            self.f = lambda x : np.sqrt(2)*torch.cos(x + self.bias)
        if f == 'heaviside':
            self.f = lambda x: 1 * (x > 0)
        if f == 'sign':
            self.f = torch.sign
                
        torch.manual_seed(self.seed)
        if self.random_projection == 'structured':
            # We use the library from https://github.com/HazyResearch/structured-nets
            self.had_dim = int(2**np.ceil(np.log2(self.input_size+self.res_size)))
            pad_dim = self.had_dim - self.res_size - self.input_size
            self.zero_pad = torch.zeros(pad_dim).to(device)  # to avoid generating a new vector at every iteration
        if not self.redraw:
            if self.random_projection == 'gaussian':
                self.W_in = torch.randn(res_size, input_size).to(device)
                self.W_res = torch.randn(res_size, res_size).to(device)
            elif self.random_projection == 'structured':
                self.diag1 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
                self.diag2 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
                self.diag3 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
        
    def forward(self, input_data, initial_state=None):
        """
        Compute the reservoir states for the given sequence.
        Parameters:
          - input: Input sequence of shape (seq_len, input_size), i.e. (t,d)
        
        Returns: a tensor of shape (seq_len, res_size)
        """
        seq_len = input_data.shape[0]
        x = torch.zeros((seq_len, self.res_size), device=device)
        if initial_state is not None:
            x[-1, :] = initial_state

        for i in range(seq_len):
            if not self.redraw:
                if self.random_projection == 'gaussian':
                    x[i,:] = \
                        (1 - self.leak_rate) * x[i-1, :] + \
                        self.leak_rate * self.f( 
                            self.input_scale * self.W_in @ input_data[i, :] + 
                            self.res_scale * self.W_res @ x[i-1, :]
                            ) / np.sqrt(self.res_size)
                elif self.random_projection == 'structured':
                    u = torch.cat((
                        self.input_scale * input_data[i, :], 
                        self.res_scale * x[i-1, :], 
                        self.zero_pad
                        ))
                    v1 = hadamard_transform(self.diag1 * u)
                    v2 = hadamard_transform(self.diag2 * v1)
                    v3 = hadamard_transform(self.diag3 * v2)
                    v3 /= self.had_dim
                    # v3 /= np.sqrt(self.had_dim**3) / np.sqrt(self.res_size+self.input_size)  # = np.sqrt(self.had_dim ** 3) / np.sqrt(self.had_dim)
                    v3 = v3[:self.res_size]
                    x[i,:] = \
                        (1 - self.leak_rate) * x[i-1, :] + \
                        self.leak_rate * self.f(v3) / np.sqrt(self.res_size)
            else:
                torch.manual_seed(self.seed + i)
                if self.random_projection == 'gaussian':
                    W_in_redraw = self.input_scale * torch.randn((self.res_size, self.input_size)).to(device)
                    input_prod = W_in_redraw @ input_data[i, :]
                    del W_in_redraw
                    W_res_redraw = self.res_scale * torch.randn((self.res_size, self.res_size)).to(device)
                    res_prod = W_res_redraw @ x[i-1, :]
                    del W_res_redraw
                    x[i,:] = \
                        (1 - self.leak_rate) * x[i-1, :] + \
                        self.leak_rate * self.f( 
                            self.input_scale * input_prod + 
                            self.res_scale * res_prod
                            ) / np.sqrt(self.res_size)
                elif self.random_projection == 'structured':
                    self.diag1 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
                    self.diag2 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
                    self.diag3 = 2 * torch.randint(0, 2, (self.had_dim, )).to(device) - 1
                    u = torch.cat((
                        self.input_scale * input_data[i, :], 
                        self.res_scale * x[i-1, :], 
                        self.zero_pad
                        ))
                    v1 = hadamard_transform(self.diag1 * u)
                    v2 = hadamard_transform(self.diag2 * v1)
                    v3 = hadamard_transform(self.diag3 * v2)
                    v3 /= self.had_dim  # = np.sqrt(self.had_dim ** 3) / np.sqrt(self.had_dim)
                    v3 = v3[:self.res_size]
                    x[i,:] = \
                        (1 - self.leak_rate) * x[i-1, :] + \
                        self.leak_rate * self.f(v3) / np.sqrt(self.res_size)

        return x

    def train(self, X, y, method='cholesky', alpha=1e-3):
        if method == 'cholesky':
            # This technique uses the Cholesky decomposition
            # It should be fast when res_size < seq_len
            Xt_y = X.T @ y  # size (res_size, k)
            K = X.T @ X  # size (res_size, res_size)
            K.view(-1)[::len(K)+1] += alpha  # add elements on the diagonal inplace
            L = torch.cholesky(K, upper=False)
            return torch.cholesky_solve(Xt_y, L, upper=False)
        elif method == 'sklearn ridge':
            from sklearn.linear_model import Ridge
            clf = Ridge(fit_intercept=False, alpha=alpha)
            clf.fit(X.cpu().numpy(), y.cpu().numpy())
            return torch.from_numpy(clf.coef_.T).to(device)
