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
      - res_size: number of units in the reservoir
      - input_scale: scaling of the input-to-reservoir matrix
      - f: activation function for the state transition function
    """

    def __init__(self, input_size, res_size, 
                 input_scale=1.0, res_scale=1.0, 
                 bias=0, leak_rate=1, f='erf', 
                 redraw=False, redraw_batch=1, 
                 seed=1):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.res_size = res_size
        self.input_scale = input_scale
        self.res_scale = res_scale
        self.bias = bias
        self.leak_rate = leak_rate
        self.redraw = redraw
        self.redraw_batch = redraw_batch
        self.seed = seed
        
        #self.f = f
        if f == 'erf':
            self.f = torch.erf
        if f == 'cos_rbf':
            self.f = lambda x : np.sqrt(2)*torch.cos(x + self.bias).to(device)
        if f == 'heaviside':
            self.f = lambda x: 1 * (x > 0)
        if f == 'sign':
            self.f = torch.sign
                
        torch.manual_seed(self.seed)
        self.W_in = torch.randn(res_size, input_size).to(device)
        self.W_res = torch.randn(res_size, res_size).to(device)
        
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
                x[i,:] = \
                    (1 - self.leak_rate) * x[i-1, :] + \
                    self.leak_rate * self.f( 
                        self.input_scale * self.W_in @ input_data[i, :] + 
                        self.res_scale * self.W_res @ x[i-1, :]
                        ) / np.sqrt(self.res_size)
            else:
                torch.manual_seed(self.seed + i)
                W_in_redraw = self.scale_in*torch.randn((self.reservoir_size, self.input_size)).to(device)
                W_res_redraw = self.scale_res*torch.randn((self.reservoir_size, self.reservoir_size)).to(device)
                x[i,:] = \
                    (1 - self.leak_rate) * x[i-1, :] + \
                    self.leak_rate * self.f( 
                        self.input_scale * W_in_redraw @ input_data[i, :] + 
                        self.res_scale * W_res_redraw @ x[i-1, :]
                        ) / np.sqrt(self.res_size)

        return x

    def train(self, X, y, method='cholesky', alpha=1e-3):
        if method == 'cholesky':
            # This technique uses the Cholesky decomposition
            # It should be fast when res_size < seq_len
            Xt_y = X.T @ y  # size (n_res, k)
            K = X.T @ X  # size (n_res, n_res)
            K.view(-1)[::len(K)+1] += alpha  # add elements on the diagonal inplace
            L = torch.cholesky(K, upper=False)
            return torch.cholesky_solve(Xt_y, L, upper=False)
        elif method == 'sklearn ridge':
            from sklearn.linear_model import Ridge
            clf = Ridge(fit_intercept=False, alpha=alpha)
            clf.fit(X.cpu().numpy(), y.cpu().numpy())
            return torch.from_numpy(clf.coef_.T).to(device)
