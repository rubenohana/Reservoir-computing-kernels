import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def arcsin(input_data, initial_state,scale_res,scale_in = 1):
    n_input = input_data.shape[0]
    input_len = input_data.shape[1]
    input_dim = input_data.shape[2]
    n_res = initial_state.shape[0]
    
    K = torch.ones((n_input,n_input)).to(device) * torch.norm(initial_state)
    for t in range(input_len):
        current_input = input_data[:,t,:].reshape(n_input,input_dim)
        input_gram = torch.matmul(current_input, current_input.t())
        
        diag_res = torch.diag(K)
        diag_in = torch.diag(input_gram)
        renorm_diag = 1 /(1+2*(scale_res**2)*diag_res +2*(scale_in)**2 *diag_in)
        renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
        K = torch.asin(2 * ((scale_res**2)*K + (scale_in**2) *input_gram) * renorm_factor) * 2 / np.pi
    return K.to(device)

def rbf(input_data, initial_state,scale_res,scale_in = 1):
	n_input = input_data.shape[0]
	input_len = input_data.shape[1]
	input_dim = input_data.shape[2]
	n_res = initial_state.shape[0]

	K = torch.ones((n_input,n_input)).to(device) * torch.norm(initial_state)
	for t in range(input_len):
	    current_input = input_data[:,t,:].reshape(n_input,input_dim)
	    input_gram = torch.matmul(current_input, current_input.t())
	    
	    diag_res = torch.diag(K) #x**2
	    diag_in = torch.diag(input_gram) #i**2

	    K = torch.exp(-0.5*(diag_res.reshape(n_input,1))*scale_res**2)\
	        *torch.exp(-0.5*(diag_res.reshape(1,n_input))*scale_res**2)\
	            *torch.exp(K*scale_res**2)\
	                * torch.exp(- 0.5*torch.cdist(current_input,current_input)**2 *scale_in**2)
	return K.to(device)

def acos_heaviside(input_data, initial_state,scale_res,scale_in = 1):
    n_input = input_data.shape[0]
    input_len = input_data.shape[1]
    input_dim = input_data.shape[2]
    n_res = initial_state.shape[0]
    
    K = torch.ones((n_input,n_input)).to(device) * torch.norm(initial_state)
    for t in range(input_len):
        current_input = input_data[:,t,:].reshape(n_input,input_dim)
        input_gram = torch.matmul(current_input, current_input.t())
        
        diag_res = torch.diag(K)
        diag_in = torch.diag(input_gram)
        renorm_diag = 1 /((scale_res**2)*diag_res + (scale_in)**2 * diag_in)
        renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
        K = 0.5 - torch.acos(((scale_res**2)*K + (scale_in**2) *input_gram) * renorm_factor) /(2*np.pi) 
    return K.to(device)

def asin_sign(input_data, initial_state,scale_res,scale_in = 1):
    n_input = input_data.shape[0]
    input_len = input_data.shape[1]
    input_dim = input_data.shape[2]
    n_res = initial_state.shape[0]
    
    K = torch.ones((n_input,n_input)).to(device) * torch.norm(initial_state)
    for t in range(input_len):
        current_input = input_data[:,t,:].reshape(n_input,input_dim)
        input_gram = torch.matmul(current_input, current_input.t())
        
        diag_res = torch.diag(K)
        diag_in = torch.diag(input_gram)
        renorm_diag = 1 /((scale_res**2)*diag_res + (scale_in)**2 * diag_in)
        renorm_factor = torch.sqrt(torch.matmul(renorm_diag.reshape(n_input, 1), renorm_diag.reshape(1, n_input)))
        K = (2/np.pi)*torch.asin(((scale_res**2)*K + (scale_in**2) *input_gram) * renorm_factor)  
    return K.to(device)