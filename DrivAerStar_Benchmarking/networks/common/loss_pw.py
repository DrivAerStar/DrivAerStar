import torch

def get_l2_loss(output, target):
    # output.dim = (batch, N, c)
    # target.dim = (batch, N, c)   
    # output = output.squeeze(-1) 
    # target = target.squeeze(-1) 

    error = output - target
    
    norm_error_sample = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    
    if norm_error_sample.shape[-1] == 1:
        norm_error_channnel = norm_error_sample.squeeze(-1) 
    else:
        norm_error_channnel = torch.mean(norm_error_sample, dim=-1)
    
    # norm_error_channnel.dim = [B, 1]
    norm_error_batch = torch.mean(norm_error_channnel)
    
    return norm_error_batch