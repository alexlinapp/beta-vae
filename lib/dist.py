import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalDistribution:
    nparams = 2

    def _split_params(self, params):
        """
        Internal function for splitting mu, logvar
        Returns tuple(mu, logvar)
        """
        if not torch.is_tensor(params):
            raise TypeError(f"params must be a torch.Tensor, got: {type(params)}")
        
        if params.shape[-1] != NormalDistribution.nparams:
            raise TypeError(f"last dim of params must be 2 for mu and logvar, got shape: {tuple(params.shape)}")    # params.shape gives a torch.size
        
        return params.chunk(2, dim=-1)
    
    def sample(self, params):
        """
        Reparameterization trick
        """
        if not torch.is_floating_point(params):
            raise TypeError(f"params must be floating point, got dtype: {params.dtype}")
        
        mu, logvar = self._split_params(params)

        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn(sigma.shape, device=params.device)

        return mu + eps * sigma

    