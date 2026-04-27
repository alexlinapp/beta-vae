import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from lib.dist import NormalDistribution as ND



class MLPEncoder(nn.Module):
    """
    
    """
    def __init__(self, latents):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1200),
            nn.ReLU()
        )

        self.mean = nn.Linear(in_features=1200, out_features=latents)
        self.logvar = nn.Linear(in_features=1200, out_features=latents)

    def forward(self, x):
        x = self.model(x)
        mean_x = self.mean(x)
        logvar_x = self.logvar(x)
        return mean_x, logvar_x

    

class MLPDecoder(nn.Module):
    """
    
    """
    def __init__(self, latents):
        super().__init__()
        self.latents = latents
        self.model = nn.Sequential(
            nn.Linear(in_features=latents, out_features=1200),
            nn.Tanh(),
            nn.Linear(in_features=1200, out_features=1200),
            nn.Tanh(),
            nn.Linear(in_features=1200, out_features=1200),
            nn.Tanh(),
            nn.Linear(in_features=1200, out_features=4096)  # no need for sigmoid since BCELogitsLoss handles for us
        )

    def forward(self, x):
        x = self.model(x)
        return x.reshape(-1, 64, 64)
    



class VAE(nn.Module):
    def __init__(self, latent_dim, encoder, decoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean_x, logvar_x):
        sigma_x = torch.exp(0.5 * logvar_x)
        eps = torch.randn(logvar_x.shape, device=logvar_x.device)
        return mean_x + sigma_x * eps

    def forward(self, x):
        mean_x, logvar_x = self.encoder(x)
        z = self.reparameterize(mean_x, logvar_x)
        recon_x = self.decoder(z)

        return recon_x, mean_x, logvar_x, z
    
    

# Loss Functions

def bernoulli_nll_from_logits(logits, x):
    """
    Compute log p(x|z) assuming Bernoulli Likelihood

    Assume BCE and that p(x|z;theta) = prod_{ij} p_{ij}^{x_ij} (1-p_{ij})^{1- x_{ij}}
    x_{ij} in {0, 1} observed binary pixel value
    p_{ij} = p(x_{ij}=1 | z), decoder's predicted probability. Decoder outputs the distribution parameters, more specifically the logits
    that will be converted into p_{ij}  
    """

    bce = F.binary_cross_entropy_with_logits(logits, x.to(torch.float32), reduction="none")

    return bce.reshape(bce.shape[0], -1).sum(axis=1)

def gaussian_nll_from_mean(recon_mean, x, sigma=1.0):
    """
    Gaussian Likelihood:

    Assumes:
        p(x|z) ~ N(recon_mean, sigma^2 I)
        i.e. p(x|z) = prod_{ij} N(recon_mean_{ij}, sigma^2), that is pixel values are CONDITIONALLy independent, since covariance matrix = I,
        so all diagonal

        Note this is equivalnet to computing the MSE/trying to minimize the MSE
    
    Compute: log p(x|z)
    """

    var = sigma**2
    logvar = math.log(var)
    
    logpx_per_pixel = 0.5 * ( math.log(2 * math.pi) + logvar + (x.to(torch.float32) - recon_mean).pow(2) / var )

    return logpx_per_pixel.reshape(x.shape[0], -1).sum(dim=1)


def standard_normal_logprob(z):
    """
    Computes log p(z), assuming p ~ N(0, I)

    Returns per dimension log pdf at z
    """
    return -0.5 * (math.log(2 * math.pi) + z.pow(2))


def normal_log_prob(z, mean, logvar):
    """
    Refernence: https://arxiv.org/abs/1802.04942 
    Used to calculate E_{q(z)}[log q(z)}] used in ELBO TC Decomposition

    Calculates log(q(z|x))
    In the paper: log q(z|n), log q(z(n_i) | n_j):  Finds the pdf of q(z|n_j). Then specifying/fixing z = z(n_i), we evaluate the pdf q(z | n_j)
    at z = z(n_i)
    """
    # componentwise operations and the math.log(2 * math.pi) gets broadcasted
    # write out the Gaussian and see that this is true
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mean).pow(2) / torch.exp(logvar))

def kl_div(mean, logvar):
    """
    KL Divergence for standard beta-vae

    KL(q(z|x)||p(z)) (both assumed to be normal, closed form exists)

    q(z|x) = N(mean, diag(exp(logvar)))
    p(z) = N(0, I)
    
    """

    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1)

def beta_vae_loss(nll, mean, logvar, beta):
    kl = kl_div(mean, logvar)

    loss = (nll + beta * kl).mean()

    # detach from computation graph, do not call item here
    logs = {
        "loss": loss.mean().detach(),
        "recon_loss": nll.mean().detach(),
        "kl_total": kl.mean().detach(),
        "beta": torch.as_tensor(beta)
    }
    return loss, logs


def tc_beta_vae_loss(
    logits,
    x,
    z,
    mean,
    logvar,
    dataset_size,
    beta=6.0,
    alpha=1.0,
    gamma=1.0
):
    """
    Compute TC Beta VAE loss function

    Paper Decomposition:
    E_p(n)[ KL(q(z|n) || p(z)) ]: This is taken over the whole dataset, where n in {1, ..., N}

    In practice we take minibatches

    Note this is an UNBIASED estimate for the empriricial risk/loss which in turn is an unbiased estimate for the true expected loss (if we use p_data)

    =   Index-Code MI
        + Total Correlation
        + Dimension-wise KL

    =   KL(q(z,n) || q(z)p(n))
        + KL(q(z) || prod_j q(z_j))
        + sum_j KL(q(z_j) || p(z_j))

        Code Estimates: 
        
        log q(z|x)
        log q(z)
        sum_j log q(z_j)
        log p(z)

    Then:

        MI          = log q(z|x) - log q(z)
        TC          = log q(z) - sum_j log q(z_j)
        dimwise_KL  = sum_j log q(z_j) - log p(z)

    beta-TCVAE objective to maximize:

    objective = log p(x|z)
                    - alpha * MI
                    - beta  * TC
                    - gamma * dimwise_KL


    logits: [B, C, W, H]
    x: [B, C, W, H]
    z: [B, latent_dim]
    mean: [B, latent_dim]
    logvar: [B, latent_dim]
    dataset_size:
    """

    """
    We compute a matrix where for latent dimension k, we have mat_logqz[i][j] = q(z(n_i) | n_j)
    """

    batch_size, latent_dim = z.shape

    # -log p(x|z)
    nllpx = bernoulli_nll_from_logits(logits, x)

    # log p(z)
    logpz = standard_normal_logprob(z).sum(dim=1)
    
    # We use and ABUSE the monte carlo estimate for each one of these terms!
    # we take the mean over batch at the end

    # log q(z|x)
    # Sample z ~ log(z | x_i)
    # use monte carlo estimate (2 steps) to estiamte 
    # first step we use monte carlo estimate and take the WHOLE BATCH
    # second step use one single monte carlo estimate to esitmate E_{q(z|n)} (after using first monte carlo estimate)
    # note we keep batch dimension here we will take mean after
    logqz_condx = normal_log_prob(z, mean, logvar).sum(dim=1)


    # used for compyting E_{q(z)}[log q(z)] again we use monte carlo estimate (and see paper: https://arxiv.org/pdf/1802.04942 for derivation)
    # [B, B, latent_dim] through broadcasting
    mat_logqz = normal_log_prob(
        z.unsqueeze(1),
        mean.unsqueeze(0),
        logvar.unsqueeze(0)
    )


    # recall by defn that we assume q(z|x) sim N(mu_phi(x), diag(sigma^2phi(x))). So q(z|x) = prod_{d=1}^D q(z_d|x) since each are conditionally independent
    logqz = (torch.logsumexp(mat_logqz.sum(dim=2), dim=1)) - math.log(batch_size * dataset_size)

    logqz_prod_marginals = (torch.logsumexp(mat_logqz, dim=1) - math.log(batch_size * dataset_size)).sum(dim=1)


    mutual_info = logqz_condx - logqz

    total_correlation = logqz - logqz_prod_marginals

    # dimwise_kl, use monte carlo with B estimates and recall q(z^{i}_j), or use mini batch to represnet 
    # WHOLE DATAset, i.e. ith sample in jth latent dimension is
    # Recall q(z) = sum_{n=1}^N q(z|n) p(n), p(n) = 1/N, N is size of whole dataset. Marginalizing we have q(z_j) = int q(z) dz\setminus_j
    # = \int \frac{1}{N} sum_{n=1}^N q(z|x_n) dz\setminus_j = \frac{1}{N} \sum_{n=1}^N \int q(z|x_n) dz \setminuz_j but note that the inner integral 
    # becomes q(z_j|x_n) (marginalizaing)
    # and since we also sum over the latent dim we arrive at
    dimwise_kl = logqz_prod_marginals - logpz

    # take means
    neg_ELBO = nllpx + alpha * mutual_info + beta * total_correlation + gamma * dimwise_kl

    loss = neg_ELBO.mean()

    tc_offset = (latent_dim - 1) * math.log(dataset_size)
    
    # detach from computationi graph. Do NOT call item here. Inefficiency with CUDA sync
    logs = {
        "loss": loss.detach(),

        # -log p(x|z), reconstruction loss
        "recon_loss": nllpx.mean().detach(),

        # log p(x|z)
        # "logpx": (-nllpx).exp().mean().detach().item(),

        # Index-Code Mutual Information
        "mi": mutual_info.mean().detach(),

        # Total Correlation
        "tc_raw": total_correlation.mean().detach(),

        "tc_centered": (total_correlation - tc_offset).mean().detach(),

        # Dimension-wise KL
        "dimwise_kl": dimwise_kl.mean().detach(),

        # Original undecomposed KL estimate:
        #     KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
        "kl_total": (logqz_condx - logpz).mean().detach(),

        "beta": torch.as_tensor(beta)
    }

    return loss, logs










if __name__ == "__main__":
    print("This is main!")
    dist = ND()
    print("Number of params in ND: ", dist.nparams)

    # (batch, latent_dim, mu/logvar)
    t1 = torch.zeros((3, 10, 2))

    z = dist.sample(t1)
    print(z, z.shape)

