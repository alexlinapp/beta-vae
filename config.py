from dataclasses import dataclass

@dataclass
class Config:
    latent_dim: int = 10
    alpha: float = 1.0
    beta: float = 4.0
    gamma: float = 1.0
    lr: float = 1e-2
    epochs: int = 130
    device: str = "cuda"
    objective: str = "B"    # differnet obejctives. B = burgess, H = Higgins, T = TC Beta VAE
    beta_warmup: int = 10000
    train_split: float = 0.8
    batch_size: int = 128
    run_name: str = "default_run"
    run_dir: str = "runs"
    checkpoint_freq: int = 5
    datasets_dir: str = "datasets"
    seed: int = 0
    likelihood: str = "bernoulli"
    optimizer: str = "Adagrad"
    beta_annealing: str = None # None or Linear
    run_timestamp: bool = True # add a timestamp to run directory
