from config import Config
import argparse
import torch
import numpy as np
from experiment import Experiment


import argparse
from dataclasses import dataclass
from experiment import Experiment


def str_to_bool(x: str) -> bool:
    if x.lower() in {"true", "1", "yes", "y"}:
        return True
    if x.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_dim", type=int, default=Config.latent_dim)
    parser.add_argument("--alpha", type=float, default=Config.alpha)
    parser.add_argument("--beta", type=float, default=Config.beta)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--objective", type=str, choices=["B", "H", "T"], default=Config.objective)
    parser.add_argument("--beta_warmup", type=int, default=Config.beta_warmup)
    parser.add_argument("--train_split", type=float, default=Config.train_split)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--run_name", type=str, default=Config.run_name)
    parser.add_argument("--run_dir", type=str, default=Config.run_dir)
    parser.add_argument("--checkpoint_freq", type=int, default=Config.checkpoint_freq)
    parser.add_argument("--datasets_dir", type=str, default=Config.datasets_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--likelihood", type=str, choices=["bernoulli", "gaussian"], default=Config.likelihood)
    parser.add_argument("--optimizer", type=str, choices=["Adagrad", "Adam", "SGD"], default=Config.optimizer)
    parser.add_argument("--beta_annealing", type=str, choices=["None", "Linear"], default=Config.beta_annealing)
    parser.add_argument("--run_timestamp", type=str_to_bool, default=Config.run_timestamp)

    args = parser.parse_args()
    cfg = Config(**vars(args))

    if cfg.beta_annealing == "None":
        cfg.beta_annealing = None

    return cfg


def main():
    cfg = parse_args()

    experiment = Experiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()