import os
import torch
from torch.utils.data import DataLoader


# models
import model
from model import MLPDecoder, MLPEncoder, VAE

# loss functions
from model import bernoulli_nll_from_logits, gaussian_nll_from_mean, beta_vae_loss, tc_beta_vae_loss

# data
from data import get_dataloaders


# for defaultdict
from collections import defaultdict

# extra util stuff
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
from dataclasses import asdict
import time


class Trainer:
    def __init__(self, model, optimizer, cfg, dataset_size):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = cfg.device
        self.dataset_size = dataset_size
        self.global_step = 0
        self.batch_size = cfg.batch_size



    def train_one_epoch(self, dataloader):
        self.model.train()

        totals = {}

        for i, train_x in enumerate(dataloader):
            # print("This is batch: i: ", i)
            self.optimizer.zero_grad()
            
            train_x = train_x.to(self.device)

            #TODO add annealing configs later
            # beta = self.linear_anneal(
            #     self.global_step,
            #     self.cfg.beta_max,
            #     self.cfg.warmup_steps,
            # )
            # find the annealing
            beta = self.get_beta()
            alpha = self.get_alpha()
            gamma = self.get_gamma()

            logits, mean, logvar, z = self.model(train_x)

            # code below used for debugging
            
            # print("mean:", mean.min().item(), mean.max().item())
            # print("logvar:", logvar.min().item(), logvar.max().item())
            # print("z:", z.min().item(), z.max().item())
            # print("logits:", logits.min().item(), logits.max().item())

            # #TODO remove this!
            # if i >= 50:
            #     break

            nll = self.reconstruction_nll(logits, train_x)

            if self.cfg.objective == "B":
                loss, logs = beta_vae_loss(
                    nll=nll,
                    mean=mean,
                    logvar=logvar,
                    beta=beta,
                )
            elif self.cfg.objective == "T":
                loss, logs = tc_beta_vae_loss(
                    logits=logits,
                    x=train_x,
                    z=z,
                    mean=mean,
                    logvar=logvar,
                    dataset_size=self.dataset_size,
                    beta=beta,
                    alpha=alpha,
                    gamma=gamma
                )
            else:
                raise ValueError(f"Unknown objective type: {self.cfg.objective}")

            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            for k, v in logs.items():
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v)
                totals[k] = totals.get(k, torch.zeros((), device=self.device)) + v
                                       
        # move back to CPU here so don't need to constantly resync with CUDA
        return {k: (v / len(dataloader)).item() for k, v in totals.items()}


    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        totals = {}


        for i, val_x in enumerate(dataloader):

            val_x = val_x.to(self.device)

            beta = self.get_beta()
            alpha = self.get_alpha()
            gamma = self.get_gamma()

            logits, mean, logvar, z = self.model(val_x)


            nll = self.reconstruction_nll(logits, val_x)

            if self.cfg.objective == "B":
                loss, logs = beta_vae_loss(
                    nll=nll,
                    mean=mean,
                    logvar=logvar,
                    beta=beta,
                )
            elif self.cfg.objective == "T":
                loss, logs = tc_beta_vae_loss(
                    logits=logits,
                    x=val_x,
                    z=z,
                    mean=mean,
                    logvar=logvar,
                    dataset_size=self.dataset_size,
                    beta=beta,
                    alpha=alpha,
                    gamma=gamma
                )
            else:
                raise ValueError(f"Unknown objective type: {self.cfg.objective}")


            for k, v in logs.items():
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v)
                totals[k] = totals.get(k, torch.zeros((), device=self.device)) + v

        return {k: (v / len(dataloader)).item() for k, v in totals.items()}

    @staticmethod
    def linear_anneal(step, max_beta, warmup_steps):
        return min(max_beta, max_beta * step / warmup_steps)

    @staticmethod
    def no_anneal(max_beta):
        return max_beta

    def get_beta(self):
        anneal_type = self.cfg.beta_annealing

        if anneal_type is None:
            return self.cfg.beta

        if anneal_type == "linear":
            return self.linear_anneal(self.global_step, self.cfg.beta, self.cfg.beta_warmup)

        raise ValueError(f"Undefined beta annealing type. Got: {anneal_type}")

    #TODO, can add annealing here as well
    def get_alpha(self):
        return self.cfg.alpha
    def get_gamma(self):
        return self.cfg.gamma

    def reconstruction_nll(self, recon_x, x):
        if self.cfg.likelihood == "bernoulli":
            return bernoulli_nll_from_logits(recon_x, x)
        elif self.cfg.likelihood == "gaussian":
            return gaussian_nll_from_mean(recon_x, x)

        raise ValueError(f"Unknown Likelihood: {self.cfg.likelihood}")





class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        # device selection
        if cfg.device != "cuda" and cfg.device != "cpu":
            raise ValueError(f"Device must be either cuda or cpu. Got {cfg.device}")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"cgf.device='cuda' was requested, but CUDA is not available")
        self.device = cfg.device

        self.train_loader, self.val_loader = get_dataloaders(cfg)

        # model selection
        self.encoder = MLPEncoder(cfg.latent_dim)
        self.decoder = MLPDecoder(cfg.latent_dim)
        self.model = VAE(cfg.latent_dim, self.encoder, self.decoder).to(self.device)


        # optimizer selection
        #TODO Changed to adaaGrad from AdamW
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=cfg.lr)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.optimizer = self._build_optimizer()

        self.trainer = Trainer(model=self.model,
                               optimizer=self.optimizer,
                               cfg=self.cfg,
                               dataset_size=len(self.train_loader.dataset))
        
        # logging
        self.history = defaultdict(list)

        self.timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d|%H:%M:%S")
        self.run_name = self._build_run_name()
        self.run_dir = Path("runs") / f"{self.run_name}_{self.timestamp}" if cfg.run_timestamp else Path("runs") / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    
    def run(self):
        print("Starting the expeirment run!", flush=True)
        for epoch in range(self.cfg.epochs):
            start_time = time.perf_counter()

            train_logs = self.trainer.train_one_epoch(self.train_loader)
            val_logs = self.trainer.evaluate(self.val_loader)

            end_time = time.perf_counter()

            print(f'Epoch {epoch + 1}| Time Elapsed: {end_time - start_time}')
            print(f'\tTrain: ', end="")
            for key, value in train_logs.items():
                if type(value) is float:
                    print(f'{key}: {value:.5f}, ', end="")
                else:
                    print(f'{key}: {value}, ', end="")

            print(f'\tValid: ', end="")
            for key, value in val_logs.items():
                if type(value) is float:
                    print(f'{key}: {value:.5f}, ', end="")
                else:
                    print(f'{key}: {value}, ', end="")
            
            print(flush=True)
            
            self._update_history(epoch, train_logs=train_logs, val_logs=val_logs)

            if (epoch + 1) % self.cfg.checkpoint_freq == 0:
                self._save_checkpoint(epoch+1)
        self._save_history()
        self._save_model_weights()


    def _update_history(self, epoch, train_logs, val_logs):
        self.history["epoch"].append(epoch+1)

        for key, value in train_logs.items():
            self.history[f"train_{key}"].append(value)  # defaultdict allows us to do this
        
        for key, value in val_logs.items():
            self.history[f"val_{key}"].append(value)
        
    def _save_history(self):
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def _save_config(self):
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    def _save_model_weights(self):
        torch.save(self.model.state_dict(), self.run_dir / "model_weights.pt")
    
    def _save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "global_step": self.trainer.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "history": self.history
        },
        self.run_dir / f"epoch{epoch}_checkpoint.pt")


    # example of how to load the json back
    def _load_history(self, path):
        with open(path, "r") as f:
            history = json.load(f)
        return history

    # example of loading a checkpoint
    def _load_checkpoint(self, path):
        return torch.load(path, map_location=self.device)

    def _build_optimizer(self):
        name = self.cfg.optimizer.lower()

        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
            )

        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
            )

        if name == "adagrad":
            return torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.cfg.lr,
            )

        valid = ["adam", "adamw", "adagrad"]
        raise ValueError(f"Unknown optimizer '{self.cfg.optimizer}'. Valid options: {valid}")

    def _build_run_name(self):
        if self.cfg.run_name is not None:
            return self.cfg.run_name

        objective = self.cfg.objective

        if objective in ["T"]:
            return (
                f"TCBeta"
                f"_alpha-{self.cfg.alpha}"
                f"_beta-{self.cfg.beta}"
                f"_gamma-{self.cfg.gamma}"
                f"_betaAnneal-{self.cfg.beta_annealing}"
                f"_epochs-{self.cfg.epochs}"
            )

        if objective in ["B"]:
            return (
                f"BetaVAE"
                f"_beta-{self.cfg.beta}"
                f"_betaAnneal-{self.cfg.beta_annealing}"
                f"_epochs-{self.cfg.epochs}"
            )

        return (
            f"{self.cfg.objective}"
            f"_betaAnneal-{self.cfg.beta_annealing}"
            f"_epochs-{self.cfg.epochs}"
        )
        


if __name__ == "__main__":
    print("experiment.py is main!")
