import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import sys
import os

from models import VAE, Discriminator
from utils import forward_pass, setup_seed

def main():
    setup_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    DATA_PATH = os.path.join("datasets", "IAD", "16QAM_Train_Test.pkl")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data["train_data"], np.ndarray):
        train_data = torch.from_numpy(data["train_data"])
    else:
        train_data = data["train_data"]
    
    # Loader
    train_loader = DataLoader(
        dataset=TensorDataset(train_data, torch.zeros(len(train_data))),
        batch_size=64,
        shuffle=True
    )
    
    # Model
    bottle = 75
    model = VAE(bottle=bottle).to(device)
    discriminator = Discriminator(bottle=bottle).to(device)
    
    optimizer_G = Adam(model.parameters(), lr=1e-5)
    optimizer_D = Adam(discriminator.parameters(), lr=1e-5) 
    
    print("Starting Debug Training (Tuned LR=1e-5)...")
    model.train()
    discriminator.train()
    
    for i, (batch_data, _) in enumerate(train_loader):
        if i > 50: break # Run 50 steps
        
        # Forward
        (recons, mean, logvar), x_input = forward_pass(model, batch_data, device)
        
        # 1. Recon
        optimizer_G.zero_grad()
        recon_loss = ((recons - x_input)**2).sum()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_G.step()
        
        # 2. Regularization
        z_fake = model.resample(mean, logvar).detach()
        z_real = torch.randn_like(z_fake).to(device)
        
        optimizer_D.zero_grad()
        d_real = discriminator(z_real)
        d_fake = discriminator(z_fake)
        
        # Label Smoothing: Real 1.0 -> 0.9, Fake 0.0 -> 0.1
        # d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
        d_loss = -torch.mean(
            0.9 * torch.log(d_real + 1e-8) +
            0.1 * torch.log(1 - d_real + 1e-8) +
            0.1 * torch.log(d_fake + 1e-8) +
            0.9 * torch.log(1 - d_fake + 1e-8)
        )
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer_D.step()
        
        # 3. Generator
        (_, mean_new, logvar_new), _ = forward_pass(model, batch_data, device)
        z_new = model.resample(mean_new, logvar_new)
        d_fake_new = discriminator(z_new)
        g_loss = -torch.mean(torch.log(d_fake_new + 1e-8))
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_G.step()
        
        print(f"Step {i}: Recon={recon_loss.item():.4f}, D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
        
        if torch.isnan(recon_loss) or torch.isnan(d_loss) or torch.isnan(g_loss):
            print("NaN detected!")
            break

if __name__ == "__main__":
    main()
