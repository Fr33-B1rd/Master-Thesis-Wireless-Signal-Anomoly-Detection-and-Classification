import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
import json

from models import VAE, Discriminator
from utils import calc_auc, setup_seed, forward_pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tuned AAE V2.")
    parser.add_argument("--dataset", default="16QAM", help="Dataset name.")
    parser.add_argument("--run", default="tuned_v2", help="Run name.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--bottle", type=int, default=75, help="Latent dimension size.")
    
    # Tuning parameters
    parser.add_argument("--d_hidden", type=int, default=512, help="Discriminator hidden dimension.")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Weight for reconstruction loss.")
    parser.add_argument("--adv_weight", type=float, default=1.0, help="Weight for adversarial loss.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    setup_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    dataset_name = args.dataset.strip()
    run_name = args.run.strip()
    # Unique output directory based on parameters
    param_str = f"dh{args.d_hidden}_rw{args.recon_weight}_aw{args.adv_weight}"
    OUT_DIR = os.path.join(".", "runs", f"{dataset_name}_aae_tuned_{param_str}_{run_name}")
    DATA_PATH = os.path.join(".", "datasets", "IAD", f"{dataset_name}_Train_Test.pkl")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = SummaryWriter(OUT_DIR)

    # Load Data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    train_data = torch.from_numpy(data["train_data"]) if isinstance(data["train_data"], np.ndarray) else data["train_data"]
    test_data = torch.from_numpy(data["test_data"]) if isinstance(data["test_data"], np.ndarray) else data["test_data"]
    test_label = torch.from_numpy(data["test_label"]) if isinstance(data["test_label"], np.ndarray) else data["test_label"]
    
    dummy_train_label = torch.zeros(len(train_data))
    
    train_loader = DataLoader(
        dataset=TensorDataset(train_data, dummy_train_label),
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=TensorDataset(test_data, test_label),
        batch_size=64,
        shuffle=False
    )

    # Initialize Model
    print(f"Initializing AAE (dim={args.d_hidden}, rw={args.recon_weight}, aw={args.adv_weight})...")
    model = VAE(bottle=args.bottle).to(device)
    # Pass hidden_dim
    discriminator = Discriminator(bottle=args.bottle, hidden_dim=args.d_hidden).to(device)

    # Optimizers
    optimizer_G = Adam(model.parameters(), lr=args.lr)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr)

    best_mae_auc = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
            
        epoch_recon_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for idx, (batch_data, _) in enumerate(train_loader):
            # Forward pass
            (recons, mean, logvar), x_input = forward_pass(model, batch_data, device)
            
            # 1. Reconstruction Phase
            optimizer_G.zero_grad()
            # Scale loss
            recon_loss = ((recons - x_input)**2).sum() * args.recon_weight
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_G.step()
            
            # 2. Regularization Phase
            model.eval() 
            z_fake = model.resample(mean, logvar).detach()
            z_real = torch.randn_like(z_fake).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_real = discriminator(z_real)
            d_fake = discriminator(z_fake)
            
            # Label Smoothing (0.9, 0.1)
            # Scale loss
            d_loss = -torch.mean(
                0.9 * torch.log(d_real + 1e-8) +
                0.1 * torch.log(1 - d_real + 1e-8) +
                0.1 * torch.log(d_fake + 1e-8) +
                0.9 * torch.log(1 - d_fake + 1e-8)
            ) * args.adv_weight
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()
            
            # Train Generator (Encoder)
            model.train()
            optimizer_G.zero_grad()
            
            (_, mean_new, logvar_new), _ = forward_pass(model, batch_data, device)
            z_new = model.resample(mean_new, logvar_new)
            
            d_fake_new = discriminator(z_new)
            # Generator wants D(fake) -> 1.0 (or 0.9)
            g_loss = -torch.mean(torch.log(d_fake_new + 1e-8)) * args.adv_weight
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_G.step()
            
            epoch_recon_loss += recon_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            step = idx + len(train_loader) * epoch
            writer.add_scalars("loss", {
                "recon": recon_loss.item(),
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item()
            }, step)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Recon={epoch_recon_loss/len(train_loader):.1f}, D={epoch_d_loss/len(train_loader):.3f}, G={epoch_g_loss/len(train_loader):.3f}")

        # Validation
        if epoch % 5 == 4:
            model.eval()
            with torch.no_grad():
                inputs_list, labels_list, recons_list = [], [], []
                
                for val_data, val_label in test_loader:
                    (val_recon, _, _), val_input = forward_pass(model, val_data, device)
                    inputs_list.append(val_input)
                    labels_list.append(val_label)
                    recons_list.append(val_recon)
                
                inputs = torch.cat(inputs_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                recons = torch.cat(recons_list, dim=0)

                # Save Images
                if epoch % 50 == 49:
                    save_image(inputs[labels==0][-64:], f"{OUT_DIR}/epoch_{epoch}_input_normal.png")
                    save_image(recons[labels==0][-64:], f"{OUT_DIR}/epoch_{epoch}_recon_normal.png")
                
                mae_auc, mse_auc, _, per_auc = calc_auc(inputs, labels, recons)
                
                print(f"[Val] Epoch {epoch} AUCs: MAE={mae_auc:.4f}, MSE={mse_auc:.4f}, PER={per_auc:.4f}")
                
                stats = {
                    "epoch": epoch,
                    "mae_auc": float(mae_auc),
                    "mse_auc": float(mse_auc),
                    "per_auc": float(per_auc)
                }
                
                writer.add_scalars("auc", {
                    "mae": mae_auc,
                    "mse": mse_auc,
                    "per": per_auc
                }, step)
                
                if mae_auc > best_mae_auc:
                    best_mae_auc = mae_auc
                    torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
                    # Save stats
                    with open(f"{OUT_DIR}/best_stats.json", "w") as f:
                        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
