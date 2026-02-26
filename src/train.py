import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import json
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np

from models import VAE, VAE_FC, ConvAE, Discriminator
from utils import calc_auc, setup_seed, forward_pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/AAE for anomaly detection.")
    parser.add_argument("--dataset", default="16QAM", help="Dataset name.")
    parser.add_argument("--run", default="run1", help="Run name.")
    parser.add_argument("--model", choices=["conv_vae", "fc_vae", "aae", "beta_vae", "conv_ae"], default="conv_vae", help="Model type.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for Adam optimizer.")
    parser.add_argument("--bottle", type=int, default=75, help="Latent dimension size.")
    parser.add_argument("--lmd", type=float, default=0.1, help="KL weight for VAE/Beta-VAE.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for Beta-VAE (overrides lmd if model is beta_vae).")
    parser.add_argument("--max_batches", type=int, default=0, help="Maximum number of train batches per epoch (0 means no limit).")
    parser.add_argument("--input_scale", type=float, default=1.0, help="Input scaling factor applied before model forward.")
    parser.add_argument("--no_output_sigmoid", action="store_true", help="Disable output sigmoid in convolutional decoder models.")
    parser.add_argument("--conv_ae_l1_weight", type=float, default=1.0, help="L1 reconstruction loss weight for Conv-AE.")
    parser.add_argument("--conv_ae_mse_weight", type=float, default=0.5, help="MSE reconstruction loss weight for Conv-AE.")
    parser.add_argument("--conv_ae_use_bn", action="store_true", help="Enable BatchNorm in Conv-AE encoder/decoder blocks.")
    parser.add_argument("--conv_ae_use_skip", action="store_true", help="Enable residual skip from input to output in Conv-AE.")
    return parser.parse_args()

def vae_loss_fn(y, mean, logvar, x, beta=1.0):
    mse = ((y-x)**2).sum()
    kld = 0.5*(1+logvar-mean**2-logvar.exp()).sum()
    return mse - beta*kld

def main():
    args = parse_args()
    
    # Adjust beta/lmd based on model type
    if args.model == "beta_vae":
        kl_weight = args.beta
    else:
        kl_weight = args.lmd

    setup_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    dataset_name = args.dataset.strip()
    run_name = args.run.strip()
    OUT_DIR = os.path.join(".", "runs", f"{dataset_name}_{args.model}_{run_name}")
    DATA_PATH = os.path.join(".", "datasets", "IAD", f"{dataset_name}_Train_Test.pkl")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = SummaryWriter(OUT_DIR)

    # Load Data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    train_data = torch.from_numpy(data["train_data"]) if isinstance(data["train_data"], np.ndarray) else data["train_data"]
    # train_label = torch.from_numpy(data["train_label"]) if isinstance(data["train_label"], np.ndarray) else data["train_label"]
    test_data = torch.from_numpy(data["test_data"]) if isinstance(data["test_data"], np.ndarray) else data["test_data"]
    test_label = torch.from_numpy(data["test_label"]) if isinstance(data["test_label"], np.ndarray) else data["test_label"]
    
    # For Unsupervised training, we only use train_data (and ignore train_label which are all clean usually)
    # But creating TensorDataset for consistency
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
    print(f"Initializing {args.model}...")
    use_sigmoid = not args.no_output_sigmoid
    if args.model == "fc_vae":
        model = VAE_FC(bottle=args.bottle).to(device)
    elif args.model == "conv_ae":
        model = ConvAE(
            bottle=args.bottle,
            use_sigmoid=use_sigmoid,
            use_bn=args.conv_ae_use_bn,
            use_skip=args.conv_ae_use_skip
        ).to(device)
    else:
        # conv_vae, beta_vae, aae all use the standard Convolutional VAE structure (encoder/decoder)
        model = VAE(bottle=args.bottle, use_sigmoid=use_sigmoid).to(device)

    # Optimizers
    if args.model == "aae":
        discriminator = Discriminator(bottle=args.bottle).to(device)
        # Encoder/Decoder optimizer - Use Tuned LR 1e-5
        lr_aae = 1e-5
        optimizer_G = Adam(model.parameters(), lr=lr_aae, weight_decay=args.weight_decay)
        # Discriminator optimizer
        optimizer_D = Adam(discriminator.parameters(), lr=lr_aae, weight_decay=args.weight_decay)
        # Encoder Generator optimizer (for adversarial part)
        # Typically one can use optimizer_G for both reconstruction and generator loss, 
        # or separate them. Here we use optimizer_G for both.
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        discriminator = None

    best_per_auc = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        if discriminator:
            discriminator.train()
            
        epoch_loss = 0
        
        for idx, (batch_data, _) in enumerate(train_loader):
            # Forward pass
            # forward_pass scales input by 10
            forward_out, x_input = forward_pass(model, batch_data, device, input_scale=args.input_scale)
            if args.model == "conv_ae":
                recons, z = forward_out
                mean = logvar = None
            else:
                recons, mean, logvar = forward_out
            
            if args.model == "aae":
                # AAE Training
                
                # 1. Reconstruction Phase
                optimizer_G.zero_grad()
                recon_loss = ((recons - x_input)**2).sum() # Reconstruction loss
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_G.step()
                
                # 2. Regularization Phase
                model.eval() # Freeze encoder BN if any (not here but good practice)
                # Generate latent code again (or detach from previous)
                z_fake = model.resample(mean, logvar).detach() # Latent code from encoder
                z_real = torch.randn_like(z_fake).to(device) # Sample from prior N(0, I)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                d_real = discriminator(z_real)
                d_fake = discriminator(z_fake)
                
                # Discriminator Loss: max log(D(z)) + log(1 - D(E(x)))
                # Label Smoothing: Real 1.0 -> 0.9, Fake 0.0 -> 0.1
                # BCE Loss
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
                
                # Train Generator (Encoder) to fool Discriminator
                model.train()
                optimizer_G.zero_grad()
                
                # We need fresh z from encoder with grads
                # Note: ref_code VAE resample uses standard reparameterization trick
                # For AAE, we usually just use the deterministic mean or the sample. 
                # Strict AAE often checks distribution of z (sample) matching Prior.
                (_, mean_new, logvar_new), _ = forward_pass(model, batch_data, device, input_scale=args.input_scale)
                z_new = model.resample(mean_new, logvar_new)
                
                d_fake_new = discriminator(z_new)
                # Generator Loss: max log(D(E(x)))
                g_loss = -torch.mean(torch.log(d_fake_new + 1e-8))
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_G.step()
                
                loss_val = recon_loss.item() + d_loss.item() + g_loss.item()
                
            elif args.model in ["conv_vae", "fc_vae", "beta_vae"]:
                # VAE / Beta-VAE Training
                loss = vae_loss_fn(recons, mean, logvar, x_input, beta=kl_weight)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            else:
                # Conv-AE Training (mixed reconstruction loss for more stable optimization)
                l1 = torch.abs(recons - x_input).mean()
                mse = ((recons - x_input) ** 2).mean()
                loss = args.conv_ae_l1_weight * l1 + args.conv_ae_mse_weight * mse
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = loss.item()

            epoch_loss += loss_val
            
            step = idx + len(train_loader) * epoch
            writer.add_scalars("loss", {"train": loss_val}, step)
            if args.max_batches > 0 and (idx + 1) >= args.max_batches:
                break

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {epoch_loss / len(train_loader)}")

        # Validation
        if epoch % 5 == 4:
            model.eval()
            with torch.no_grad():
                inputs_list, labels_list, recons_list = [], [], []
                
                for val_data, val_label in test_loader:
                    val_forward_out, val_input = forward_pass(model, val_data, device, input_scale=args.input_scale)
                    if args.model == "conv_ae":
                        val_recon, _ = val_forward_out
                    else:
                        val_recon, _, _ = val_forward_out
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
                    save_image(inputs[labels==1][-64:], f"{OUT_DIR}/epoch_{epoch}_input_anom.png")
                    save_image(recons[labels==1][-64:], f"{OUT_DIR}/epoch_{epoch}_recon_anom.png")

                # Calculate AUCs
                mae_auc, mse_auc, rel_auc, per_auc = calc_auc(inputs, labels, recons)
                
                print(f"[Val] Epoch {epoch} AUCs: MAE={mae_auc:.4f}, MSE={mse_auc:.4f}, PER={per_auc:.4f}, Rel={rel_auc:.4f}")
                
                stats = {
                    "epoch": epoch,
                    "mae_auc": float(mae_auc),
                    "mse_auc": float(mse_auc),
                    "per_auc": float(per_auc),
                    "rel_auc": float(rel_auc)
                }
                
                writer.add_scalars("auc", {
                    "mae": mae_auc,
                    "mse": mse_auc,
                    "per": per_auc,
                    "rel": rel_auc
                }, step)
                
                if per_auc > best_per_auc:
                    best_per_auc = per_auc
                    torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
                    with open(f"{OUT_DIR}/best_stats.json", "w") as f:
                        json.dump(stats, f, indent=4)
                    print(f"New best model saved with PER AUC: {best_per_auc:.4f}")

if __name__ == "__main__":
    main()
