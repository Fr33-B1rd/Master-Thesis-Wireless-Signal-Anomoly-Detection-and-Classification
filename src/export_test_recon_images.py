import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from models import VAE, ConvAE
from utils import forward_pass


@dataclass
class ExportConfig:
    model_name: str
    ckpt_path: str
    input_scale: float = 1.0
    no_output_sigmoid: bool = False
    conv_ae_use_bn: bool = False
    conv_ae_use_skip: bool = False
    bottle: int = 75


def build_model(cfg: ExportConfig, device: torch.device):
    use_sigmoid = not cfg.no_output_sigmoid
    if cfg.model_name in ("conv_vae", "beta_vae", "aae"):
        model = VAE(bottle=cfg.bottle, use_sigmoid=use_sigmoid).to(device)
    elif cfg.model_name == "conv_ae":
        model = ConvAE(
            bottle=cfg.bottle,
            use_sigmoid=use_sigmoid,
            use_bn=cfg.conv_ae_use_bn,
            use_skip=cfg.conv_ae_use_skip,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {cfg.model_name}")
    state = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_reconstruction(model_name: str, model, batch, device, input_scale: float):
    forward_out, x_input = forward_pass(model, batch, device, input_scale=input_scale)
    if model_name == "conv_ae":
        recon, _ = forward_out
    else:
        recon, _, _ = forward_out
    return x_input, recon


def export_one(cfg: ExportConfig, test_loader: DataLoader, out_dir: str, device: torch.device):
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(cfg, device)

    inputs, labels, recons = [], [], []
    with torch.no_grad():
        for batch, label in test_loader:
            x_input, recon = get_reconstruction(cfg.model_name, model, batch, device, cfg.input_scale)
            inputs.append(x_input.cpu())
            labels.append(label.cpu())
            recons.append(recon.cpu())

    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    recons = torch.cat(recons, dim=0)

    save_image(inputs[labels == 0][-64:], os.path.join(out_dir, "input-test-n.png"))
    save_image(recons[labels == 0][-64:], os.path.join(out_dir, "output-test-n.png"))
    save_image(inputs[labels == 1][-64:], os.path.join(out_dir, "input-test-a.png"))
    save_image(recons[labels == 1][-64:], os.path.join(out_dir, "output-test-a.png"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "16QAM"
    data_path = os.path.join(".", "datasets", "IAD", f"{dataset}_Train_Test.pkl")
    out_root = os.path.join(".", "artifacts", "spectrograms", dataset)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    test_data = torch.from_numpy(data["test_data"]) if isinstance(data["test_data"], np.ndarray) else data["test_data"]
    test_label = torch.from_numpy(data["test_label"]) if isinstance(data["test_label"], np.ndarray) else data["test_label"]

    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=64, shuffle=False)

    configs = [
        ExportConfig(
            model_name="conv_vae",
            ckpt_path=os.path.join(".", "runs", "16QAM_conv_vae_review_fix_200ep_gpu_iis", "best_model.pth"),
            input_scale=1.0,
            no_output_sigmoid=False,
        ),
        ExportConfig(
            model_name="aae",
            ckpt_path=os.path.join(".", "runs", "16QAM_aae_review_fix_200ep_gpu_iis", "best_model.pth"),
            input_scale=1.0,
            no_output_sigmoid=False,
        ),
        ExportConfig(
            model_name="beta_vae",
            ckpt_path=os.path.join(".", "runs", "16QAM_beta_vae_beta_bestcombo_bestbeta_200ep", "best_model.pth"),
            input_scale=10.0,
            no_output_sigmoid=True,
        ),
        ExportConfig(
            model_name="conv_ae",
            ckpt_path=os.path.join(".", "runs", "16QAM_conv_ae_convae_stageB_60ep_bn", "best_model.pth"),
            input_scale=10.0,
            no_output_sigmoid=True,
            conv_ae_use_bn=True,
            conv_ae_use_skip=False,
        ),
    ]

    for cfg in configs:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
        export_one(
            cfg=cfg,
            test_loader=test_loader,
            out_dir=os.path.join(out_root, cfg.model_name),
            device=device,
        )

    print(f"Saved spectrogram images to: {out_root}")


if __name__ == "__main__":
    main()
