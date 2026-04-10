import argparse
import os

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.policy import VITAPolicy

# TODO: Import your actual Real-world Dataset here
# from utils.real_env import BimalRealDataset


def train_real(config: dict) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Initialize Policy
    policy = VITAPolicy(
        action_dim=config["task"]["action_dim"],
        seq_len=config["task"]["seq_len"],
        num_cameras=config["task"]["num_cameras"],
        latent_dim=config["model"]["latent_dim"],
        flow_hidden_dim=config["model"]["flow_hidden_dim"],
        flow_num_layers=config["model"]["flow_num_layers"],
        num_sampling_steps=config["model"]["num_sampling_steps"],
    ).to(device)

    # If fine-tuning from a simulation checkpoint, load it here
    pretrained_ckpt = config["train"].get("pretrained_ckpt", "")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"Loading pretrained simulation checkpoint: {pretrained_ckpt}")
        policy.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    elif pretrained_ckpt:
        print(f"WARNING: Pretrained checkpoint not found at {pretrained_ckpt}. Training from scratch.")

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    save_freq = config["train"]["save_freq"]
    ckpt_dir = config["train"]["ckpt_dir"]

    # 2. Dataloader (Placeholder)
    # dataset = BimalRealDataset(data_dir=config['task']['data_dir'])
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Dummy variables for compilation testing
    print("WARNING: Using dummy dataloader. Replace with BimalRealDataset.")
    dataloader = [
        (
            {
                "cam_top": torch.randn(batch_size, 3, 224, 224),
                "cam_left": torch.randn(batch_size, 3, 224, 224),
                "cam_right": torch.randn(batch_size, 3, 224, 224),
            },
            torch.randn(batch_size, config["task"]["seq_len"], config["task"]["action_dim"]),
        )
        for _ in range(5)
    ]

    # 3. Training Loop
    policy.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        with tqdm(dataloader, desc=f"Real Epoch {epoch+1}/{epochs}") as pbar:
            for _batch_idx, (images_dict, actions) in enumerate(pbar):
                # Move to device
                images_dict = {k: v.to(device) for k, v in images_dict.items()}
                actions = actions.to(device)

                # Forward & Loss
                optimizer.zero_grad()
                loss, metrics = policy.compute_loss(images_dict, actions)

                # Backward
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['total_loss']:.4f}",
                        "flow": f"{metrics['flow_loss']:.4f}",
                        "vae": f"{metrics['vae_recon_loss']:.4f}",
                    }
                )

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"policy_real_ep{epoch+1}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VITA Policy on Real-world Data")
    parser.add_argument("--config", type=str, default="config/real.yml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_real(config)
