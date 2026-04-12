import argparse
import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
import wandb
import yaml

# LeRobotDataset import
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.policy import VITAPolicy


def extract_batch(batch: dict, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """배치에서 이미지와 액션을 추출"""
    images_dict = {}
    for key, val in batch.items():
        if key.startswith("observation.images."):
            cam_name = key.split(".")[-1]
            img_tensor = val.to(device, dtype=torch.float32)
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            images_dict[cam_name] = img_tensor
    actions = batch["action"].to(device)
    return images_dict, actions


@torch.no_grad()
def validate(policy: VITAPolicy, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Validation loop"""
    policy.eval()
    total_metrics: dict[str, float] = {}
    num_batches = 0

    for batch in val_loader:
        images_dict, actions = extract_batch(batch, device)
        _, metrics = policy.compute_loss(images_dict, actions)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

    # Average
    avg_metrics = {f"val/{k}": v / num_batches for k, v in total_metrics.items()}
    policy.train()
    return avg_metrics


def train_sim(config: dict) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 0. Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("wandb", {}).get("run_name", "run")
    run_name = f"{run_name}_{timestamp}"
    wandb.init(
        project=config.get("wandb", {}).get("project", "BIMAL"),
        name=run_name,
        config=config,
    )

    # 1. Initialize Policy
    model_cfg = config["model"]
    policy = VITAPolicy(
        action_dim=config["task"]["action_dim"],
        seq_len=config["task"]["seq_len"],
        num_cameras=config["task"]["num_cameras"],
        latent_dim=model_cfg["latent_dim"],
        flow_hidden_dim=model_cfg["flow_hidden_dim"],
        flow_num_layers=model_cfg["flow_num_layers"],
        num_sampling_steps=model_cfg["num_sampling_steps"],
        # VITA core parameters
        decode_flow_latents=model_cfg.get("decode_flow_latents", True),
        consistency_weight=model_cfg.get("consistency_weight", 1.0),
        enc_contrastive_weight=model_cfg.get("enc_contrastive_weight", 0.0),
        flow_contrastive_weight=model_cfg.get("flow_contrastive_weight", 0.0),
        sigma=model_cfg.get("sigma", 0.0),
    ).to(device)

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    save_freq = config["train"]["save_freq"]
    seq_len = config["task"]["seq_len"]

    # wandb run name으로 체크포인트 폴더 생성
    base_ckpt_dir = config["train"]["ckpt_dir"]
    ckpt_dir = os.path.join(base_ckpt_dir, wandb.run.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # 2. Dataloader with LeRobotDataset (Train / Val split)
    dataset_repo = config["task"].get("dataset_repo", "lerobot/aloha_sim_transfer_cube_human")
    print(f"Loading LeRobotDataset from Hugging Face: {dataset_repo}")

    fps = 50
    action_deltas = [i / fps for i in range(seq_len)]

    # 에피소드 단위 split (Subset 사용)
    dataset = LeRobotDataset(dataset_repo, delta_timestamps={"action": action_deltas})
    num_episodes = dataset.num_episodes
    val_ratio = config["train"].get("val_ratio", 0.1)
    num_val = max(1, int(num_episodes * val_ratio))
    num_train = num_episodes - num_val

    train_indices = []
    val_indices = []
    for i in range(len(dataset)):
        ep = dataset.hf_dataset[i]["episode_index"]
        if ep < num_train:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train: {len(train_dataset)} frames ({num_train} episodes)")
    print(f"Val:   {len(val_dataset)} frames ({num_val} episodes)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    policy.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Sim Epoch {epoch + 1}/{epochs}") as pbar:
            for _batch_idx, batch in enumerate(pbar):
                images_dict, actions = extract_batch(batch, device)

                # Forward & Loss
                optimizer.zero_grad()
                loss, metrics = policy.compute_loss(images_dict, actions)

                # Backward
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step = epoch * len(train_loader) + _batch_idx
                wandb.log(metrics, step=global_step)
                postfix = {
                    "loss": f"{metrics['total_loss']:.4f}",
                    "flow": f"{metrics['flow_loss']:.4f}",
                    "vae": f"{metrics['vae_recon_loss']:.4f}",
                }
                if "consistency_loss" in metrics:
                    postfix["consist"] = f"{metrics['consistency_loss']:.4f}"
                if "enc_contrastive_loss" in metrics:
                    postfix["enc_cl"] = f"{metrics['enc_contrastive_loss']:.4f}"
                if "flow_contrastive_loss" in metrics:
                    postfix["flow_cl"] = f"{metrics['flow_contrastive_loss']:.4f}"
                pbar.set_postfix(postfix)

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_metrics = validate(policy, val_loader, device)
        val_loss = val_metrics["val/total_loss"]
        wandb.log({"train/epoch_avg_loss": avg_train_loss, "epoch": epoch + 1, **val_metrics})

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Flow: {val_metrics['val/flow_loss']:.4f} | "
            f"Val VAE: {val_metrics['val/vae_recon_loss']:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, "policy_sim_best.pth")
            torch.save(policy.state_dict(), best_path)
            wandb.save(best_path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_sim_ep{epoch + 1}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VITA Policy on Simulation Data")
    parser.add_argument("--config", type=str, default="config/sim.yml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_sim(config)
