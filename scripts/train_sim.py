import argparse
import os

import torch
import torch.optim as optim
import yaml

# LeRobotDataset import
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.policy import VITAPolicy


def train_sim(config: dict) -> None:
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

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    save_freq = config["train"]["save_freq"]
    ckpt_dir = config["train"]["ckpt_dir"]
    seq_len = config["task"]["seq_len"]

    # 2. Dataloader with LeRobotDataset
    dataset_repo = config["task"].get("dataset_repo", "lerobot/aloha_sim_transfer_cube_human")
    print(f"Loading LeRobotDataset from Hugging Face: {dataset_repo}")

    # 50Hz 시뮬레이션 환경 기준으로 seq_len 만큼 미래의 액션을 가져오기 위한 설정
    fps = 50
    action_deltas = [i / fps for i in range(seq_len)]

    dataset = LeRobotDataset(
        dataset_repo,
        delta_timestamps={"action": action_deltas},
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Training Loop
    policy.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        with tqdm(dataloader, desc=f"Sim Epoch {epoch + 1}/{epochs}") as pbar:
            for _batch_idx, batch in enumerate(pbar):
                # LeRobotDataset은 딕셔너리 형태로 데이터를 반환합니다.
                # observation.images.top, observation.images.left_wrist 등의 키를 추출합니다.
                images_dict = {}
                for key, val in batch.items():
                    if key.startswith("observation.images."):
                        cam_name = key.split(".")[-1]
                        img_tensor = val.to(device)
                        # LeRobot의 이미지가 uint8일 경우 float(0~1)로 변환
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.float() / 255.0
                        images_dict[cam_name] = img_tensor

                # 액션 추출 (B, seq_len, action_dim)
                actions = batch["action"].to(device)

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

        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"policy_sim_ep{epoch + 1}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VITA Policy on Simulation Data")
    parser.add_argument("--config", type=str, default="config/sim.yml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_sim(config)
