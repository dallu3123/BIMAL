import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader

from model.policy import VITAPolicy


def play_sim(config: dict, ckpt_path: str, num_episodes: int = 5, save_dir: str = "./outputs/play") -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Load Policy
    model_cfg = config["model"]
    policy = VITAPolicy(
        action_dim=config["task"]["action_dim"],
        seq_len=config["task"]["seq_len"],
        num_cameras=config["task"]["num_cameras"],
        latent_dim=model_cfg["latent_dim"],
        flow_hidden_dim=model_cfg["flow_hidden_dim"],
        flow_num_layers=model_cfg["flow_num_layers"],
        num_sampling_steps=model_cfg["num_sampling_steps"],
        decode_flow_latents=model_cfg.get("decode_flow_latents", True),
        consistency_weight=model_cfg.get("consistency_weight", 1.0),
        enc_contrastive_weight=model_cfg.get("enc_contrastive_weight", 0.0),
        flow_contrastive_weight=model_cfg.get("flow_contrastive_weight", 0.0),
        sigma=model_cfg.get("sigma", 0.0),
    ).to(device)

    policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    policy.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # 2. Load Dataset
    seq_len = config["task"]["seq_len"]
    dataset = LeRobotDataset(
        config["task"].get("dataset_repo", "lerobot/aloha_sim_transfer_cube_human"),
        delta_timestamps={"action": [i / 50 for i in range(seq_len)]},
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    os.makedirs(save_dir, exist_ok=True)

    # 3. Evaluate & Visualize
    all_mse = []
    for ep_idx, batch in enumerate(dataloader):
        if ep_idx >= num_episodes:
            break

        # Extract observation images
        images_dict = {}
        display_img = None
        for key, val in batch.items():
            if key.startswith("observation.images."):
                cam_name = key.split(".")[-1]
                img_tensor = val.to(device, dtype=torch.float32)
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                images_dict[cam_name] = img_tensor
                # 시각화용 이미지 저장 (첫 카메라)
                if display_img is None:
                    display_img = img_tensor[0].cpu().permute(1, 2, 0).numpy()

        gt_actions = batch["action"].to(device)

        # Predict actions
        with torch.no_grad():
            pred_actions = policy.get_action(images_dict)

        # Compute MSE
        mse = torch.nn.functional.mse_loss(pred_actions, gt_actions).item()
        all_mse.append(mse)
        print(f"Episode {ep_idx + 1}: MSE = {mse:.6f}")

        # Visualize: observation image + action comparison
        gt_np = gt_actions[0].cpu().numpy()  # (seq_len, 14)
        pred_np = pred_actions[0].cpu().numpy()  # (seq_len, 14)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Top-left: observation image
        axes[0, 0].imshow(display_img)
        axes[0, 0].set_title("Observation (top camera)")
        axes[0, 0].axis("off")

        # Top-right: left arm joints (0~5)
        t = np.arange(seq_len)
        for j in range(6):
            axes[0, 1].plot(t, gt_np[:, j], "--", alpha=0.6, label=f"GT joint {j}")
            axes[0, 1].plot(t, pred_np[:, j], "-", alpha=0.8, label=f"Pred joint {j}")
        axes[0, 1].set_title("Left Arm Joints (0-5)")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Joint Position")
        axes[0, 1].legend(fontsize=6, ncol=2)

        # Bottom-left: right arm joints (7~12)
        for j in range(7, 13):
            axes[1, 0].plot(t, gt_np[:, j], "--", alpha=0.6, label=f"GT joint {j}")
            axes[1, 0].plot(t, pred_np[:, j], "-", alpha=0.8, label=f"Pred joint {j}")
        axes[1, 0].set_title("Right Arm Joints (7-12)")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("Joint Position")
        axes[1, 0].legend(fontsize=6, ncol=2)

        # Bottom-right: grippers (6, 13)
        axes[1, 1].plot(t, gt_np[:, 6], "b--", label="GT left gripper")
        axes[1, 1].plot(t, pred_np[:, 6], "b-", label="Pred left gripper")
        axes[1, 1].plot(t, gt_np[:, 13], "r--", label="GT right gripper")
        axes[1, 1].plot(t, pred_np[:, 13], "r-", label="Pred right gripper")
        axes[1, 1].set_title("Grippers (6, 13)")
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("Gripper Position")
        axes[1, 1].legend()

        fig.suptitle(f"Episode {ep_idx + 1} | MSE: {mse:.6f}", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"episode_{ep_idx + 1}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")

    # Summary
    print(f"\n{'=' * 40}")
    print(f"Evaluated {len(all_mse)} episodes")
    print(f"Mean MSE: {np.mean(all_mse):.6f}")
    print(f"Std MSE:  {np.std(all_mse):.6f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play/Evaluate trained VITA Policy")
    parser.add_argument("--config", type=str, default="config/sim.yml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--save_dir", type=str, default="./outputs/play", help="Directory to save results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    play_sim(config, args.ckpt, args.num_episodes, args.save_dir)
