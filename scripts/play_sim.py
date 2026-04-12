import argparse
import os
import sys

# 프로젝트 루트 및 utils를 Python path에 추가
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model.policy import VITAPolicy
from utils.sim_env import BOX_POSE, make_sim_env


def run_episode(
    policy: VITAPolicy,
    task_name: str,
    camera_name: str,
    max_timesteps: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[float], float]:
    """
    Closed-loop rollout: 매 timestep마다 관측 이미지를 받아 정책으로 액션을 예측하고 환경에 적용.

    Returns:
        frames: 렌더링된 이미지 리스트
        rewards: 각 timestep의 reward 리스트
        max_reward: 에피소드 최대 reward
    """
    # 큐브 초기 위치 랜덤 설정
    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    env = make_sim_env(task_name)
    ts = env.reset()

    frames = []
    rewards = []

    for t in range(max_timesteps):
        # 관측 이미지를 모델 입력 형태로 변환
        obs_img = ts.observation["images"][camera_name].copy()  # (480, 640, 3) uint8
        img_tensor = torch.from_numpy(obs_img).float() / 255.0  # (H, W, C)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

        images_dict = {camera_name: img_tensor}

        # 정책으로 액션 예측
        with torch.no_grad():
            pred_actions = policy.get_action(images_dict)  # (1, seq_len, 14)

        # 첫 번째 액션만 사용 (receding horizon)
        action = pred_actions[0, 0].cpu().numpy()  # (14,)

        # 환경에 액션 적용
        ts = env.step(action)
        reward = ts.reward
        rewards.append(reward)
        frames.append(obs_img.copy())

    max_reward = max(rewards) if rewards else 0
    return frames, rewards, max_reward


def save_video(frames: list[np.ndarray], save_path: str, fps: int = 50) -> None:
    """프레임 리스트를 mp4 비디오로 저장"""
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def save_reward_plot(rewards: list[float], max_reward: float, success: bool, save_path: str) -> None:
    """Reward 변화 그래프 저장"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, "b-", linewidth=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_ylim(-0.5, 4.5)
    ax.axhline(y=4, color="g", linestyle="--", alpha=0.5, label="Max Reward (4)")

    status = "SUCCESS" if success else "FAIL"
    color = "green" if success else "red"
    ax.set_title(f"[{status}] Max Reward: {max_reward:.0f}", fontsize=14, color=color)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


REWARD_DESCRIPTIONS = {
    0: "No contact",
    1: "Right gripper touched cube",
    2: "Cube lifted (right gripper)",
    3: "Left gripper touched cube (transfer attempted)",
    4: "Successful transfer (cube in left gripper, off table)",
}


def play_sim(
    config: dict,
    ckpt_path: str,
    task_name: str = "sim_transfer_cube_scripted",
    camera_name: str = "top",
    num_episodes: int = 10,
    max_timesteps: int = 400,
    save_dir: str = "./outputs/play",
) -> None:
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
    print(f"Task: {task_name} | Camera: {camera_name} | Max timesteps: {max_timesteps}")

    os.makedirs(save_dir, exist_ok=True)

    # 2. Run Episodes
    successes = []
    max_rewards = []

    for ep_idx in range(num_episodes):
        frames, rewards, max_reward = run_episode(
            policy, task_name, camera_name, max_timesteps, device
        )

        success = max_reward == 4  # TransferCube: reward 4 = 성공
        successes.append(success)
        max_rewards.append(max_reward)

        status = "SUCCESS" if success else "FAIL"
        desc = REWARD_DESCRIPTIONS.get(int(max_reward), "Unknown")
        print(f"Episode {ep_idx + 1:3d}: [{status}] Max Reward = {max_reward:.0f} ({desc})")

        # 비디오 저장
        video_path = os.path.join(save_dir, f"episode_{ep_idx + 1}.mp4")
        save_video(frames, video_path)

        # Reward 그래프 저장
        plot_path = os.path.join(save_dir, f"episode_{ep_idx + 1}_reward.png")
        save_reward_plot(rewards, max_reward, success, plot_path)

    # 3. Summary
    num_success = sum(successes)
    success_rate = num_success / num_episodes
    avg_max_reward = np.mean(max_rewards)

    # Reward 분포
    reward_counts = {r: 0 for r in range(5)}
    for mr in max_rewards:
        reward_counts[int(mr)] += 1

    print(f"\n{'=' * 60}")
    print(f"  CLOSED-LOOP EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'=' * 60}")
    print(f"  Success Rate     : {num_success}/{num_episodes} ({success_rate:.1%})")
    print(f"  Avg Max Reward   : {avg_max_reward:.2f}")
    print(f"{'─' * 60}")
    print(f"  Reward Distribution:")
    for r in range(5):
        bar = "#" * reward_counts[r]
        desc = REWARD_DESCRIPTIONS[r]
        print(f"    Reward {r}: {reward_counts[r]:3d} {bar:20s} ({desc})")
    print(f"{'─' * 60}")
    print(f"  Videos saved to: {save_dir}")
    print(f"{'=' * 60}")

    # Summary 이미지
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d32f2f" if not s else "#388e3c" for s in successes]
    ax.bar(range(1, num_episodes + 1), max_rewards, color=colors)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max Reward")
    ax.set_ylim(-0.5, 4.5)
    ax.set_title(f"Success Rate: {success_rate:.1%} ({num_success}/{num_episodes})", fontsize=14)
    ax.axhline(y=4, color="g", linestyle="--", alpha=0.5, label="Success threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop Evaluation of VITA Policy in MuJoCo Sim")
    parser.add_argument("--config", type=str, default="config/sim.yml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--task", type=str, default="sim_transfer_cube_scripted", help="Task name")
    parser.add_argument("--camera", type=str, default="top", help="Camera name for observation")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max_timesteps", type=int, default=400, help="Max timesteps per episode")
    parser.add_argument("--save_dir", type=str, default="./outputs/play", help="Directory to save results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    play_sim(config, args.ckpt, args.task, args.camera, args.num_episodes, args.max_timesteps, args.save_dir)
