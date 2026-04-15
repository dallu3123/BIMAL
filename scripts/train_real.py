import argparse
import os
import random
import sys
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.policy import VITAPolicy
from utils.real_dataset import BimalRealDataset


@torch.no_grad()
def validate(
    policy: VITAPolicy,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Validation loop"""
    policy.eval()
    total_metrics: dict[str, float] = {}
    num_batches = 0

    for images_dict, actions in val_loader:
        images_dict = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
        actions = actions.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, metrics = policy.compute_loss(images_dict, actions)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

    avg_metrics = {f"val/{k}": v / num_batches for k, v in total_metrics.items()}
    policy.train()
    return avg_metrics


def train_real(config: dict) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 0. Initialize W&B
    wandb_cfg = config.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", True))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{wandb_cfg.get('run_name', 'real')}_{timestamp}"
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "BIMAL"),
            name=run_name,
            entity=wandb_cfg.get("entity"),
            mode=wandb_cfg.get("mode", "online"),
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
        vision_encoder=model_cfg.get("vision_encoder", "resnet18"),
        vision_freeze=model_cfg.get("vision_freeze", True),
    ).to(device)

    # Fine-tuning from a simulation checkpoint
    pretrained_ckpt = config["train"].get("pretrained_ckpt", "")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"Loading pretrained simulation checkpoint: {pretrained_ckpt}")
        policy.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    elif pretrained_ckpt:
        print(f"WARNING: Pretrained checkpoint not found at {pretrained_ckpt}. Training from scratch.")

    # torch.compile (선택적)
    compile_cfg = config["train"].get("compile", {})
    compile_enabled = bool(compile_cfg.get("enabled", False)) and device.type == "cuda"
    if compile_enabled:
        compile_mode = compile_cfg.get("mode", "reduce-overhead")
        print(f"torch.compile enabled (mode={compile_mode}) — first step will be slow (warm-up)")
        policy = torch.compile(policy, mode=compile_mode)

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    save_freq = config["train"]["save_freq"]

    # AMP 설정
    amp_cfg = config["train"].get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype_str = str(amp_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    # float16일 때만 GradScaler 필요 (bf16은 불필요)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    print(f"AMP: enabled={amp_enabled}, dtype={amp_dtype}")

    # run name 아래로 체크포인트 폴더 생성
    base_ckpt_dir = config["train"]["ckpt_dir"]
    ckpt_dir = os.path.join(base_ckpt_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # 2. Dataset (Train / Val split by episode)
    dataset = BimalRealDataset(
        data_dir=config["task"]["data_dir"],
        seq_len=config["task"]["seq_len"],
        image_size=config["task"].get("image_size", 224),
        camera_keys=tuple(
            config["task"].get(
                "camera_keys",
                [
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ],
            )
        ),
        use_cache=config["task"].get("use_cache", False),
        cache_dir=config["task"].get("cache_dir"),
    )

    num_episodes = len(dataset._episodes)
    val_ratio = config["train"].get("val_ratio", 0.1)
    num_val = max(1, int(num_episodes * val_ratio))
    num_train = num_episodes - num_val

    # 에피소드 단위 랜덤 split (시드 고정으로 재현 가능)
    split_seed = int(config["train"].get("split_seed", 42))
    rng = random.Random(split_seed)
    all_slots = list(range(num_episodes))
    rng.shuffle(all_slots)
    train_ep_set = set(all_slots[:num_train])
    val_ep_set = set(all_slots[num_train:])
    print(
        f"Episode split (seed={split_seed}): "
        f"train={sorted(train_ep_set)[:5]}... ({len(train_ep_set)} eps), "
        f"val={sorted(val_ep_set)} ({len(val_ep_set)} eps)"
    )

    train_indices, val_indices = [], []
    for flat_idx, (ep_slot, _start) in enumerate(dataset._samples):
        if ep_slot in train_ep_set:
            train_indices.append(flat_idx)
        else:
            val_indices.append(flat_idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Train: {len(train_dataset)} frames ({num_train} episodes)")
    print(f"Val:   {len(val_dataset)} frames ({num_val} episodes)")

    num_workers = config["train"].get("num_workers", 4)
    prefetch_factor = config["train"].get("prefetch_factor", 2) if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    policy.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Real Epoch {epoch + 1}/{epochs}") as pbar:
            for _batch_idx, (images_dict, actions) in enumerate(pbar):
                images_dict = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
                actions = actions.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss, metrics = policy.compute_loss(images_dict, actions)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                global_step = epoch * len(train_loader) + _batch_idx
                if use_wandb:
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
        val_metrics = validate(policy, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        val_loss = val_metrics["val/total_loss"]
        if use_wandb:
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
            best_path = os.path.join(ckpt_dir, "policy_real_best.pth")
            # torch.compile 래핑 시 원본 모델의 state_dict 저장
            state = policy._orig_mod.state_dict() if hasattr(policy, "_orig_mod") else policy.state_dict()
            torch.save(state, best_path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_real_ep{epoch + 1}.pth")
            state = policy._orig_mod.state_dict() if hasattr(policy, "_orig_mod") else policy.state_dict()
            torch.save(state, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    # 학습 종료 시, 최종 best 모델만 한 번 W&B에 업로드
    if use_wandb:
        final_best = os.path.join(ckpt_dir, "policy_real_best.pth")
        if os.path.exists(final_best):
            wandb.save(final_best)
            print(f"Uploaded final best model to W&B: {final_best}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VITA Policy on Real-world Data")
    parser.add_argument("--config", type=str, default="config/real.yml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_real(config)
