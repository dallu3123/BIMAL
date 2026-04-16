#!/usr/bin/env python3
# -- coding: UTF-8
"""
VITA Policy 실시간 ROS 추론 스크립트.

ACT용 inference.py와 달리:
  - qpos / depth / dataset_stats 미사용 (vision-only)
  - VITAPolicy.get_action(images_dict) 만 호출
  - 액션은 raw 단위 그대로 publish (정규화 없음)

ROS I/O (RosOperator)는 inference.py에서 import하여 재사용.
"""

import os
import sys
import time
import threading
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

import cv2
import numpy as np
import torch
import yaml
import rospy

from model.policy import VITAPolicy
from inference import RosOperator  # ROS 콜백/sync/publish 그대로 재사용


# ─── 전역 (ACT 코드와 동일한 패턴: 비동기 추론용) ───────────────
inference_lock = threading.Lock()
inference_actions = None       # np.ndarray (1, seq_len, action_dim) | None
inference_thread = None


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_policy(cfg, device):
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]
    policy = VITAPolicy(
        action_dim=task_cfg["action_dim"],
        seq_len=task_cfg["seq_len"],
        num_cameras=task_cfg["num_cameras"],
        latent_dim=model_cfg["latent_dim"],
        flow_hidden_dim=model_cfg["flow_hidden_dim"],
        flow_num_layers=model_cfg["flow_num_layers"],
        num_sampling_steps=model_cfg["num_sampling_steps"],
        decode_flow_latents=model_cfg.get("decode_flow_latents", True),
        consistency_weight=model_cfg.get("consistency_weight", 1.0),
        enc_contrastive_weight=model_cfg.get("enc_contrastive_weight", 0.0),
        flow_contrastive_weight=model_cfg.get("flow_contrastive_weight", 0.0),
        sigma=model_cfg.get("sigma", 0.0),
        vision_encoder=model_cfg.get("vision_encoder", "dinov2_vits14"),
        vision_freeze=model_cfg.get("vision_freeze", True),
    ).to(device)
    return policy


def preprocess_image(img, image_size, bgr_to_rgb):
    """ROS 카메라 frame → (3, image_size, image_size) float tensor in [0, 1].

    학습 파이프라인(utils/real_dataset.py:_to_tensor)과 동일하게:
      BGR→RGB → resize(INTER_AREA) → /255.0
    ImageNet mean/std normalize는 모델 내부(DINOv2Observer/ResNetObserver)에서 처리하므로
    여기선 하지 않는다.
    """
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    return t


def build_images_dict(img_front, img_left, img_right, args, device):
    """sorted() 순서가 학습 시(observation.images.cam_high < ...left_wrist < ...right_wrist)와
    동일하도록 ROS 측에서도 cam_high < cam_left_wrist < cam_right_wrist 키를 사용한다.
    MultiCameraObserver는 sorted(images_dict.keys()) 순서로 fusion하므로 prefix 차이는 무관."""
    d = {}
    d["cam_high"] = preprocess_image(img_front, args.image_size, args.bgr_to_rgb).unsqueeze(0).to(device)
    d["cam_left_wrist"] = preprocess_image(img_left, args.image_size, args.bgr_to_rgb).unsqueeze(0).to(device)
    d["cam_right_wrist"] = preprocess_image(img_right, args.image_size, args.bgr_to_rgb).unsqueeze(0).to(device)
    return d


def inference_worker(args, ros_operator, policy, device):
    """한 번의 frame을 읽어 policy.get_action을 돌리고 inference_actions에 저장."""
    global inference_actions
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    while not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, _, _, _, _, _, _) = result

        images_dict = build_images_dict(img_front, img_left, img_right, args, device)

        t0 = time.time()
        with torch.no_grad():
            actions = policy.get_action(images_dict)   # (1, seq_len, action_dim)
        t1 = time.time()
        print(f"VITA inference cost: {(t1 - t0) * 1000:.1f} ms")

        actions_np = actions.detach().cpu().numpy()
        with inference_lock:
            inference_actions = actions_np
        return


def model_inference(args, cfg, ros_operator):
    global inference_actions, inference_thread

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 모델
    policy = build_policy(cfg, device)

    # 2) 체크포인트 로드 (없으면 디버그용으로 random init 사용)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        state = torch.load(args.ckpt_path, map_location=device)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing:
            print(f"[VITA] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[VITA] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"[VITA] checkpoint loaded: {args.ckpt_path}")
    else:
        print(f"[VITA] WARNING: ckpt_path '{args.ckpt_path}' not found. Running with RANDOM weights.")

    policy.eval()

    seq_len = cfg["task"]["seq_len"]
    action_dim = cfg["task"]["action_dim"]
    max_publish_step = args.max_publish_step

    # 3) 초기 자세 (ACT inference.py와 동일한 워밍업)
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
             -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
              -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
             -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375,
              -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.puppet_arm_publish_continuous(left1, right1)

    # 4) 추론/publish 루프 (LeRobot 비동기 패턴: 추론 중에도 이전 chunk 계속 publish)
    rate = rospy.Rate(args.publish_rate)
    all_actions = None        # 현재 사용 중인 chunk (1, seq_len, action_dim)
    chunk_step = 0            # 현재 chunk에서 몇 번째 step을 publish 중인지
    t = 0
    refresh_every = max(1, args.chunk_refresh)   # 몇 step마다 새 추론을 trigger할지

    with torch.inference_mode():
        while not rospy.is_shutdown() and t < max_publish_step:
            # 새 추론 trigger 조건
            need_refresh = (all_actions is None) or (chunk_step >= refresh_every)
            if need_refresh:
                # 이전 스레드 결과 수거
                if inference_thread is not None and not inference_thread.is_alive():
                    with inference_lock:
                        if inference_actions is not None:
                            all_actions = inference_actions
                            inference_actions = None
                            chunk_step = 0
                    inference_thread = None

                # 새 스레드 시작 (join 안 함 — 비동기)
                if inference_thread is None:
                    inference_thread = threading.Thread(
                        target=inference_worker,
                        args=(args, ros_operator, policy, device),
                        daemon=True,
                    )
                    inference_thread.start()

                # 추론 완료 즉시 수거 시도 (이미 끝났으면)
                with inference_lock:
                    if inference_actions is not None:
                        all_actions = inference_actions
                        inference_actions = None
                        chunk_step = 0
                        inference_thread = None

            # 첫 chunk가 아직 안 왔으면 대기
            if all_actions is None:
                rate.sleep()
                continue

            # 현재 step의 action을 publish
            idx = min(chunk_step, seq_len - 1)
            action = all_actions[0, idx]   # (action_dim,)
            left_action = action[:7]
            right_action = action[7:14]
            ros_operator.puppet_arm_publish(left_action, right_action)

            chunk_step += 1
            t += 1
            rate.sleep()


def get_arguments():
    parser = argparse.ArgumentParser(description="VITA Policy ROS Inference")

    # 모델 / config
    parser.add_argument("--config", type=str, default="config/real.yml",
                        help="학습 시 사용한 yaml config (모델 차원 일치용)")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="VITA 체크포인트 .pth 경로 (예: ./outputs/checkpoints/real/<run>/policy_real_best.pth)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="모델 입력 정사각 크기 (DINOv2 = 224 권장, 14 배수)")
    parser.add_argument("--bgr_to_rgb", action="store_true", default=True,
                        help="ROS frame이 BGR이면 True. RealSense passthrough가 RGB면 --no_bgr_to_rgb")
    parser.add_argument("--no_bgr_to_rgb", dest="bgr_to_rgb", action="store_false")

    # 추론 동작
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--publish_rate", type=int, default=40)
    parser.add_argument("--chunk_refresh", type=int, default=8,
                        help="N step마다 새 추론을 trigger (seq_len 이내 권장)")

    # ROS 토픽 (inference.py 기본값과 동일)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic",  type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic",  type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--puppet_arm_left_cmd_topic",  type=str, default="/master/joint_left")
    parser.add_argument("--puppet_arm_right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic",  type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--robot_base_topic",     type=str, default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic", type=str, default="/cmd_vel")

    # RosOperator가 참조하지만 VITA에선 항상 끔
    parser.add_argument("--use_depth_image", type=bool, default=False)
    parser.add_argument("--use_robot_base",  type=bool, default=False)
    parser.add_argument("--arm_steps_length", type=float,
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
                        help="puppet_arm_publish_continuous 보간 step 크기")

    return parser.parse_args()


def main():
    args = get_arguments()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    cfg = load_config(cfg_path)
    print(f"Loaded config: {cfg_path}")

    ros_operator = RosOperator(args)
    model_inference(args, cfg, ros_operator)


if __name__ == "__main__":
    main()
