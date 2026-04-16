#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import sys
sys.path.append("./")

# numpy._core compatibility patch (numpy <1.25 vs >=1.25)
import numpy as _np_patch
if not hasattr(_np_patch, "_core"):
    import types as _types
    _np_patch._core = _np_patch.core
    import sys as _sys
    _sys.modules["numpy._core"] = _np_patch.core
    for _attr in dir(_np_patch.core):
        _mod = getattr(_np_patch.core, _attr)
        if isinstance(_mod, _types.ModuleType):
            _sys.modules[f"numpy._core.{_attr}"] = _mod

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)   # act의 상위 폴더
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
import torch.nn.functional as F
from gradcam_utils import MultiGradCAM, get_module_by_path, overlay_cam_on_bgr

import torch
import numpy as np
import pickle
import argparse
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, VQVAEACTPolicy, CNNMLPPolicy, DiffusionPolicy
# ── LeRobot 모델 래퍼 ──────────────────────────────────
try:
    from lerobot_policies import (
        LeRobotDiffusionPolicy,
        LeRobotACTPolicy,
        LeRobotVQBeTPolicy,
    )
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading


import atexit

def _cleanup():
    global video_writer, multi_gradcam
    try:
        if video_writer is not None:
            video_writer.release()
            print("[GradCAM-Video] video released")
    except Exception as e:
        print("[GradCAM-Video] release error:", e)

    try:
        if multi_gradcam is not None:
            multi_gradcam.close()
    except Exception as e:
        print("[GradCAM-Video] multi_gradcam close error:", e)

atexit.register(_cleanup)


task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

multi_gradcam = None
video_writer = None

def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max(math.floor(math.fabs(result[-1][j] - post_action[i][j]) / steps[j]) for j in range(pre_action.shape[0]))
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)
   
    # 如果是ACT策略
    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {
            'lr': args.lr,
            'lr_backbone': args.lr_backbone,
            'backbone': args.backbone,
            'masks': args.masks,
            'weight_decay': args.weight_decay,
            'dilation': args.dilation,
            'position_embedding': args.position_embedding,
            'loss_function': args.loss_function,
            'chunk_size': args.chunk_size,     # 查询
            'camera_names': task_config['camera_names'],
            'use_depth_image': args.use_depth_image,
            'use_robot_base': args.use_robot_base,
            'kl_weight': args.kl_weight,        # kl散度权重
            'hidden_dim': args.hidden_dim,      # 隐藏层维度
            'dim_feedforward': args.dim_feedforward,
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'dropout': args.dropout,
            'pre_norm': args.pre_norm
        }
    elif args.policy_class == 'VQVAEACT':
        policy_config = {
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "weight_decay": args.weight_decay,
            "masks": args.masks,
            "loss_function": args.loss_function,
            "dilation": args.dilation,
            "position_embedding": args.position_embedding,

            "backbone": args.backbone,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "dropout": args.dropout,
            "pre_norm": args.pre_norm,

            "camera_names": task_config["camera_names"],
            "use_depth_image": args.use_depth_image,
            "use_robot_base": args.use_robot_base,

            "chunk_size": args.chunk_size,

            # VQ params
            "vq_weight": args.vq_weight,
            "latent_dim": args.latent_dim,
            "num_latent_tokens": args.num_latent_tokens,
            "vq_codebook_size": args.vq_codebook_size,
            "vq_beta": args.vq_beta,

            # posterior / prior params
            "posterior_enc_layers": args.posterior_enc_layers,
            "posterior_dec_layers": args.posterior_dec_layers,
            "prior_weight": args.prior_weight,
            "prior_enc_layers": args.prior_enc_layers,
            "prior_dec_layers": args.prior_dec_layers,
            "cond_dropout": args.cond_dropout,
        }
    elif args.policy_class == 'CNNMLP':
        policy_config = {
            'lr': args.lr,
            'lr_backbone': args.lr_backbone,
            'backbone': args.backbone,
            'masks': args.masks,
            'weight_decay': args.weight_decay,
            'dilation': args.dilation,
            'position_embedding': args.position_embedding,
            'loss_function': args.loss_function,
            'chunk_size': 1,     # 查询
            'camera_names': task_config['camera_names'],
            'use_depth_image': args.use_depth_image,
            'use_robot_base': args.use_robot_base
        }

    elif args.policy_class == 'Diffusion':
        policy_config = {
            'lr': args.lr,
            'lr_backbone': args.lr_backbone,
            'backbone': args.backbone,
            'masks': args.masks,
            'weight_decay': args.weight_decay,
            'dilation': args.dilation,
            'position_embedding': args.position_embedding,
            'loss_function': args.loss_function,
            'chunk_size': args.chunk_size,     # 查询
            'camera_names': task_config['camera_names'],
            'use_depth_image': args.use_depth_image,
            'use_robot_base': args.use_robot_base,
            'observation_horizon': args.observation_horizon,
            'action_horizon': args.action_horizon,
            'num_inference_timesteps': args.num_inference_timesteps,
            'ema_power': args.ema_power
        }
    # ── LeRobot 모델들 ──────────────────────────────────
    elif args.policy_class == 'LeRobotDiffusion':
        policy_config = {
            'lr':                       args.lr,
            'weight_decay':             args.weight_decay,
            'camera_names':             task_config['camera_names'],
            'state_dim':                args.state_dim,
            'chunk_size':               args.chunk_size,
            'observation_horizon':      args.observation_horizon,
            'use_depth_image':          args.use_depth_image,
            'use_robot_base':           args.use_robot_base,
            'pretrained_path':          getattr(args, 'pretrained_path', ''),
            'num_inference_timesteps':  args.num_inference_timesteps,
            'image_shape':         (3, 480, 848),
        }
    elif args.policy_class == 'LeRobotACT':
        policy_config = {
            'lr':                  args.lr,
            'weight_decay':        args.weight_decay,
            'camera_names':        task_config['camera_names'],
            'state_dim':           args.state_dim,
            'chunk_size':          args.chunk_size,
            'observation_horizon': getattr(args, 'observation_horizon', 1),
            'kl_weight':           args.kl_weight,
            'hidden_dim':          args.hidden_dim,
            'dim_feedforward':     args.dim_feedforward,
            'enc_layers':          args.enc_layers,
            'dec_layers':          args.dec_layers,
            'nheads':              args.nheads,
            'use_depth_image':     args.use_depth_image,
            'use_robot_base':      args.use_robot_base,
            'pretrained_path':     getattr(args, 'pretrained_path', ''),
        }
    elif args.policy_class == 'LeRobotVQBeT':
        policy_config = {
            'lr':                  args.lr,
            'weight_decay':        args.weight_decay,
            'camera_names':        task_config['camera_names'],
            'state_dim':           args.state_dim,
            'chunk_size':          args.chunk_size,
            'observation_horizon': args.observation_horizon,
            'use_depth_image':     args.use_depth_image,
            'use_robot_base':      args.use_robot_base,
            'pretrained_path':     getattr(args, 'pretrained_path', ''),
        }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': args.state_dim,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'VQVAEACT':
        policy = VQVAEACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    # ── LeRobot 모델들 ──────────────────────────────────
    elif policy_class == 'LeRobotDiffusion':
        policy = LeRobotDiffusionPolicy(policy_config)
    elif policy_class == 'LeRobotACT':
        policy = LeRobotACTPolicy(policy_config)
    elif policy_class == 'LeRobotVQBeT':
        policy = LeRobotVQBeTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep

    global video_writer, multi_gradcam   # ✅ 여기 추가(함수 최상단)
    
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        #image_dict[config['camera_names'][0]] = img_front
        #image_dict[config['camera_names'][1]] = img_left
        #image_dict[config['camera_names'][2]] = img_right

        # 수정
        image_dict[config['camera_names'][0]] = img_front
        if len(config['camera_names']) > 1:
            image_dict[config['camera_names'][1]] = img_left
        if len(config['camera_names']) > 2:
            image_dict[config['camera_names'][2]] = img_right


        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            #image_depth_dict[config['camera_names'][0]] = img_front_depth
            #image_depth_dict[config['camera_names'][1]] = img_left_depth
            #image_depth_dict[config['camera_names'][2]] = img_right_depth
            image_depth_dict[config['camera_names'][0]] = img_front_depth
            if len(config['camera_names']) > 1:
                image_depth_dict[config['camera_names'][1]] = img_left_depth
            if len(config['camera_names']) > 2:
                image_depth_dict[config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict

        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])


        def _resize_h_keep_ar(bgr, H):
            h, w = bgr.shape[:2]
            if h == H:
                return bgr
            new_w = int(round(w * (H / float(h))))
            return cv2.resize(bgr, (new_w, H), interpolation=cv2.INTER_AREA)

        start_time = time.time()

        # 우선순위: video > single gradcam > normal
        if args.gradcam_video and multi_gradcam is not None and config['policy_class'] in ["ACT", "VQVAEACT"]:
            # --- 3캠 Grad-CAM + 3분할 mp4 ---
            target_dim = None if args.gradcam_dim < 0 else args.gradcam_dim

            cams, a_hat = multi_gradcam(
                curr_image, curr_depth_image, qpos,
                target_step=args.gradcam_step,
                target_dim=target_dim
            )
            all_actions = a_hat.detach()

            # (필요하면 여기서 오버레이/concat/video_writer.write(frame) 수행)
            # ※ 이 부분은 너가 이미 넣어둔 “3분할 합치기 + write” 코드 그대로 넣으면 됨.

            def _make_vis(bgr, cam_tensor):
                H, W = bgr.shape[:2]
                cam_up = F.interpolate(cam_tensor, size=(H, W), mode="bilinear", align_corners=False)
                cam_np = cam_up[0, 0].detach().cpu().numpy()
                return overlay_cam_on_bgr(bgr, cam_np, alpha=args.gradcam_alpha)
            
            """
            # 기존
            vis_front = _make_vis(img_front, cams["front"])
            vis_left  = _make_vis(img_left,  cams["left"])
            vis_right = _make_vis(img_right, cams["right"])

            # 높이 통일 후 가로 concat
            Hpanel = args.gradcam_resize_h
            vis_front = _resize_h_keep_ar(vis_front, Hpanel)
            vis_left  = _resize_h_keep_ar(vis_left,  Hpanel)
            vis_right = _resize_h_keep_ar(vis_right, Hpanel)

            frame = np.concatenate([vis_front, vis_left, vis_right], axis=1)  # (H, Wsum, 3) BGR
            """

            # 수정
            Hpanel = args.gradcam_resize_h
            panels = []
            vis_front = _make_vis(img_front, cams["front"])
            panels.append(_resize_h_keep_ar(vis_front, Hpanel))
            if "left" in cams:
                vis_left = _make_vis(img_left, cams["left"])
                panels.append(_resize_h_keep_ar(vis_left, Hpanel))
            if "right" in cams:
                vis_right = _make_vis(img_right, cams["right"])
                panels.append(_resize_h_keep_ar(vis_right, Hpanel))

            frame = np.concatenate(panels, axis=1)

            # VideoWriter lazy init
            if video_writer is None:
                # 경로에 폴더가 있으면 만들어주기
                out_dir = os.path.dirname(args.gradcam_video_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.gradcam_video_path, fourcc, args.gradcam_fps, (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter: {args.gradcam_video_path}")
                print(f"[GradCAM-Video] Writing mp4: {args.gradcam_video_path} ({w}x{h} @{args.gradcam_fps}fps)")

            video_writer.write(frame)

        else:
            # --- 일반 추론 (중요: thread라서 inference_mode 적용 안 됨 -> 반드시 no_grad) ---
            with torch.no_grad():
                all_actions = policy(curr_image, curr_depth_image, qpos)

        end_time = time.time()
        print("model cost time: ", end_time - start_time)


        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型数据  继承nn.Module
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)
    
    # 2 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    global multi_gradcam, video_writer

    if args.gradcam_video and config['policy_class'] in ["ACT", "VQVAEACT"]:
        """
        # 기존
        # cam_id 0/1/2 각각 layer4
        layer_front = get_module_by_path(policy, "model.backbones.0.0.body.layer4")
        layer_left  = get_module_by_path(policy, "model.backbones.1.0.body.layer4")
        layer_right = get_module_by_path(policy, "model.backbones.2.0.body.layer4")

        multi_gradcam = MultiGradCAM(policy, {
            "front": layer_front,
            "left":  layer_left,
            "right": layer_right,
        })
        """
        # 수정
        cam_layers = {}
        cam_layers["front"] = get_module_by_path(policy, "model.backbones.0.0.body.layer4")
        if len(config['camera_names']) > 1:
            cam_layers["left"]  = get_module_by_path(policy, "model.backbones.1.0.body.layer4")
        if len(config['camera_names']) > 2:
            cam_layers["right"] = get_module_by_path(policy, "model.backbones.2.0.body.layer4")

        multi_gradcam = MultiGradCAM(policy, cam_layers)
        print("[GradCAM-Video] MultiGradCAM enabled for front/left/right layer4")

        # VideoWriter는 프레임 크기를 알아야 하니, 첫 프레임에서 lazy-init 하는 게 가장 안전함
        video_writer = None

    # 4 加载统计值
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    # 发布基础的姿态
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0
            rate = rospy.Rate(args.publish_rate)
            # LeRobot 모델 내부 obs buffer 초기화
            if hasattr(policy, "reset"):
                policy.reset()
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])
            all_actions = None
            while t < max_publish_step and not rospy.is_shutdown():
                # start_time = time.time()
                # query policy
                if config['policy_class'] in ["ACT", "VQVAEACT"]:
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, ros_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01

                        # 현재 (오래된 예측에 높은 가중치)
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))

                        # 변경 (최신 예측에 높은 가중치)
                        #exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))[::-1])

                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                elif config['policy_class'] in ['LeRobotDiffusion', 'LeRobotACT', 'LeRobotVQBeT']:
                    # 비동기 추론: t >= max_t이면 추론 스레드 시작하되 join하지 않음.
                    # 추론 중에도 이전 chunk 캐시를 계속 publish → 끊김 없이 부드럽게 동작.
                    if t >= max_t:
                        # 이전 추론 스레드가 아직 돌고 있으면 결과 수거
                        if inference_thread is not None and not inference_thread.is_alive():
                            inference_lock.acquire()
                            if inference_actions is not None:
                                all_actions = inference_actions
                                inference_actions = None
                                max_t = t + args.pos_lookahead_step
                                if config['temporal_agg']:
                                    all_time_actions[[t], t:t + chunk_size] = all_actions
                            inference_lock.release()
                            inference_thread = None

                        # 새 추론 스레드 시작 (join 없이)
                        if inference_thread is None:
                            pre_action = action
                            inference_thread = threading.Thread(
                                target=inference_process,
                                args=(args, config, ros_operator, policy, stats, t, pre_action)
                            )
                            inference_thread.daemon = True
                            inference_thread.start()

                        # 추론 완료 여부와 무관하게 결과 수거 시도
                        inference_lock.acquire()
                        if inference_actions is not None:
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                            inference_thread = None
                        inference_lock.release()

                    if all_actions is None:
                        rate.sleep()
                        t += 1
                        continue

                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        if len(actions_for_curr_step) == 0:
                            rate.sleep()
                            t += 1
                            continue
                        k = 0.01

                        # 현재 (오래된 예측에 높은 가중치)
                        #exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))

                        # 변경 (최신 예측에 높은 가중치)
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))[::-1])

                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                left_action = action[:7]  # 取7维度
                right_action = action[7:14]
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                if args.use_robot_base:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()
    

    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print("[GradCAM-Video] video released")

    if multi_gradcam is not None:
        multi_gradcam.close()
        multi_gradcam = None


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        img_left = cv2.flip(img_left, -1)  # 상하 반전

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str,
                        help='policy_class: ACT, VQVAEACT, CNNMLP, Diffusion, LeRobotDiffusion, LeRobotACT, LeRobotVQBeT',
                        default='ACT', required=False)
    # ── LeRobot 전용 인자 ──────────────────────────────
    parser.add_argument('--pretrained_path', action='store', type=str,
                        help='LeRobot HF Hub 모델 ID 또는 로컬 경로. 예: lerobot/diffusion_pusht',
                        default='', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    
    # for VQVAEACT
    parser.add_argument('--vq_weight', type=float, default=1.0)
    parser.add_argument('--num_latent_tokens', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--vq_codebook_size', type=int, default=512)
    parser.add_argument('--vq_beta', type=float, default=0.25)
    parser.add_argument('--prior_weight', type=float, default=0.1)
    parser.add_argument('--prior_enc_layers', type=int, default=None)
    parser.add_argument('--prior_dec_layers', type=int, default=None)
    parser.add_argument('--cond_dropout', type=float, default=0.0)
    parser.add_argument('--posterior_enc_layers', type=int, default=None)
    parser.add_argument('--posterior_dec_layers', type=int, default=None)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    # Grad-CAM options
    parser.add_argument('--gradcam_step', type=int, default=0, help='which predicted step T to explain')
    parser.add_argument('--gradcam_dim', type=int, default=-1,
                        help='which action dim D to explain. -1 => use L2 norm over dims')
    parser.add_argument('--gradcam_every', type=int, default=1, help='save every N inference calls')
    parser.add_argument('--gradcam_alpha', type=float, default=0.45, help='overlay alpha')

    # Grad-CAM video (3-split mp4)
    parser.add_argument('--gradcam_video', action='store_true', help='write 3-split gradcam mp4')
    parser.add_argument('--gradcam_video_path', type=str, default='gradcam_3split.mp4')
    parser.add_argument('--gradcam_fps', type=int, default=30)
    parser.add_argument('--gradcam_resize_h', type=int, default=360, help='height for each panel')


    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/
'''
python inference.py \
  --ckpt_dir ~/cobot_magic/ckpt_hangju2 \
  --img_front_topic /cam_high/color/image_raw \
  --img_left_topic /cam_left_wrist/color/image_raw \
  --img_right_topic /cam_right_wrist/color/image_raw \
  --puppet_arm_left_topic /puppet/joint_left \
  --puppet_arm_right_topic /puppet/joint_right \
  --puppet_arm_left_cmd_topic /master/joint_left \
  --puppet_arm_right_cmd_topic /master/joint_right \
  --gradcam_video \
  --gradcam_video_path gradcam_3split.mp4
'''