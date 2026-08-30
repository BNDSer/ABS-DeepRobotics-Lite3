"""Play a policy and encode Isaac Gym camera frames without creating a viewer."""

import argparse
import math
import os
import shutil
import subprocess
import sys

# BaseTask reads this before the environment is constructed.
os.environ.setdefault("LEGGED_GYM_OFFSCREEN_RENDER", "1")

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


def parse_video_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video_seconds", type=float, default=30.0)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)
    parser.add_argument("--video_path", type=str, default=None)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return options


def play(options):
    args = get_args()
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.erfi = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.domain_rand.randomize_timer_minus = 0.0

    # The existing camera is attached to the robot and is created before prepare_sim().
    env_cfg.sensors.depth_cam.enable = True
    env_cfg.sensors.depth_cam.resolution = [options.video_width, options.video_height]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = False
    env.terrain_levels[:] = min(9, env.max_terrain_level - 1)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("Loaded policy from:", task_registry.loaded_policy_path)

    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies"
    )
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        export_dir,
        os.path.basename(task_registry.loaded_policy_path),
    )

    video_path = options.video_path or os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "Lite3_pos_rough_headless.mp4",
    )
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the headless video")

    width = options.video_width
    height = options.video_height
    fps = options.video_fps
    ffmpeg_cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        video_path,
    ]
    encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    obs = env.get_observations()
    sim_steps = int(math.ceil(options.video_seconds / env.dt))
    capture_every = max(1, int(round(1.0 / (fps * env.dt))))
    captured = 0

    try:
        with torch.no_grad():
            for step in range(sim_steps):
                actions = policy(obs.detach())
                obs, _, _, _, _ = env.step(actions.detach())

                if step % capture_every != 0:
                    continue
                frame = env.get_color_image(0).numpy()
                if frame.shape != (height, width, 4):
                    raise RuntimeError(f"Unexpected RGB frame shape: {frame.shape}")
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                captured += 1
    except KeyboardInterrupt:
        print("Interrupted; finalizing the frames captured so far")
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {encoder.returncode}")
    print(f"Saved {captured} frames to {video_path}")


if __name__ == "__main__":
    play(parse_video_options())
