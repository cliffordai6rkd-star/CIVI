import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch

import wandb
import json
import random
from omegaconf import open_dict, OmegaConf

# from unified_video_action.utils.load_env import load_env_runner, load_env_runner_generator
from libero.force.force_benchmark import LiberoForceSocketEvaluator


"""
Usage:
cd ~/code/LIBERO-FT/
conda activate libero
export PYTHONPATH=~code/LIBERO-FT/
CUDA_VISIBLE_DEVICES=5 python eval_socket.py  \
    -o output/dp_force_eval  \
    -p 7075  \
    --close_online  \
    -t k1_open_top_put_bowl  
"""

LOAD_CAMERAS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
]

@click.command()
@click.option("-c", "--checkpoint", required=False, default=None, help="Not used")
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-p", "--port", default=6060, help="Port for socket communication")
@click.option("-t", "--task_id", default="", help="Task ID of HDF5 dataset")
@click.option('--close_online', is_flag=True, default=False, help='whether to open online updating')
@click.option("-f", "--friction_scale", default=1., help="Scale the friction force")
@click.option("-g", "--gz_scale", default=1., help="Scale the gravity z force")
@click.option("-s", "--solref_scale", default=1., help="Scale the solref force")
@click.option("-vd", "--visual_darken", is_flag=True, default=False, help="Enable single flashlight darken shift")
@click.option("-vc", "--visual_color", is_flag=True, default=False, help="Enable visual color light shift")
@click.option("-vt", "--visual_texture", is_flag=True, default=False, help="Enable visual table texture shift")
@click.option("--visual_viewshift", is_flag=True, default=False, help="Enable visual camera view shift (alias: -vv)")
@click.option("-v", "--vis", is_flag=True, default=False, help='whether to save rollout videos')
def main(checkpoint, output_dir, device, port,
         task_id: str = "", close_online: bool = False,
         friction_scale: float = 1., gz_scale: float = 1., solref_scale: float = 1.,
         visual_darken: bool = False, visual_color: bool = False, visual_texture: bool = False,
         visual_viewshift: bool = False,
         vis: bool = False,
         ):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    eval_socket_config = OmegaConf.create({
        "training": {
            "seed": 42,
            "debug": False,  # ori:False
        },
        "output_dir": output_dir,
        "task": {
            "name": "libero10",
            "dataset": {
                "_target_": "unified_video_action.env_runner.libero_socket_runner.LiberoDataset",
                "dataset_path": "data/libero_10",
            },
            "env_runner": {
                "_target_": "unified_video_action.env_runner.libero_socket_runner.LiberoImageSocketRunner",
                "dataset_path": "data/libero_10",
                "abs_action": True,
                "crf": 22,
                "fps": 10,
                "max_steps": 400,  # ori:400
                "n_action_steps": 1,  # ori:8
                "n_envs": None,  # ori: None
                "n_obs_steps": 4*3,  # ori:4+1, now:4*5 for online update
                "n_test": 50,  # ori:50
                "n_test_vis": 1,  # ori:1
                "n_train": 0,  # ori:1
                "n_train_vis": 0,  # ori:1
                "past_action": False,
                "render_obs_key": "agentview_image",
                "test_start_seed": 130000,  # ori:100000
                "tqdm_interval_sec": 1.0,
                "shape_meta": {
                    "image_resolution": 128,
                    "action": {"shape": (10,),},
                    "obs": {
                        "agentview_image": {"shape": (3, 128, 128), "type": "rgb"},
                    }
                },
                # socket related
                "policy_url": f"http://localhost:{port}",
                "send_per_frames": 1,  # ori:12, will be modified in init() and reset()
            },
        },
    })
    cfg = eval_socket_config

    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Tell env to load more data
    # DEBUG: add extra observation keys
    if True and hasattr(cfg.task.env_runner, 'shape_meta') and cfg.task.env_runner.shape_meta is not None:

        # 转换为普通字典进行修改
        shape_meta = OmegaConf.to_container(cfg.task.env_runner.shape_meta, resolve=True)
        shape_meta['obs']["robot0_eye_in_hand_image"] = {
            "shape": [3, 128, 128],
            "type": "rgb"
        }
        shape_meta['obs']["robot0_joint_pos"] = {
            "shape": [7],
            "type": "low_dim"
        }
        shape_meta['obs']["robot0_eef_pos"] = {
            "shape": [3],
            "type": "low_dim"
        }
        shape_meta['obs']["robot0_eef_quat"] = {
            "shape": [4],
            "type": "low_dim"
        }
        shape_meta['obs']["robot0_gripper_qpos"] = {
            "shape": [2],
            "type": "low_dim"
        }

        # 重新创建OmegaConf对象替换原配置
        cfg.task.env_runner.shape_meta = OmegaConf.create(shape_meta)
        print("[DEBUG] Updated shape_meta with eye_in_hand_rgb")

    # Op1. Load all envs at once
    # env_runners = load_env_runner(cfg, output_dir)
    # Op2. Load envs one by one using generator (save memory)
    # env_runners = load_env_runner_generator(cfg, output_dir)
    dataset_root = "/home/geyuan/code/LIBERO-FT/libero/datasets/libero_force"
    # dataset_subname = "libero_90"
    # hdf5_fn = "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_demo.hdf5"
    dataset_subname = "libero_90"
    HDF5_MAP = {
        ## Debug
        # "tmp": "tmp_replayed_wrench.hdf5",

        ## Kitchen
        "k10_close_put_blackbowl": "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_wrench.hdf5",
        "k1_open_top": "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo_wrench.hdf5",
        "k5_close": "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo_wrench.hdf5",
        "k10_close": "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo_wrench.hdf5",
        "k2_open": "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo_wrench.hdf5",
        "k6_close": "KITCHEN_SCENE6_close_the_microwave_demo_wrench.hdf5",
        "k1_open_bottom": "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo_wrench.hdf5",
        "k4_close_bottom_open_top": "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_demo_wrench.hdf5",
        "k7_open": "KITCHEN_SCENE7_open_the_microwave_demo_wrench.hdf5",
        "k1_open_top_put_bowl": "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench.hdf5",
        "k4_close_bottom": "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo_wrench.hdf5",
        #
        # ## Study
        "s3_pick_book_place_leftcaddy": "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo_wrench.hdf5",
    }

    # Helper function to run single task evaluation
    def run_single_task(
        current_task_id,
        cfg,
        seed,
        close_online,
        friction_scale,
        gz_scale,
        solref_scale,
        visual_darken,
        visual_color,
        visual_texture,
        visual_viewshift,
        vis,
    ):
        """Run evaluation for a single task and return json_log with test_mean_score"""
        assert current_task_id in HDF5_MAP, f"task_id {current_task_id} not found in HDF5_MAP"
        hdf5_fn = HDF5_MAP[current_task_id]
        hdf5_read_path = os.path.join(dataset_root, dataset_subname, hdf5_fn)

        env_runners = [
            LiberoForceSocketEvaluator(
                hdf5_read_path=hdf5_read_path,
                camera_hw=(128, 128),
                env_seed=seed,
                test_cnt=cfg.task.env_runner.n_test,
                test_start_seed=cfg.task.env_runner.test_start_seed,
                test_max_steps=cfg.task.env_runner.max_steps,
                policy_url=cfg.task.env_runner.policy_url,
                send_per_frames=cfg.task.env_runner.send_per_frames,
                phy_keywords=(
                    "cabinet", "drawer", "door", "basket", "object", "tool", "microwave",
                    # "robot",
                    "gripper",
                    "bowl",
                    "book", "mug",
                )
            )
        ]

        if "libero" in cfg.task.name:
            step_log = {}
            current_mean_scores = None
            all_mean_scores = []  # save mean scores of all envs
            total_envs = 10  # hard code for libero10
            for i, env_runner in enumerate(env_runners):
                # print stats will start from 2nd env
                if len(all_mean_scores) >= 1 and current_mean_scores is not None:
                    print(current_mean_scores)

                    remaining_envs = total_envs - len(all_mean_scores)
                    current_sum = sum(all_mean_scores)
                    estimated_max = (current_sum + remaining_envs * 1.0) / total_envs
                    estimated_min = (current_sum + remaining_envs * 0.5) / total_envs

                    print(f"\n=== Env {i}/10 finished ===")
                    print(f"current mean_scores: {current_mean_scores}")
                    print(f"current avg mean_score: {np.mean(all_mean_scores):.3f}")
                    print(f"estimate max mean_score: {estimated_max:.3f} (assuming all future success)")
                    print(f"estimate min mean_score: {estimated_min:.3f} (assuming all future half)")

                # runner_log = env_runner.run(policy)
                env_runner.init_socket()
                env_runner.send_reset()
                runner_log = env_runner.run_eval(
                    device="cuda",
                    load_cameras=LOAD_CAMERAS,
                    close_online=close_online,
                    friction_scale=friction_scale,
                    gz_scale=gz_scale,
                    solref_scale=solref_scale,
                    darken=visual_darken,
                    color_light=visual_color,
                    texture=visual_texture,
                    camera_shift=visual_viewshift,
                    vis_rollout_video=vis,
                )
                step_log.update(runner_log)
                print(runner_log)

                # extract current mean_score
                current_mean_scores = [
                    v for k, v in runner_log.items()
                    if k.endswith("_mean_score") and "test/" in k
                ]
                all_mean_scores.extend(current_mean_scores)

            assert "test_mean_score" not in step_log
            all_test_mean_score = {
                k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
            }
            step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

            runner_log = step_log
        else:
            env_runner = env_runners
            runner_log = env_runner.run(policy)

        # dump log to json
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value

        return json_log

    # Decide whether to run all tasks or a single task
    if task_id == "all":
        print("\n" + "="*80)
        print("Running evaluation for ALL tasks")
        print("="*80 + "\n")

        all_task_results = {}  # Store {task_id: test_mean_score}

        for idx, current_task_id in enumerate(HDF5_MAP.keys()):
            print(f"\n{'='*80}")
            print(f"[{idx+1}/{len(HDF5_MAP)}] Evaluating task: {current_task_id}")
            print(f"{'='*80}\n")

            json_log = run_single_task(
                current_task_id, cfg, seed, close_online,
                friction_scale, gz_scale, solref_scale,
                visual_darken, visual_color, visual_texture, visual_viewshift,
                vis
            )

            # Extract test_mean_score
            test_mean_score = json_log.get("test_mean_score", 0.0)
            all_task_results[current_task_id] = test_mean_score

            # Print result for this task
            print(f"\n{'='*80}")
            print(f"Task: {current_task_id}")
            print(f"test_mean_score: {test_mean_score:.4f}")
            print(f"{'='*80}\n")

            # Save individual task log
            out_path = os.path.join(output_dir, f'eval_log_{current_task_id}.json')
            print(f"Saving log to {out_path}")
            json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

        # Print final summary of all tasks
        print("\n" + "="*80)
        print("FINAL RESULTS - ALL TASKS")
        print("="*80)
        for current_task_id, score in all_task_results.items():
            print(f"{current_task_id}: {score:.4f}")
        print("="*80)
        overall_mean = np.mean(list(all_task_results.values()))
        print(f"Overall Average Success Rate: {overall_mean:.4f}")
        print("="*80 + "\n")

        # Save summary to JSON
        summary_path = os.path.join(output_dir, 'eval_log_all_summary.json')
        summary_data = {
            "all_task_results": all_task_results,
            "overall_mean_score": overall_mean
        }
        json.dump(summary_data, open(summary_path, "w"), indent=2, sort_keys=True)
        print(f"Saved summary to {summary_path}")

    else:
        # Run single task
        assert task_id in HDF5_MAP, f"task_id {task_id} not found in HDF5_MAP"

        print(f"\n{'='*80}")
        print(f"Running evaluation for task: {task_id}")
        print(f"{'='*80}\n")

        json_log = run_single_task(
            task_id, cfg, seed, close_online,
            friction_scale, gz_scale, solref_scale,
            visual_darken, visual_color, visual_texture, visual_viewshift,
            vis
        )

        for k, v in json_log.items():
            print(k, v)

        # out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
        out_path = os.path.join(output_dir, f'eval_log_{task_id}.json')
        print("Saving log to %s" % out_path)
        json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

        # Print final result for this task
        test_mean_score = json_log.get("test_mean_score", 0.0)
        domain_shift_tags = []
        if visual_color:
            domain_shift_tags.append("vc")
        if visual_texture:
            domain_shift_tags.append("vt")
        if visual_viewshift:
            domain_shift_tags.append("vv")
        if visual_darken:
            domain_shift_tags.append("vd")
        domain_shift_str = "+".join(domain_shift_tags) if domain_shift_tags else "none"
        print(f"\n{'='*80}")
        print(f"Task: {task_id}")
        print(f"Domain shift: {domain_shift_str}")
        print(f"test_mean_score: {test_mean_score:.4f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # click does not cleanly support `-vv` as a distinct flag while `-v` is already used.
    # We normalize user input here so both `-vv` and `--visual_viewshift` are supported.
    sys.argv = ["--visual_viewshift" if arg == "-vv" else arg for arg in sys.argv]
    main()
