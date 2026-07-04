import os
import re
import numpy as np
import requests
from typing import Tuple, Optional, List, Dict, Any, Union, Sequence
from tqdm import tqdm

import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero.force.modules import WrenchObsWrapper
from libero.force.physics_helper import PhysicsHelper
from libero.force.visual_helper import VisualHelper
from libero.force.hdf5_replayer import HDF5Replayer
from libero.force.hdf5_to_task import get_task_from_hdf5
from libero.force.utils import dict_apply

from robokit.connects.protocols import StepRequestFromPolicy, StepRequestFromEvaluator
from robokit.data_manager.utils_multiview import cat_multiview_video_with_another
from robokit.debug_utils.images import concatenate_rgb_images, save_frames_as_video, plot_force_sensor_wrt_time
from robokit.debug_utils.printer import print_batch


class SingleForceEnv:
    def __init__(
            self,
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = "/home/geyuan/code/LIBERO-FT/",
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool"),
    ):
        self.hdf5_read_path = hdf5_read_path
        self.camera_hw = camera_hw
        self.env_seed = env_seed

        self.step_cnt = 0

        # 0) Create env
        assert (hdf5_read_path is not None or
                bddl_path is not None is not None), \
            "hdf5_read_path or bddl_path must be provided."
        if hdf5_read_path is not None:
            task_info = get_task_from_hdf5(hdf5_read_path)
            '''
            suite_name: libero_90
            task_id: 23
            task_name: KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer
            task_language: close the bottom drawer of the cabinet and open the top drawer
            bddl_file_from_h5: libero/libero/bddl_files/libero_90/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl
            '''
            task_suite_name = task_info["suite_name"]
            task_id = task_info["task_id"]
            task_name = task_info["task_name"]
            task_language = task_info["task_language"]
            task_bddl_file_path = task_info["bddl_file_from_h5"]
            task_env_kwargs = task_info["env_kwargs"]

            self.bddl_path = os.path.join(libero_code_root, task_bddl_file_path)
            self.task_name = task_name
            self.task_language = task_language
            self.hdf5_env_kwargs = task_env_kwargs
            self.hdf5_env_kwargs.update({
                "bddl_file_name": self.bddl_path,
                "camera_heights": self.camera_hw[0],
                "camera_widths": self.camera_hw[1],
            })
            if "controller_configs" in self.hdf5_env_kwargs:
                del self.hdf5_env_kwargs["controller_configs"]

            task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
            task = task_suite.get_task(task_id)
            self.init_states = task_suite.get_task_init_states(task_id)
            # Note: can be used to set: env.set_init_state(init_states[init_state_id])
            print("[DEBUG] loaded init_states cnt:", len(self.init_states))

        elif bddl_path is not None:
            self.bddl_path = bddl_path
        elif benchmark_task_suite_name is not None:
            print("[Warning] Using benchmark_task_suite_name to get BDDL path.")
            task_suite = benchmark.get_benchmark_dict()[benchmark_task_suite_name]()
            task = task_suite.get_task(0)  # just to get bddl path
            init_states = task_suite.get_task_init_states(0)  # for benchmarking purpose, we fix the set of initial states
            init_state_id = 0
            self.bddl_path = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        self.init_env()

        # 1) PhysicsHelper for dynamics randomization
        self.physics_helper = PhysicsHelper(
            self.env.env.sim.model,
            object_keywords=phy_keywords,
            random_seed=self.env_seed,
        )
        self.visual_helper = VisualHelper(
            self.env.env.sim.model,
            random_seed=self.env_seed,
        )
        # Note: usage: apply_dynamics_shift; restore_original_params;

        print("[Info] SingleForceEnv initialized with BDDL:", self.bddl_path)

    def init_env(self):
        base_env = OffScreenRenderEnv(
            **self.hdf5_env_kwargs
        )
        self.env = WrenchObsWrapper(
            base_env,
            force_sensor="gripper0_force_ee",
            torque_sensor="gripper0_torque_ee",
        )
        self.reset_env()

    def close_env(self):
        self.env.close()

    def reset_env(self, env_seed: Optional[int] = None):
        self.env_seed = env_seed if env_seed is not None else self.env_seed
        self.env.seed(self.env_seed)
        self.env.reset()
        self.step_cnt = 0
        if hasattr(self, "visual_helper"):
            self.visual_helper.clear_dynamic_light_base()
            self.visual_helper.clear_camera_shift_base()

    def reset_to_benchmark_init(self, init_state_id=0, settle_steps=5, env_seed: Optional[int] = None):
        self.reset_env(env_seed=env_seed)
        self.env.set_init_state(self.init_states[init_state_id % len(self.init_states)])
        assert settle_steps > 0, "settle_steps must be greater than 0."
        for _ in range(settle_steps):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))
        return obs

    def set_benchmark_init_without_reset(self, init_state_id=0, settle_steps=5):
        self.step_cnt = 0
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()
        self.env.set_init_state(self.init_states[init_state_id % len(self.init_states)])
        if hasattr(self.env, "build_reader_if_needed"):
            self.env.build_reader_if_needed()
        assert settle_steps > 0, "settle_steps must be greater than 0."
        for _ in range(settle_steps):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))
        return obs

    def step(self, action_D: np.ndarray):
        obs, reward, done, info = self.env.step(action_D)
        self.step_cnt += 1
        success = self.check_success()
        return obs, reward, done, {**info, "success": success}

    def check_success(self):
        return bool(self.env.env.check_success())

    def regenerate_obs_from_current_state(self):
        sim_state = self.env.env.get_sim_state()
        obs = self.env.env.regenerate_obs_from_state(sim_state)
        obs["wrench_ee"] = self.env.read_wrench_from_sim()
        return obs

    def apply_physics_shift(
            self,
            frictionloss_scale: Tuple[float, float] = (0.5, 2.0),
            damping_scale: Tuple[float, float] = (0.5, 2.0),
            sliding_friction_scale: Tuple[float, float] = (0.7, 1.3),
            gravity_z_range: Tuple[float, float] = (-10.3, -9.3),
            tweak_solref_range: Tuple[float, float] = (0.8, 1.2),
            rng: np.random.RandomState = None,
    ):
        self.physics_helper.apply_dynamics_shift(
            model=self.env.env.sim.model,  # NOTE: don't know why, but we have to resend the sim.model here
            frictionloss_scale=frictionloss_scale,
            damping_scale=damping_scale,
            sliding_friction_scale=sliding_friction_scale,
            gravity_z_range=gravity_z_range,
            tweak_solref_range=tweak_solref_range,
            rng=rng,
        )
        self.env.env.sim.forward()
        print(f"[Info] Applied physics shift. gravity={self.env.env.sim.model.opt.gravity}, "
              f"friction/damping/sliding={damping_scale}, solref={tweak_solref_range}.")

    def restore_physics(self):
        self.physics_helper.restore_original_params()

    def apply_visual_shift(
            self,
            table_center_xyz: tuple = None,
            darken: bool = False,
            color_light: bool = False,
            texture: bool = False,
            texture_dir: str = "images/dtd",
            camera_shift: bool = False,
            rng: np.random.RandomState = None,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        if (not darken) and (not color_light) and (not texture) and (not camera_shift):
            return {"applied": False, "reset_applied": False}
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()

        texture_info = None
        flashlight_info = None
        color_info = None
        camera_info = None

        if texture:
            texture_info = self.visual_helper.apply_table_texture_via_xml_wrapper(
                texture_dir=texture_dir,
                env=self.env,
                preferred_material_names=("desk", "table_texture", "table_mat"),
                apply_to_all_keyword_textures=True,
                reset_env=True,
                rng=rng,
                verbose=verbose,
            )
            # reset_from_xml_string rebuilds sim/model, refresh helper model refs.
            self.physics_helper.model = self.env.env.sim.model
            if getattr(self.env, "reader", None) is not None:
                self.env.reader = None

        if darken or color_light:
            light_info = self.visual_helper.apply_color_and_darken(
                model=self.env.env.sim.model,
                rng=rng,
                darken=darken,
                color_light=color_light,
                table_center_xyz=table_center_xyz if table_center_xyz is not None else (0.0, 0.0, 0.9),
                table_half_size_xy=(0.05, 0.10),
                preserve_luminance=True,
                verbose=verbose,
            )
            flashlight_info = light_info.get("flashlight_info")
            color_info = light_info.get("color_info")

        if camera_shift:
            camera_info = self.visual_helper.apply_camera_shift(
                model=self.env.env.sim.model,
                rng=rng,
                verbose=verbose,
            )

        self.env.env.sim.forward()
        return {
            "applied": True,
            "reset_applied": bool(texture_info.get("reset_applied", False)) if texture_info is not None else False,
            "texture_info": texture_info,
            "flashlight_info": flashlight_info,
            "color_info": color_info,
            "camera_info": camera_info,
        }

    def restore_visual(self):
        self.visual_helper.restore_original_params(model=self.env.env.sim.model)
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()


class LiberoForceSocketEvaluator:
    def __init__(
            self,
            # SingleForceEnv params
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = "/home/geyuan/code/LIBERO-FT/",
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool", "microwave"),
            # Eval params
            test_cnt: int = 20,
            test_start_seed: int = 10000,
            test_max_steps: int = 400,
            # Socket related
            policy_url='localhost:6006',
            send_per_frames: int = 1,
    ):
        self.env = SingleForceEnv(
            hdf5_read_path=hdf5_read_path,
            camera_hw=camera_hw,
            env_seed=env_seed,
            libero_code_root=libero_code_root,
            bddl_path=bddl_path,
            benchmark_task_suite_name=benchmark_task_suite_name,
            phy_keywords=phy_keywords,
        )
        self.language_goal = self.env.task_name.replace("_", " ")

        # Eval related
        self.test_cnt = test_cnt
        self.test_start_seed = test_start_seed
        self.current_test_id = 0
        self.test_max_steps = test_max_steps

        # Socket related
        self.policy_url = policy_url
        self.send_per_frames = send_per_frames
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.http_session = requests.Session()
        self.task_instruction = None
        self.cache_actions_B_T_D = None

        self.frame_buffer = {
            'primary_rgb': [],
            'gripper_rgb': [],
            'joint_state': [],
        }  # some policies require 2 or more observation frames

    def init_socket(self, task_instruction: str = None) -> int:
        resp = self.http_session.get(f"{self.policy_url}/init")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.send_per_frames = max(max_cache_action, 1)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.cache_frames_B_H_W_C = None

        if task_instruction is not None:
            self.task_instruction = task_instruction
        else:
            self.task_instruction = self.language_goal.lower()
            self.task_instruction = re.sub(r'(\d)(\s)', r' \1 ', self.task_instruction)
            self.task_instruction = self.task_instruction.replace("_", " ")
        print("[LiberoForceSocketEvaluator] Connected to policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames}",
              f"task_instruction={self.task_instruction}")
        return max_cache_action

    def send_reset(self, task_instruction: str = None) -> int:
        assert self.task_instruction is not None, "Please call init_socket first."
        self.task_instruction = task_instruction if task_instruction is not None else self.task_instruction

        resp = self.http_session.get(f"{self.policy_url}/reset")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.send_per_frames = max(max_cache_action, 1)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.cache_frames_B_H_W_C = None

        print("[LiberoForceSocketEvaluator] Reset policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames},"
              f" task_instruction={self.task_instruction[:50]}")
        return max_cache_action

    def send_obs_and_get_action(self,
                                image_B_T_H_W_C: np.ndarray,
                                stage_flag: int,
                                robot_states_B_T_D: np.ndarray = None,
                                ) -> np.ndarray:
        assert self.task_instruction is not None, "Please call init_socket first."

        if self.send_cnt % self.send_per_frames == 0:
            video_B_v2_H_W_C = image_B_T_H_W_C  # (B,Ts,H,W,3) uint8
            request_to_policy = StepRequestFromEvaluator.encode_from_raw(
                instruction=self.task_instruction,
                stage_flag=stage_flag,
                gt_video=video_B_v2_H_W_C,  # (B,Ts,H,W,3) uint8
                tcp_state=robot_states_B_T_D,  # (B,Ts,D) float32, not used
                num_camera_views=2,
                max_cache_action=self.send_per_frames,
            )

            # print(f"[LiberoForceSocketEvaluator] sending frames, i={self.send_cnt}, per={self.send_per_frames}, "
            #       f"i%per={self.send_cnt % self.send_per_frames}, video_B_v2_H_W_C:{video_B_v2_H_W_C.shape}, ")
            sending_dict = request_to_policy.model_dump(mode="json")
            response = self.http_session.post(
                f"{self.policy_url}/step",
                json=sending_dict
            )
            response.raise_for_status()

            response = response.json()
            raw_actions = StepRequestFromPolicy(action=response["action"]).decode_to_raw()["action"]  # (B,H,7)

            # assert raw_actions.shape[0] == self.send_per_frames
            self.cache_actions_B_T_D = raw_actions
            self.cache_frames_B_H_W_C = None
        else:  # don't send anything, to save network bandwidth
            pass

        ret_actions = self.cache_actions_B_T_D[:, self.send_cnt % self.send_per_frames]  # (B,D)
        ret_actions_B_1_D = ret_actions[:, None]  # [B,D] -> (B,1,D)

        self.send_cnt += 1
        return ret_actions_B_1_D

    def run_eval(self,
                 device: Union[torch.device, str] = "cuda",
                 vis_pred_video=False,
                 load_cameras: List[str] = None,
                 close_online: bool = True,
                 friction_scale: float = 1.,
                 gz_scale: float = 1.,
                 solref_scale: float = 1.,
                 table_center_xyz: tuple = None,
                 darken: bool = False,
                 color_light: bool = False,
                 texture: bool = False,
                 texture_dir: str = "images/dtd/",
                 camera_shift: bool = False,
                 vis_rollout_video: bool = False,
                 **kwargs
    ):
        if kwargs.get("selected_light_id", None) is not None:
            darken = True
        use_dynamic_color_light = bool(color_light)
        domain_shift_tags = []
        if color_light:
            domain_shift_tags.append("vc")
        if texture:
            domain_shift_tags.append("vt")
        if camera_shift:
            domain_shift_tags.append("vv")
        if darken:
            domain_shift_tags.append("vd")
        domain_shift_str = "+".join(domain_shift_tags) if domain_shift_tags else "none"

        env = self.env
        env.reset_env()

        self.send_reset()
        stage_flag = 0  # 0:cold start, 1:hot start

        # print("env_runner: ", self.language_goal)
        eval_results = {
            "success_list": [],
        }
        test_seed = self.test_start_seed
        for test_idx in range(self.test_cnt):
            # a) init envs
            ## NOTE: some policies' reset requires a long time, so we only reset env once
            ## You need to make sure this is correct for your policy.
            # self.send_reset()
            self.current_test_id = test_idx
            _ = env.reset_to_benchmark_init(
                init_state_id=test_idx,
                env_seed=test_seed
            )

            # add visual randomization here if needed
            visual_info = None
            if darken or texture or camera_shift:
                visual_info = env.apply_visual_shift(
                    table_center_xyz=table_center_xyz,
                    darken=darken,
                    color_light=False,
                    texture=texture,
                    texture_dir=texture_dir,
                    camera_shift=camera_shift,
                    rng=np.random.RandomState(test_seed),
                    verbose=False,
                )
                # texture reset rebuilds env internals; restore benchmark init state
                # without calling env.reset() again (which may overwrite patched XML).
                if visual_info.get("reset_applied", False):
                    env.set_benchmark_init_without_reset(
                        init_state_id=test_idx,
                    )

            # add physics randomization here if needed
            if friction_scale != 1.0 or gz_scale != 1.0 or solref_scale != 1.0:
                if gz_scale > 1.0:
                    possible_gz_range = (-9.81 * (1 + gz_scale), -9.81)
                elif gz_scale < 1.0:
                    possible_gz_range = (-9.81 * (gz_scale ** 2), -9.81 * gz_scale)
                else:
                    possible_gz_range = None

                if friction_scale > 1.0:
                    possible_friction_range = (friction_scale, friction_scale ** 2)
                elif friction_scale < 1.0:
                    possible_friction_range = (friction_scale ** 2, friction_scale)
                else:
                    possible_friction_range = None

                if solref_scale > 1.0:
                    possible_solref_range = (solref_scale, solref_scale ** 2)
                elif solref_scale < 1.0:
                    possible_solref_range = (solref_scale ** 2, solref_scale)
                else:
                    possible_solref_range = None

                env.apply_physics_shift(
                    frictionloss_scale=possible_friction_range,
                    damping_scale=possible_friction_range,
                    sliding_friction_scale=possible_friction_range,
                    gravity_z_range=possible_gz_range,
                    tweak_solref_range=possible_solref_range,
                    rng=np.random.RandomState(test_seed),
                )
            if use_dynamic_color_light:
                env.visual_helper.capture_dynamic_light_base(
                    model=env.env.sim.model,
                )
                env.visual_helper.apply_dynamic_rainbow_light(
                    model=env.env.sim.model,
                    # step_idx=env.step_cnt,
                    step_idx=1,
                )
                env.env.sim.forward()

            print(
                "[DEBUG] Visual (ep="
                f"{test_idx+1}/{self.test_cnt}, seed={test_seed}): "
                f"darken(vd)={bool(darken)} "
                f"texture(vt)={bool(texture)} "
                f"color(vc)={bool(use_dynamic_color_light)} "
                f"view(vv)={bool(camera_shift)}"
            )
            obs = env.regenerate_obs_from_current_state()

            env_name = env.task_name
            cur_sr = float(np.array(eval_results['success_list']).mean()) * 100. if len(
                eval_results['success_list']) > 0 else 0.0
            pbar = tqdm(
                total=self.test_max_steps,
                desc=f"Eval {env_name[:20]} ({test_idx+1}/{self.test_cnt}) "
                     f"(SR={cur_sr:.2f}%)",
                leave=False,
                # mininterval=self.tqdm_interval_sec,  # e.g, 5.0 seconds
            )
            vis_frames = []
            vis_forces = []
            done = False
            success = False

            # b) start rollout
            while not done:
                np_obs_dict = dict(obs)
                '''
                [DEBUG] 1st obs::                                    
                --robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2077, max=1.1671, dtype=float64
                --robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0367, max=0.9993, dtype=float64
                --robot0_gripper_qpos, <class 'numpy.ndarray'>, shape=(2,), min=-0.0207, max=0.0207, dtype=float64
                --robot0_eye_in_hand_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=240.0000, dtype=uint8
                --agentview_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=255.0000, dtype=uint8                                                                                                                                                                         
                --wrench_ee, <class 'numpy.ndarray'>, shape=(6,), min=-5.0506, max=0.1079, dtype=float32
                '''
                vis_frames.append(
                    concatenate_rgb_images(
                        np.flipud(np_obs_dict["agentview_image"]),
                        np.flipud(np_obs_dict["robot0_eye_in_hand_image"]),
                        vertical=True,
                    )
                )
                vis_forces.append(obs["wrench_ee"])

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )
                # add B,T dims
                for key in obs_dict:
                    obs_dict[key] = obs_dict[key].unsqueeze(0).unsqueeze(0)  # (H,W,C) -> (1,1,H,W,C)
                # print_batch("[DEBUG] transferred obs:", np_obs_dict)

                # not used
                if load_cameras is None:
                    load_cameras = ["agentview_image", "robot0_eye_in_hand_image"]
                latest_obs_B_T_H_W_C_tensors = []
                for camera in load_cameras:
                    latest_obs_B_T_H_W_C_tensors.append(
                        obs_dict[camera]
                    )  # (B,Ts,H,W,C)
                latest_obs_B_T_H_W_C = (torch.cat(
                    latest_obs_B_T_H_W_C_tensors, dim=1
                ).cpu().numpy()).astype(np.uint8)
                # (B,V*Ts,H,W,C) uint8 [0,255] WARN: already uint8 in [0,255], so no need to *255.

                # latest_robot_states_B_T_D = torch.cat([
                #     obs_dict["robot0_joint_pos"],
                #     obs_dict["robot0_gripper_qpos"][:, :, :1],  # only use one finger state
                # ], dim=2).cpu().numpy()  # (B,T,8) float32
                ee_pos = obs_dict["robot0_eef_pos"].cpu().numpy()  # (B,T,3)
                ee_ori = HDF5Replayer.quat_xyzw_to_rotvec(obs_dict["robot0_eef_quat"].cpu().numpy().squeeze())  # (3)
                ee_ori = ee_ori[None, None, :]  # (B,Ts,3)
                latest_robot_states_B_T_D = np.concatenate((ee_pos, ee_ori), axis=2)  # (B,T,6)
                latest_force_B_T_D = obs_dict["wrench_ee"].cpu().numpy()  # (B,Ts,6) float32
                # NOTE: need to be concatenated with force data
                latest_robot_states_B_T_D = np.concatenate(
                    (latest_robot_states_B_T_D, latest_force_B_T_D), axis=2
                )  # (B,Ts,6+6)

                cur_primary_rgb = np_obs_dict["agentview_image"]  # (H,W,C) uint8
                cur_gripper_rgb = np_obs_dict["robot0_eye_in_hand_image"]  # (H,W,C) uint8
                cur_joint_state = latest_robot_states_B_T_D[0, 0]  # (6+6) with force

                # work like a queue, only keep the latest `buffer_size` frames
                buffer_size = self.send_per_frames
                for key, val in zip(
                        ['primary_rgb', 'gripper_rgb', 'joint_state'],
                        [cur_primary_rgb, cur_gripper_rgb, cur_joint_state]
                ):
                    if len(self.frame_buffer[key]) >= buffer_size:
                        self.frame_buffer[key].pop(0)
                    self.frame_buffer[key].append(val)

                # construct output, add pad before the data (len = buffer_size)
                def pad_to_buffer(data_list, pad_with):
                    """
                    :return: (buffer_size,D) np.ndarray or List
                    """
                    padded = [pad_with for _ in range(buffer_size - len(data_list))] + data_list
                    return np.stack(padded) if isinstance(pad_with, np.ndarray) else padded

                cur_primary_rgb = pad_to_buffer(
                    self.frame_buffer['primary_rgb'],
                    self.frame_buffer['primary_rgb'][0]
                )  # (T,H,W,C)
                cur_gripper_rgb = pad_to_buffer(
                    self.frame_buffer['gripper_rgb'],
                    self.frame_buffer['gripper_rgb'][0]
                )
                cur_joint_state = pad_to_buffer(
                    self.frame_buffer['joint_state'],
                    self.frame_buffer['joint_state'][0]
                )  # (T,D)
                gt_video = cat_multiview_video_with_another(
                    video_A_B_VT_H_W_C=cur_primary_rgb[None, :],  # (1,Ts,H,W,C)
                    video_B_B_VT_H_W_C=cur_gripper_rgb[None, :],  # (1,Ts,H,W,C)
                    sample_n_views=1,
                )  # (1,V*Ts,H,W,C)
                # print("[DEBUG] gt_video:", gt_video.shape, gt_video.dtype, gt_video.min(), gt_video.max())
                # print("[DEBUG] cur_joint_state[None, :]:", cur_joint_state[None, :].shape, cur_joint_state.dtype,)

                # run policy
                with torch.no_grad():
                    out_action_B_1_D = self.send_obs_and_get_action(
                        image_B_T_H_W_C=gt_video,  # (B,V*Ts,H,W,C) uint8
                        stage_flag=stage_flag,
                        robot_states_B_T_D=cur_joint_state[None, :],  # (B,Ts,6+6)
                    )
                    # print("[DEBUG] out_action_B_1_D:", out_action_B_1_D.shape, out_action_B_1_D.dtype, out_action_B_1_D.min(), out_action_B_1_D.max())
                    action_dict = {
                        "action": torch.from_numpy(out_action_B_1_D).to(device=device),  # (B,1,2)
                    }

                    if close_online:
                        pass
                    else:
                        stage_flag = 1  # after first step, all are hot start

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]  # (1, T, Da)
                # print("[DEBUG] action:", action, action.shape, action.dtype, action.min(), action.max())
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action: np.ndarray = action[0, 0]  # (Da,)
                # if self.abs_action:
                #     env_action = self.undo_transform_action(action)
                if use_dynamic_color_light:
                    env.visual_helper.apply_dynamic_rainbow_light(
                        model=env.env.sim.model,
                        step_idx=env.step_cnt,
                    )
                    env.env.sim.forward()

                obs, reward, done, info = env.step(env_action)

                # for i in range(len(reward)):
                #     if reward[i] == 1:
                #         done[i] = True
                #
                # done = np.all(done)
                if reward == 1:
                    done = True
                    success = True

                if info["success"]:
                    done = True
                    success = True

                if env.step_cnt >= self.test_max_steps:
                    done = True  # failure due to timeout
                    success = False

                # # past_action = action
                # past_action_list.append(action)
                # if len(past_action_list) > 2:
                #     past_action_list.pop(0)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            vis_force_T_D = np.array(vis_forces)  # (T,6)
            # NOTE: add a setting here
            # vis_force_frames, *_ = plot_force_sensor_wrt_time(vis_force_T_D, only_last_frame=True)
            # vis_frames = [concatenate_rgb_images(f, vis_force_frames[-1], resize_ratio=2.) for f in vis_frames]
            if vis_rollout_video:
                save_frames_as_video(
                    vis_frames,
                    save_path=f"/home/geyuan/code/LIBERO-FT/debug/libero_rollout/"
                              f"shift={domain_shift_str}_merge_liberoft_eval{test_idx:02d}_suc={success:1d}.mp4"
                )
            eval_results["success_list"].append(float(success))
            eval_results[f"test/seed{test_seed}_mean_score"] = float(success)

            # # collect data for this round
            # all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call("get_attr", "reward")[
            #     this_local_slice
            # ]

            if friction_scale != 1.0 or gz_scale != 1.0 or solref_scale != 1.0:
                env.restore_physics()
            test_seed += 1

        # clear out video buffer
        _ = env.reset_env()
        env.close_env()

        return eval_results

        # # log
        # max_rewards = collections.defaultdict(list)
        # log_data = dict()
        # # results reported in the paper are generated using the commented out line below
        # # which will only report and average metrics from first n_envs initial condition and seeds
        # # fortunately this won't invalidate our conclusion since
        # # 1. This bug only affects the variance of metrics, not their mean
        # # 2. All baseline methods are evaluated using the same code
        # # to completely reproduce reported numbers, uncomment this line:
        # # for i in range(len(self.env_fns)):
        # # and comment out this line
        # for i in range(n_inits):
        #     seed = self.env_seeds[i]
        #     prefix = self.env_prefixs[i]
        #     max_reward = np.max(all_rewards[i])
        #     max_rewards[prefix].append(max_reward)
        #     log_data[prefix + f"sim_max_reward_{seed}"] = max_reward
        #
        #     # visualize sim
        #     video_path = all_video_paths[i]
        #     if video_path is not None:
        #         sim_video = wandb.Video(video_path)
        #         log_data[prefix + f"sim_video_{seed}"] = sim_video

        # # log aggregate metrics
        # for prefix, value in max_rewards.items():
        #     name = prefix + "mean_score"
        #     value = np.mean(value)
        #     log_data[name] = value

        # return log_data


def eval_one_episode(env, policy, horizon=400, record_images=False):
    """
    Return: success(bool), episode_data(dict)
    episode_data: includes wrench norm、reward、images...
    """
    images = []
    wrench_norms = []

    obs = env.reset()
    if record_images:
        images.append(obs["agentview_image"].copy())

    for t in range(horizon):
        action = policy(obs)  # (7,)
        obs, reward, done, info = env.step(action)

        if record_images:
            images.append(obs["agentview_image"].copy())

        if "wrench_ee" in obs:
            wrench_norms.append(float(np.linalg.norm(obs["wrench_ee"][:3])))

        if hasattr(env, "env") and hasattr(env.env, "check_success"):
            if env.env.check_success():
                return True, {"images": images, "wrench_norms": wrench_norms}

    # check again at the end of episode, in case some policies only set success=True at the end
    success = env.env.check_success() if (hasattr(env, "env") and hasattr(env.env, "check_success")) else False
    return bool(success), {"images": images, "wrench_norms": wrench_norms}


def evaluate_task_suite(
    task_suite_name="libero_10",
    task_id=0,
    num_episodes=50,
    seed=0,
    use_wrench=False,
    record_images=False,
    horizon=400,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    print("[task]", task.name)
    print("[lang]", task.language)
    print("[bddl]", task_bddl_file)

    init_states = task_suite.get_task_init_states(task_id)  # i.e. 50 for LIBERO
    assert len(init_states) >= num_episodes, f"init_states={len(init_states)} < num_episodes={num_episodes}"

    env = make_env(task_bddl_file, use_wrench=use_wrench)
    env.seed(seed)

    successes = []
    for ep in tqdm(range(num_episodes), desc="eval"):
        obs = env.reset()

        # fix init_state like LIBERO benchmark
        env.set_init_state(init_states[ep])

        # stabilize the env for a few steps, in case some policies require stable initial observations
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        success, ep_data = eval_one_episode(
            env=env,
            policy=policy,
            horizon=horizon,
            record_images=record_images,
        )
        successes.append(int(success))

        # DEBUG: save video for the first episode if it fails, to check if the env and policy are working correctly
        # if record_images and (not success) and ep == 0:
        #     save_video(ep_data["images"], "debug.mp4", fps=10)

    env.close()

    sr = float(np.mean(successes))
    print(f"[result] success_rate={sr:.3f} ({sum(successes)}/{len(successes)})")
    return sr


# -------------------------
# a simple dummy policy for debugging
# -------------------------
class ZeroPolicy:
    def __call__(self, obs):
        return np.zeros(7, dtype=np.float32)


if __name__ == "__main__":
    policy = ZeroPolicy()
    evaluate_task_suite(
        task_suite_name="libero_10",
        task_id=0,
        num_episodes=50,
        seed=0,
        use_wrench=True,        # if add wrench into the observation
        record_images=False,    # if save videos
        horizon=400,
    )
