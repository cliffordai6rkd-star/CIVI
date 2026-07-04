import os
import h5py
import json
import re
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional, List
from PIL import Image
from tqdm import tqdm

from robosuite.utils import transform_utils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero.force.modules import MujocoSensorReader, WrenchObsWrapper
from libero.force.physics_helper import PhysicsHelper
from libero.force.visual_helper import VisualHelper

from robokit.debug_utils.printer import print_batch
from robokit.debug_utils.images import (
    plot_action_wrt_time, save_frames_as_video, plot_force_sensor_wrt_time, concatenate_rgb_images
)


class HDF5Replayer:
    def __init__(
            self,
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = "/home/geyuan/code/LIBERO-FT/",
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            retry_obs_source: str = "hdf5",
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool"),
    ):
        assert retry_obs_source in ("hdf5", "env"), \
            f"retry_obs_source must be 'hdf5' or 'env', got {retry_obs_source}."
        self.hdf5_read_path = hdf5_read_path
        self.camera_hw = camera_hw
        self.env_seed = env_seed
        self.retry_obs_source = retry_obs_source

        # 0) Create env
        assert (hdf5_read_path is not None or
                bddl_path is not None is not None), \
            "hdf5_read_path or bddl_path must be provided."
        if hdf5_read_path is not None:
            with h5py.File(hdf5_read_path, "r") as f:
                self.bddl_path = os.path.join(libero_code_root, f["data"].attrs["bddl_file_name"])
                hdf5_env_meta = json.loads(f["data"].attrs["env_args"])
                self.hdf5_env_kwargs = hdf5_env_meta["env_kwargs"]
                self.hdf5_env_kwargs.update({
                    "bddl_file_name": self.bddl_path,
                    "camera_heights": self.camera_hw[0],
                    "camera_widths": self.camera_hw[1],
                })
                if "controller_configs" in self.hdf5_env_kwargs:
                    del self.hdf5_env_kwargs["controller_configs"]
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
        )

        print("[Info] HDF5Replayer initialized with BDDL:", self.bddl_path)

    def init_env(self):
        base_env = OffScreenRenderEnv(
            **self.hdf5_env_kwargs
        )
        self.env = WrenchObsWrapper(
            base_env,
            force_sensor="gripper0_force_ee",
            torque_sensor="gripper0_torque_ee",
        )
        self.env.seed(self.env_seed)
        self.env.reset()

    def close_env(self):
        self.env.close()

    def apply_physics_shift(
            self,
            frictionloss_scale: Tuple[float, float] = (0.5, 2.0),
            damping_scale: Tuple[float, float] = (0.5, 2.0),
            sliding_friction_scale: Tuple[float, float] = (0.7, 1.3),
            gravity_z_range: Tuple[float, float] = (-10.3, -9.3),
            tweak_solref_range: Tuple[float, float] = (0.8, 1.2),
            rng: np.random.RandomState = None,
            verbose: bool = False,
    ):
        if frictionloss_scale is None and damping_scale is None and sliding_friction_scale is None \
                and gravity_z_range is None and tweak_solref_range is None:
            print(f"[Info] Force shift ranges are not provided, skipping applying physics shift.")
            return  # do nothing
        self.physics_helper.apply_dynamics_shift(
            model=self.env.env.sim.model,  # NOTE: don't know why, but we have to resend the sim.model here
            frictionloss_scale=frictionloss_scale,
            damping_scale=damping_scale,
            sliding_friction_scale=sliding_friction_scale,
            gravity_z_range=gravity_z_range,
            tweak_solref_range=tweak_solref_range,
            rng=rng,
            verbose=verbose,
        )
        self.env.env.sim.forward()
        if verbose:
            print(f"[Info] Applied physics shift. gravity={self.env.env.sim.model.opt.gravity}, "
                  f"friction/damping/sliding={damping_scale}, solref={tweak_solref_range}.")

    def restore_physics(self):
        self.physics_helper.restore_original_params()

    def regenerate_obs_from_current_state(self):
        sim_state = self.env.env.get_sim_state()
        obs = self.env.env.regenerate_obs_from_state(sim_state)
        obs["wrench_ee"] = self.env.read_wrench_from_sim()
        return obs

    def apply_visual_shift(
            self,
            table_center_xyz: tuple = None,
            darken: bool = False,
            color_light: bool = False,
            texture: bool = False,
            texture_dir: str = "images/dtd/",
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
            # reset_from_xml_string may rebuild sim/model; refresh helper model pointers.
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

    def vis_hdf5_contents(self, hdf5_path: str = None,
                          save_videos: bool = False,
                          save_video_prefix: str = "saved_replay_images",
                          ):
        hdf5_path = self.hdf5_read_path if hdf5_path is None else hdf5_path
        assert os.path.exists(hdf5_path), f"hdf5_path={hdf5_path} does not exist!"
        with h5py.File(hdf5_path, "r") as f:
            print("Top-level keys:", list(f.keys()))

            # recursively print all groups and datasets
            def print_h5(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"[DSET] {name} shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"[GRP ] {name}")

            f.visititems(print_h5)

            demos = sorted(f["data"].keys())  # ["demo_0", "demo_1", ...]
            all_w = []
            has_wrenches = False

            for dk in demos:
                if "wrenches" in f["data"][dk]:
                    has_wrenches = True
                    w = np.array(f["data"][dk]["wrenches"], dtype=np.float32)  # (T,6)
                    all_w.append(w)
                else:
                    break

            # save images
            if save_videos:
                demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))
                for iter_idx, demo_key in enumerate(tqdm(demo_keys, desc="Saving videos")):  # e.g., "demo_0", "demo_1", ...
                    obs = f["data"][demo_key]["obs"]
                    resized_agentview_images = [np.flipud(np.array(Image.fromarray(x).resize((256, 256)))) for x in
                                                obs["agentview_rgb"]]
                    resized_eyeinhand_images = [np.flipud(np.array(Image.fromarray(x).resize((256, 256)))) for x in
                                                obs["eye_in_hand_rgb"]]
                    resized_images = [
                        concatenate_rgb_images(h_img, eih_img, resize_ratio=1., vertical=True)
                        for h_img, eih_img in zip(resized_agentview_images, resized_eyeinhand_images)
                    ]
                    save_frames_as_video(
                        resized_images,f"{save_video_prefix}_{iter_idx:02d}.mp4", fps=20)

        if has_wrenches:
            W = np.concatenate(all_w, axis=0)  # (sum_T, 6)
            stats_w = self.summarize_array(W)

            F = W[:, 0:3]
            M = W[:, 3:6]
            stats_Fnorm = self.summarize_array(np.linalg.norm(F, axis=1, keepdims=True))  # (N,1)
            stats_Mnorm = self.summarize_array(np.linalg.norm(M, axis=1, keepdims=True))  # (N,1)
            stats_Wnorm = self.summarize_array(np.linalg.norm(W, axis=1, keepdims=True))  # (N,1)

            names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

            print("=== Per-dimension wrench stats over ALL demos ===")
            for i, n in enumerate(names):
                print(
                    f"{n:>2}  mean={stats_w['mean'][i]:9.3f}  "
                    f"max={stats_w['max'][i]:9.3f}  "
                    f"p01={stats_w['p01'][i]:9.3f}  "
                    f"p99={stats_w['p99'][i]:9.3f}"
                )

            print("\n=== Norm stats (scalar) ===")
            print(
                f"||F|| mean={stats_Fnorm['mean'][0]:.3f} max={stats_Fnorm['max'][0]:.3f} p01={stats_Fnorm['p01'][0]:.3f} p99={stats_Fnorm['p99'][0]:.3f}")
            print(
                f"||M|| mean={stats_Mnorm['mean'][0]:.3f} max={stats_Mnorm['max'][0]:.3f} p01={stats_Mnorm['p01'][0]:.3f} p99={stats_Mnorm['p99'][0]:.3f}")
            print(
                f"||W|| mean={stats_Wnorm['mean'][0]:.3f} max={stats_Wnorm['max'][0]:.3f} p01={stats_Wnorm['p01'][0]:.3f} p99={stats_Wnorm['p99'][0]:.3f}")

            Fn = np.linalg.norm(W[:, :3], axis=1)
            print("contact ratio (Fn>1N):", (Fn > 1).mean())
            print("contact ratio (Fn>10N):", (Fn > 10).mean())
            print("percentiles:", np.percentile(Fn, [50, 90, 95, 99, 99.5, 99.9]).astype(np.int64))

            # per_demo_max = []
            # for each demo:
            #     per_demo_max.append(np.linalg.norm(w[:, :3], axis=1).max())
            # print(np.percentile(per_demo_max, [50, 90, 95, 99]))

    def replay_hdf5_by_actions(
            self,
            hdf5_read_path: str = None,
            hdf5_save_path: str = "tmp_replayed_wrench.hdf5",
            video_save_dir: str = None,
            save_max_videos: int = 1,
            replay_seed: int = None,
            retry_with_reset: int = 5,
            # physical shift related
            friction_scale: float = 1.,
            gz_scale: float = 1.,
            solref_scale: float = 1.,
            # visual shift related
            table_center_xyz: tuple = None,
            darken: bool = False,
            color_light: bool = False,
            texture: bool = False,
            texture_dir: str = "images/dtd/",
            camera_shift: bool = False,
            demo_slices: Optional[Union[slice, str, Tuple[Union[slice, str], ...]]] = None,
    ):
        hdf5_read_path = self.hdf5_read_path if hdf5_read_path is None else hdf5_read_path
        assert os.path.exists(hdf5_read_path), f"hdf5_read_path={hdf5_read_path} does not exist!"
        if replay_seed is None:
            replay_seed = self.env_seed

        use_force_shift = (friction_scale != 1.) or (gz_scale != 1.) or (solref_scale != 1.)

        with h5py.File(hdf5_read_path, "r") as fin, h5py.File(hdf5_save_path, "w") as fout:
            # 复制全局 attrs（如果有）
            num_out_attrs = self.copy_attrs(fin, fout)
            print(f"Copied {num_out_attrs} global attributes from input to output HDF5")

            print("Demo keys:", list(fin["data"].keys()))
            fout.create_group("data")
            num_out_attrs = self.copy_attrs(fin["data"], fout["data"])
            print(f"Copied {num_out_attrs} 'data' group attributes from input to output HDF5")

            success_stats = {"total": 0}
            all_demo_keys = sorted(list(fin["data"].keys()), key=lambda x: int(x.split("_")[1]))
            demo_keys = self._select_demo_keys(all_demo_keys, demo_slices=demo_slices)
            if len(demo_keys) == 0:
                raise ValueError(
                    f"demo_slices={demo_slices} selected 0 demos out of {len(all_demo_keys)}. "
                    f"Available demo index range: [0:{len(all_demo_keys)})."
                )
            print(
                f"[Info] Selected {len(demo_keys)}/{len(all_demo_keys)} demos for replay "
                f"with demo_slices={demo_slices}."
            )
            if "num_demos" in fout["data"].attrs:
                fout["data"].attrs["num_demos"] = len(demo_keys)
            for iter_idx, demo_key in enumerate(  # e.g., "demo_0", "demo_1", ...
                    tqdm(demo_keys,"Replay by actions")
            ):
                # print("Processing demo group:", demo_key)
                g_in = fin["data"][demo_key]
                g_out = fout["data"].create_group(demo_key)
                self.copy_attrs(g_in, g_out)

                replay_results = self._replay_actions_and_save_hdf5(
                    g_in, g_out, demo_key, action_shift=1,
                    replay_seed=replay_seed,
                    retry_with_reset=retry_with_reset,
                    friction_scale=friction_scale,
                    gz_scale=gz_scale,
                    solref_scale=solref_scale,
                    table_center_xyz=table_center_xyz,
                    darken=darken,
                    color_light=color_light,
                    texture=texture,
                    texture_dir=texture_dir,
                    camera_shift=camera_shift,
                )
                success = replay_results["success"]  # bool
                wrenches = replay_results["wrenches"]  # (T,6)
                reset_every = replay_results["reset_every"]  # int
                vis_hdf5_images = replay_results["vis_hdf5_images"]  # list of (H,W,3)
                vis_replay_images = replay_results["vis_replay_images"]  # list of (H,W,3)
                num_tries = replay_results["num_tries"]

                if iter_idx < save_max_videos and video_save_dir is not None:  # only visualize first 10 demos
                    os.makedirs(video_save_dir, exist_ok=True)
                    vis_images = [
                        concatenate_rgb_images(hdf5_image, replay_image, resize_ratio=1.)
                        for hdf5_image, replay_image in zip(vis_hdf5_images, vis_replay_images)
                    ]
                    if use_force_shift:
                        force_frames, _, _ = plot_force_sensor_wrt_time(wrenches)
                        combined_vis_images = [
                            concatenate_rgb_images(img, force_img, resize_ratio=1.)
                            for img, force_img in zip(vis_images, force_frames)
                        ]
                    else:
                        combined_vis_images = vis_images
                    save_frames_as_video(
                        combined_vis_images,
                        f"{video_save_dir}/replay_{iter_idx:02d}_"
                        f"rst={reset_every}_suc={int(success)}_{os.path.basename(self.bddl_path)}.mp4",
                        fps=10
                    )
                    # break  # only do one demo for now

                if not success:
                    print(f"[Warning] {demo_key} failed with {num_tries} tries.")
                success_stats[
                    f"reset={reset_every}_success={success}"
                ] = success_stats.get(f"reset={reset_every}_success={success}", 0) + 1
                success_stats["total"] += 1
                replay_seed += 1

            print(f"Replay done. Success count: {success_stats}")
        return success_stats

    @staticmethod
    def _parse_demo_slice_token(token: str) -> slice:
        token = token.strip()
        if not (token.startswith("[") and token.endswith("]")):
            raise ValueError(f"Invalid demo slice token: {token}. Expected format like [20:] or [:10].")
        body = token[1:-1].strip()
        if body == "":
            return slice(None, None, None)
        if ":" not in body:
            idx = int(body)
            return slice(idx, idx + 1, 1)

        parts = body.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid demo slice token: {token}. Too many ':' in slice.")
        values = [None if p.strip() == "" else int(p.strip()) for p in parts]
        while len(values) < 3:
            values.append(None)
        return slice(values[0], values[1], values[2])

    @classmethod
    def _normalize_demo_slices(
            cls,
            demo_slices: Optional[Union[slice, str, Tuple[Union[slice, str], ...], List[Any]]],
    ) -> List[slice]:
        if demo_slices is None:
            return [slice(None, None, None)]
        if isinstance(demo_slices, slice):
            return [demo_slices]
        if isinstance(demo_slices, str):
            tokens = re.findall(r"\[[^\[\]]*\]", demo_slices)
            if not tokens:
                raise ValueError(
                    f"Invalid demo_slices={demo_slices}. "
                    f"Use formats like [20:] or ([40:], [:10])."
                )
            return [cls._parse_demo_slice_token(tok) for tok in tokens]
        if isinstance(demo_slices, (tuple, list)):
            out: List[slice] = []
            for item in demo_slices:
                out.extend(cls._normalize_demo_slices(item))
            return out
        raise TypeError(
            f"Unsupported demo_slices type: {type(demo_slices)}. "
            f"Supported: None, slice, str, tuple/list of slice or str."
        )

    @classmethod
    def _select_demo_keys(
            cls,
            all_demo_keys: List[str],
            demo_slices: Optional[Union[slice, str, Tuple[Union[slice, str], ...], List[Any]]] = None,
    ) -> List[str]:
        if demo_slices is None:
            return list(all_demo_keys)
        slice_list = cls._normalize_demo_slices(demo_slices)
        selected_indices = []
        seen = set()
        n = len(all_demo_keys)
        for slc in slice_list:
            for idx in range(*slc.indices(n)):
                if idx in seen:
                    continue
                seen.add(idx)
                selected_indices.append(idx)
        return [all_demo_keys[i] for i in selected_indices]

    def _replay_actions_and_save_hdf5(
            self,
            g_in: h5py.Group,
            g_out: h5py.Group,
            demo_key: str,
            action_shift: int= 1,  # to align with states
            replay_seed: int = None,
            retry_with_reset: int = -1,  # `-1` means not enabled
            # physical shift related
            friction_scale: float = 1.,
            gz_scale: float = 1.,
            solref_scale: float = 1.,
            # visual shift related
            table_center_xyz: tuple = None,
            darken: bool = False,
            color_light: bool = False,
            texture: bool = False,
            texture_dir: str = "images/dtd/",
            camera_shift: bool = False,
    ):
        """
        # In HDF5:
        #    s1 s2 ... s9 s10
        # a0 a1 a2 ... a9
        # In Replay:
        #    s1 s2 ... s9 s10
        #    a1 a2 ... a9
        :param g_in:
        :param g_out:
        :param demo_key:
        :param action_shift:
        :return:
        """
        # 1) 读取 actions
        if "actions" not in g_in:
            raise KeyError(f"{demo_key} has no 'actions' dataset; cannot do actions-replay.")
        hdf5_obs = g_in["obs"]
        hdf5_states = np.array(g_in["states"], dtype=np.float64)  # (T, state_dim), e.g., (251, 51)
        hdf5_actions = np.array(g_in["actions"], dtype=np.float64)  # (T, action_dim), e.g., (251, 7)
        hdf5_robot_states = np.array(g_in["robot_states"], dtype=np.float64)  # (T, robot_state_dim), e.g., (251, 9)
        hdf5_rewards = np.array(g_in["rewards"], dtype=np.uint8)  # (T,)
        hdf5_dones = np.array(g_in["dones"], dtype=np.uint8)  # (T,)
        """
        In HDF5:
        [GRP ] data/demo_48/obs
        -[DSET] data/demo_48/obs/agentview_rgb shape=(243, 128, 128, 3) dtype=uint8
        [DSET] data/demo_48/obs/ee_ori shape=(243, 3) dtype=float64
        [DSET] data/demo_48/obs/ee_pos shape=(243, 3) dtype=float64
        [DSET] data/demo_48/obs/ee_states shape=(243, 6) dtype=float64
        -[DSET] data/demo_48/obs/eye_in_hand_rgb shape=(243, 128, 128, 3) dtype=uint8
        [DSET] data/demo_48/obs/gripper_states shape=(243, 2) dtype=float64
        [DSET] data/demo_48/obs/joint_states shape=(243, 7) dtype=float64
        [DSET] data/demo_48/rewards shape=(243,) dtype=uint8
        [DSET] data/demo_48/robot_states shape=(243, 9) dtype=float64
        [DSET] data/demo_48/states shape=(243, 51) dtype=float64
        e.g.
        robot_states: [ 0.03626977 -0.03618784 -0.20186226  0.01013853  1.1801711   0.9996168
         -0.00674171 -0.02033794 -0.0175275 ] = [gripper_states(2), ee_pos(3), obs.robot0_eef_quat(4)]
        ee_states: [-0.20186226  0.01013853  1.1801711   3.17592    -0.02141934 -0.06461642]
        ee_pos: [-0.20186226  0.01013853  1.1801711 ]
        ee_ori: [ 3.17592    -0.02141934 -0.06461642] = quat_to_rotvec(obs.robot0_eef_quat)
        gripper_states: [ 0.03626977 -0.03618784]
        In env obs:
        --robot0_joint_pos, <class 'numpy.ndarray'>, shape=(7,), min=-2.4276, max=2.2156, dtype=float64
        --robot0_joint_pos_cos, <class 'numpy.ndarray'>, shape=(7,), min=-0.7558, max=1.0000, dtype=float64
        --robot0_joint_pos_sin, <class 'numpy.ndarray'>, shape=(7,), min=-0.6548, max=0.7992, dtype=float64
        --robot0_joint_vel, <class 'numpy.ndarray'>, shape=(7,), min=-0.0079, max=0.0047, dtype=float64
        --robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2044, max=1.1776, dtype=float64
        --robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0276, max=0.9996, dtype=float64
        --robot0_gripper_qpos, <class 'numpy.ndarray'>, shape=(2,), min=-0.0210, max=0.0210, dtype=float64
        --robot0_gripper_qvel, <class 'numpy.ndarray'>, shape=(2,), min=-0.0009, max=0.0009, dtype=float64
        --agentview_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=255.0000, dtype=uint8
        --robot0_eye_in_hand_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=237.0000, dtype=uint8
        --akita_black_bowl_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0629, max=0.8984, dtype=float64
        --akita_black_bowl_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0000, max=0.7071, dtype=float64
        --akita_black_bowl_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=0.0705, max=0.2659, dtype=float64
        --akita_black_bowl_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.7067, max=0.7070, dtype=float32
        --wine_bottle_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.1581, max=0.8989, dtype=float64
        --wine_bottle_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0000, max=1.0000, dtype=float64
        --wine_bottle_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0415, max=0.2758, dtype=float64
        --wine_bottle_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.9996, max=0.0277, dtype=float32
        --robot0_proprio-state, <class 'numpy.ndarray'>, shape=(39,), min=-2.4276, max=2.2156, dtype=float64
        --object-state, <class 'numpy.ndarray'>, shape=(28,), min=-0.9996, max=1.0000, dtype=float64
        --force_ee, <class 'numpy.ndarray'>, shape=(3,), min=-5.0936, max=0.0027, dtype=float32
        --torque_ee, <class 'numpy.ndarray'>, shape=(3,), min=0.0014, max=0.0319, dtype=float32
        --wrench_ee, <class 'numpy.ndarray'>, shape=(6,), min=-5.0936, max=0.0319, dtype=float32
        In env reward:
        reward: <class 'float'>, value=0.0
        """
        vis_hdf5_agent_images = [
            np.flipud(np.array(Image.fromarray(x).resize((256, 256))))
            for x in hdf5_obs["agentview_rgb"]
        ]
        vis_hdf5_eyeinhand_images = [
            np.flipud(np.array(Image.fromarray(x).resize((256, 256))))
            for x in hdf5_obs["eye_in_hand_rgb"]
        ]
        vis_hdf5_images = [
            concatenate_rgb_images(h_img, eih_img, resize_ratio=1., vertical=True)
            for h_img, eih_img in zip(vis_hdf5_agent_images, vis_hdf5_eyeinhand_images)
        ]

        # 2) init storage for replayed data
        action_len = hdf5_actions.shape[0]
        T_out = action_len - action_shift
        H, W = self.camera_hw
        agent_rgb = np.zeros((T_out, H, W, 3), dtype=np.uint8)
        eyeinhand_rgb = np.zeros((T_out, H, W, 3), dtype=np.uint8)
        ee_ori = np.zeros((T_out, 3), dtype=np.float64)
        ee_pos = np.zeros((T_out, 3), dtype=np.float64)
        ee_states = np.zeros((T_out, 6), dtype=np.float64)
        gripper_states = np.zeros((T_out, 2), dtype=np.float64)
        joint_states = np.zeros((T_out, 7), dtype=np.float64)
        wrenches = np.zeros((T_out, 6), np.float32)

        rewards_out = np.zeros((T_out,), dtype=np.uint8)
        robot_states_out = np.zeros((T_out, hdf5_robot_states.shape[1]), dtype=np.float64)
        states_out = np.zeros((T_out, hdf5_states.shape[1]), dtype=np.float64)

        env = self.env
        env.reset()
        # 3) run 5 zero-action steps to stabilize the sim. NOTE: disabling this leads to better results
        # for _ in range(5):
        #     obs, reward, done, info = env.step(np.zeros_like(hdf5_actions[0]))
        #     sim_state = env.env.get_sim_state()
        #     assert sim_state.shape == hdf5_states[0].shape, \
        #         f"Sim state shape {sim_state.shape} does not match hdf5 state shape {hdf5_states[0].shape}!"
        env.set_flattened_state_and_forward(hdf5_states[0])  # set as the 1st state of hdf5 demo

        # add physics randomization here if needed
        possible_gz_range = None
        possible_friction_range = None
        possible_solref_range = None
        if friction_scale != 1.0 or gz_scale != 1.0 or solref_scale != 1.0:
            if gz_scale > 1.0:
                possible_gz_range = (-9.81 * (1 + gz_scale), -9.81)
            elif gz_scale < 1.0:
                possible_gz_range = (-9.81 * (gz_scale ** 2), -9.81 * gz_scale)
            else:
                pass

            if friction_scale > 1.0:
                possible_friction_range = (friction_scale, friction_scale ** 2)
            elif friction_scale < 1.0:
                possible_friction_range = (friction_scale ** 2, friction_scale)
            else:
                pass

            if solref_scale > 1.0:
                possible_solref_range = (solref_scale, solref_scale ** 2)
            elif solref_scale < 1.0:
                possible_solref_range = (solref_scale ** 2, solref_scale)
            else:
                pass

        use_dynamic_color_light = bool(color_light)
        visual_info = self.apply_visual_shift(
            table_center_xyz=table_center_xyz,
            darken=darken,
            color_light=False,
            texture=texture,
            texture_dir=texture_dir,
            camera_shift=camera_shift,
            rng=np.random.RandomState(replay_seed),
            verbose=True,
        )
        if visual_info.get("reset_applied", False):
            # Texture XML reset may rebuild sim/model and discard the demo state.
            env.set_flattened_state_and_forward(hdf5_states[0])
        self.apply_physics_shift(
            frictionloss_scale=possible_friction_range,
            damping_scale=possible_friction_range,
            sliding_friction_scale=possible_friction_range,
            gravity_z_range=possible_gz_range,
            tweak_solref_range=possible_solref_range,
            rng=np.random.RandomState(replay_seed),
            verbose=True,
        )
        if use_dynamic_color_light:
            self.visual_helper.capture_dynamic_light_base(model=env.env.sim.model)
            self.visual_helper.apply_dynamic_rainbow_light(
                model=env.env.sim.model,
                step_idx=0,
            )
            env.env.sim.forward()

        obs = self.regenerate_obs_from_current_state()
        reward = 0
        sim_state = env.env.get_sim_state()

        vis_replay_images = []

        # 4) step through actions
        # 4.1) try to fully rollout without resets, we'll handle all the env states by ourselves
        num_tries = 1
        reset_every = -1  # do not reset
        for t in range(0, T_out):
            vis_replay_images.append(
                concatenate_rgb_images(
                    np.flipud(np.array(Image.fromarray(obs["agentview_image"]).resize((256, 256)))),
                    np.flipud(np.array(Image.fromarray(obs["robot0_eye_in_hand_image"]).resize((256, 256)))),
                    resize_ratio=1.0,
                    vertical=True,
                )
            )

            agent_rgb[t] = obs["agentview_image"]
            eyeinhand_rgb[t] = obs["robot0_eye_in_hand_image"]
            wrenches[t] = obs["wrench_ee"]
            ee_ori[t] = self.quat_xyzw_to_rotvec(obs["robot0_eef_quat"])
            ee_pos[t] = obs["robot0_eef_pos"]
            ee_states[t] = np.concatenate((ee_pos[t], ee_ori[t]), axis=0)
            gripper_states[t] = obs["robot0_gripper_qpos"]
            joint_states[t] = obs["robot0_joint_pos"]
            rewards_out[t] = int(reward)
            robot_states_out[t] = np.concatenate((gripper_states[t], ee_pos[t], obs["robot0_eef_quat"]), axis=0)
            states_out[t] = sim_state

            ## Option 2. Step with actions
            if use_dynamic_color_light:
                self.visual_helper.apply_dynamic_rainbow_light(
                    model=env.env.sim.model,
                    step_idx=t,
                )
                env.env.sim.forward()
            obs, reward, done, info = env.step(hdf5_actions[t + action_shift])
            sim_state = env.env.get_sim_state()

        vis_replay_images.append(
            concatenate_rgb_images(
                np.flipud(np.array(Image.fromarray(obs["agentview_image"]).resize((256, 256)))),
                np.flipud(np.array(Image.fromarray(obs["robot0_eye_in_hand_image"]).resize((256, 256)))),
                resize_ratio=1.0,
                vertical=True,
            )
        )  # vis the final step
        success = env.env.check_success()

        # 4.2) try to reset every N steps to reduce compounding errors, we'll copy the states from hdf5
        #      Although this can also fail, but since we do not load states from env, it's ok to read force data
        if retry_with_reset >= 1 and not success:
            num_tries += 1
            print(f"[Warning] 1st try to replay ({demo_key}) failed, will make 2nd try with periodic resets...")
            env.reset()
            _ = self.apply_visual_shift(
                table_center_xyz=table_center_xyz,
                darken=darken,
                color_light=False,
                texture=texture,
                texture_dir=texture_dir,
                camera_shift=camera_shift,
                rng=np.random.RandomState(replay_seed),
                verbose=True,
            )
            self.apply_physics_shift(
                frictionloss_scale=possible_friction_range,
                damping_scale=possible_friction_range,
                sliding_friction_scale=possible_friction_range,
                gravity_z_range=possible_gz_range,
                tweak_solref_range=possible_solref_range,
                rng=np.random.RandomState(replay_seed),
                verbose=True,
            )
            # Set demo state after shifts (texture shift may internally reset XML/sim).
            env.set_flattened_state_and_forward(hdf5_states[0])
            if use_dynamic_color_light:
                self.visual_helper.capture_dynamic_light_base(model=env.env.sim.model)
                self.visual_helper.apply_dynamic_rainbow_light(
                    model=env.env.sim.model,
                    step_idx=0,
                )
                env.env.sim.forward()
            obs = self.regenerate_obs_from_current_state()
            reward = 0
            sim_state = env.env.get_sim_state()
            vis_replay_images = []  # reset
            reset_every = retry_with_reset  # reset sim state every N steps to reduce compounding errors

            for t in range(0, T_out):
                vis_replay_images.append(
                    concatenate_rgb_images(
                        np.flipud(np.array(Image.fromarray(obs["agentview_image"]).resize((256, 256)))),
                        np.flipud(np.array(Image.fromarray(obs["robot0_eye_in_hand_image"]).resize((256, 256)))),
                        resize_ratio=1.0,
                        vertical=True,
                    )
                )

                wrenches[t] = obs["wrench_ee"]  # NOTE: always from sim
                if self.retry_obs_source == "hdf5":
                    agent_rgb[t] = hdf5_obs["agentview_rgb"][t]
                    eyeinhand_rgb[t] = hdf5_obs["eye_in_hand_rgb"][t]
                    ee_ori[t] = hdf5_obs["ee_ori"][t]
                    ee_pos[t] = hdf5_obs["ee_pos"][t]
                    ee_states[t] = hdf5_obs["ee_states"][t]
                    gripper_states[t] = hdf5_obs["gripper_states"][t]
                    joint_states[t] = hdf5_obs["joint_states"][t]
                    rewards_out[t] = hdf5_rewards[t]
                    robot_states_out[t] = hdf5_robot_states[t]
                    states_out[t] = hdf5_states[t]
                else:
                    agent_rgb[t] = obs["agentview_image"]
                    eyeinhand_rgb[t] = obs["robot0_eye_in_hand_image"]
                    ee_ori[t] = self.quat_xyzw_to_rotvec(obs["robot0_eef_quat"])
                    ee_pos[t] = obs["robot0_eef_pos"]
                    ee_states[t] = np.concatenate((ee_pos[t], ee_ori[t]), axis=0)
                    gripper_states[t] = obs["robot0_gripper_qpos"]
                    joint_states[t] = obs["robot0_joint_pos"]
                    rewards_out[t] = int(reward)
                    robot_states_out[t] = np.concatenate((gripper_states[t], ee_pos[t], obs["robot0_eef_quat"]), axis=0)
                    states_out[t] = sim_state

                if t % reset_every == (reset_every - 1):
                    ## Option 1. Very direct way: set flattened state
                    env.set_flattened_state_and_forward(hdf5_states[t])
                ## Option 2. Step with actions
                if use_dynamic_color_light:
                    self.visual_helper.apply_dynamic_rainbow_light(
                        model=env.env.sim.model,
                        step_idx=t,
                    )
                    env.env.sim.forward()
                obs, reward, done, info = env.step(hdf5_actions[t + action_shift])
                assert "wrench_ee" in obs, "Wrench obs missing in env!"
                sim_state = env.env.get_sim_state()

            success = env.env.check_success()
            vis_replay_images.append(
                concatenate_rgb_images(
                    np.flipud(np.array(Image.fromarray(obs["agentview_image"]).resize((256, 256)))),
                    np.flipud(np.array(Image.fromarray(obs["robot0_eye_in_hand_image"]).resize((256, 256)))),
                    resize_ratio=1.0,
                    vertical=True,
                )
            )  # vis the final step
        assert len(vis_hdf5_images) == len(vis_replay_images), \
            "Length mismatch: len(hdf5 images) should = len(replay images)!"

        # 新增 dataset
        obs_out = g_out.create_group("obs")
        obs_out.create_dataset("agentview_rgb", data=agent_rgb, dtype="uint8")
        obs_out.create_dataset("eye_in_hand_rgb", data=eyeinhand_rgb, dtype="uint8")
        obs_out.create_dataset("joint_states", data=joint_states, dtype="float64")
        obs_out.create_dataset("gripper_states", data=gripper_states, dtype="float64")
        obs_out.create_dataset("ee_pos", data=ee_pos, dtype="float64")
        obs_out.create_dataset("ee_ori", data=ee_ori, dtype="float64")
        obs_out.create_dataset("ee_states", data=ee_states, dtype="float64")
        obs_out.create_dataset("wrenches", data=wrenches, dtype="float32")

        g_out.create_dataset("dones", data=rewards_out, dtype="uint8")
        g_out.create_dataset("rewards", data=rewards_out, dtype="uint8")
        g_out.create_dataset("robot_states", data=robot_states_out, dtype="float64")
        g_out.create_dataset("states", data=states_out, dtype="float64")
        g_out.create_dataset("actions", data=hdf5_actions[action_shift:], dtype="float64")

        g_out.create_dataset("replay_reset_every", data=np.array([reset_every], dtype=np.int8), dtype="int8")
        g_out.create_dataset("replay_success", data=np.array([int(success)], dtype=np.int8), dtype="int8")
        g_out.attrs["retry_obs_source"] = self.retry_obs_source

        return {
            "success": success,
            "vis_hdf5_images": vis_hdf5_images,
            "vis_replay_images": vis_replay_images,
            "wrenches": wrenches,
            "reset_every": reset_every,  # -1 means no resets, >=1 means reset every N steps
            "num_tries": num_tries,
        }

    @staticmethod
    def copy_attrs(src, dst) -> int:
        cnt = 0
        for k, v in src.attrs.items():
            dst.attrs[k] = v
            cnt += 1
        return cnt

    @staticmethod
    def quat_xyzw_to_rotvec(q_xyzw_D: np.ndarray) -> np.ndarray:
        """
        Input: quat in xyzw
        Output: rotation vector (axis-angle), shape (3,) = theta * axis
        """
        # robosuite provided function
        rotvec_D = transform_utils.quat2axisangle(q_xyzw_D)  # (3,)
        return rotvec_D

    @staticmethod
    def summarize_array(x: np.ndarray):
        """
        x: shape (N, D)
        returns dict of stats per dim
        """
        return {
            "mean": np.mean(x, axis=0),
            "max": np.max(x, axis=0),
            "p01": np.percentile(x, 1, axis=0),
            "p99": np.percentile(x, 99, axis=0),
        }
