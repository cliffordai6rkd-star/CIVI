import hashlib
import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult
from transforms3d.quaternions import qnorm


from robosuite.utils.binding_utils import MjModel


class ColorLightHelper:
    def __init__(
        self,
        sim_model: MjModel,
        random_seed: int,
        original_params_snap: Dict[str, Any],
    ):
        self.model = sim_model
        self.random_seed = random_seed
        self.original_params_snap = original_params_snap

    def _make_rng(self, rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        if rng is not None:
            return rng
        return np.random.RandomState(self.random_seed)

    @staticmethod
    def _set_headlight_field(headlight: Any, name: str, value: np.ndarray):
        if not hasattr(headlight, name):
            return
        try:
            getattr(headlight, name)[:] = value
        except Exception:
            setattr(headlight, name, value)

    def apply_color_light(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        color_rgb: Optional[Sequence[float]] = None,
        color_range: Tuple[float, float] = (0.25, 1.0),
        preserve_luminance: bool = False,
        diffuse_scale: Tuple[float, float] = (1.2, 2.0),
        specular_scale: Tuple[float, float] = (1.0, 1.8),
        ambient_scale: Tuple[float, float] = (0.8, 1.4),
        affect_headlight: bool = True,
        headlight_scale: Tuple[float, float] = (1.0, 1.6),
        base_from_current: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        rng = self._make_rng(rng)

        if color_rgb is None:
            tint = rng.uniform(color_range[0], color_range[1], size=3).astype(np.float64)
        else:
            tint = np.asarray(color_rgb, dtype=np.float64).reshape(3)

        tint = np.clip(tint, 0.0, None)
        if preserve_luminance:
            m = float(np.mean(tint))
            if m > 1e-8:
                tint = tint / m

        d_scale = float(rng.uniform(*diffuse_scale))
        s_scale = float(rng.uniform(*specular_scale))
        a_scale = float(rng.uniform(*ambient_scale))

        if base_from_current:
            base_diff = np.asarray(model.light_diffuse, dtype=np.float64).copy()
            base_spec = np.asarray(model.light_specular, dtype=np.float64).copy()
            base_amb = np.asarray(model.light_ambient, dtype=np.float64).copy()
        else:
            base_diff = np.asarray(self.original_params_snap.get("light_diffuse", model.light_diffuse), dtype=np.float64)
            base_spec = np.asarray(self.original_params_snap.get("light_specular", model.light_specular), dtype=np.float64)
            base_amb = np.asarray(self.original_params_snap.get("light_ambient", model.light_ambient), dtype=np.float64)

        if hasattr(model, "light_diffuse"):
            model.light_diffuse[:] = base_diff * d_scale * tint.reshape(1, 3)
        if hasattr(model, "light_specular"):
            model.light_specular[:] = base_spec * s_scale * tint.reshape(1, 3)
        if hasattr(model, "light_ambient"):
            model.light_ambient[:] = base_amb * a_scale * tint.reshape(1, 3)

        h_scale = None
        if affect_headlight and hasattr(model, "vis") and hasattr(model.vis, "headlight"):
            h_scale = float(rng.uniform(*headlight_scale))
            headlight = model.vis.headlight
            if base_from_current:
                base_head = {
                    name: np.asarray(getattr(headlight, name), dtype=np.float64).copy()
                    for name in ("diffuse", "specular", "ambient")
                    if hasattr(headlight, name)
                }
            else:
                base_head = self.original_params_snap.get("headlight", {})

            if "diffuse" in base_head:
                self._set_headlight_field(headlight, "diffuse", np.asarray(base_head["diffuse"]) * h_scale * tint)
            if "specular" in base_head:
                self._set_headlight_field(headlight, "specular", np.asarray(base_head["specular"]) * h_scale * tint)
            if "ambient" in base_head:
                self._set_headlight_field(headlight, "ambient", np.asarray(base_head["ambient"]) * h_scale * tint)

        info = {
            "color_rgb": tint,
            "diffuse_scale": d_scale,
            "specular_scale": s_scale,
            "ambient_scale": a_scale,
            "headlight_scale": h_scale,
            "base_from_current": base_from_current,
        }
        if verbose:
            print(f"[Info] Applied color light: {info}")
        return info


class DarkenHelper:
    def __init__(
        self,
        sim_model: MjModel,
        random_seed: int,
        original_params_snap: Dict[str, Any],
    ):
        self.model = sim_model
        self.random_seed = random_seed
        self.original_params_snap = original_params_snap

    def _make_rng(self, rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        if rng is not None:
            return rng
        return np.random.RandomState(self.random_seed)

    @staticmethod
    def _set_headlight_field(headlight: Any, name: str, value: np.ndarray):
        if not hasattr(headlight, name):
            return
        try:
            getattr(headlight, name)[:] = value
        except Exception:
            setattr(headlight, name, value)

    def apply_single_flashlight(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        table_center_xyz: Sequence[float] = (0.0, 0.0, 0.9),
        table_half_size_xy: Sequence[float] = (0.03, 0.03),
        light_height_range: Tuple[float, float] = (1.6, 1.9),
        cutoff_range: Tuple[float, float] = (25.0, 45.0),
        exponent_range: Tuple[float, float] = (35.0, 50.0),
        attenuation: Sequence[float] = (1.0, 0.0, 0.08),
        diffuse_intensity: Tuple[float, float] = (2.0, 2.6),
        specular_intensity: Tuple[float, float] = (0.05, 0.18),
        ambient_intensity: Tuple[float, float] = (0.5, 0.6),
        other_diffuse_scale: Tuple[float, float] = (0.1, 0.20),
        other_specular_scale: Tuple[float, float] = (0.1, 0.20),
        other_ambient_scale: Tuple[float, float] = (0.1, 0.15),
        headlight_scale: Tuple[float, float] = (0.03, 0.10),
        selected_light_id: int = 0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        rng = self._make_rng(rng)

        nlight = int(getattr(model, "nlight", 0))
        if nlight <= 0:
            raise RuntimeError("No light source found in this model.")
        sid = int(np.clip(selected_light_id, 0, nlight - 1))

        base_diff = np.asarray(self.original_params_snap.get("light_diffuse", model.light_diffuse), dtype=np.float64)
        base_spec = np.asarray(self.original_params_snap.get("light_specular", model.light_specular), dtype=np.float64)
        base_amb = np.asarray(self.original_params_snap.get("light_ambient", model.light_ambient), dtype=np.float64)

        d_other = float(rng.uniform(*other_diffuse_scale))
        s_other = float(rng.uniform(*other_specular_scale))
        a_other = float(rng.uniform(*other_ambient_scale))

        if hasattr(model, "light_diffuse"):
            model.light_diffuse[:] = base_diff * d_other
        if hasattr(model, "light_specular"):
            model.light_specular[:] = base_spec * s_other
        if hasattr(model, "light_ambient"):
            model.light_ambient[:] = base_amb * a_other

        d_sel = float(rng.uniform(*diffuse_intensity))
        s_sel = float(rng.uniform(*specular_intensity))
        a_sel = float(rng.uniform(*ambient_intensity))

        model.light_diffuse[sid] = np.array([d_sel, d_sel, d_sel], dtype=np.float64)
        model.light_specular[sid] = np.array([s_sel, s_sel, s_sel], dtype=np.float64)
        model.light_ambient[sid] = np.array([a_sel, a_sel, a_sel], dtype=np.float64)

        cx, cy, cz = [float(x) for x in table_center_xyz]
        hx, hy = [float(x) for x in table_half_size_xy]
        tx = rng.uniform(cx - hx, cx + hx)
        ty = rng.uniform(cy - hy, cy + hy)
        tz = cz
        light_z = float(rng.uniform(*light_height_range))

        source = np.array([tx, ty, light_z], dtype=np.float64)
        target = np.array([tx, ty, tz], dtype=np.float64)
        direction = target - source
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        if hasattr(model, "light_pos"):
            model.light_pos[sid] = source
        if hasattr(model, "light_dir"):
            model.light_dir[sid] = direction
        if hasattr(model, "light_directional"):
            model.light_directional[sid] = 0
        if hasattr(model, "light_cutoff"):
            model.light_cutoff[sid] = float(rng.uniform(*cutoff_range))
        if hasattr(model, "light_exponent"):
            model.light_exponent[sid] = float(rng.uniform(*exponent_range))
        if hasattr(model, "light_attenuation"):
            model.light_attenuation[sid] = np.asarray(attenuation, dtype=np.float64)
        if hasattr(model, "light_active"):
            model.light_active[:] = 1

        h_scale = None
        if hasattr(model, "vis") and hasattr(model.vis, "headlight"):
            h_scale = float(rng.uniform(*headlight_scale))
            headlight = model.vis.headlight
            base_head = self.original_params_snap.get("headlight", {})
            if "diffuse" in base_head:
                self._set_headlight_field(headlight, "diffuse", np.asarray(base_head["diffuse"]) * h_scale)
            if "specular" in base_head:
                self._set_headlight_field(headlight, "specular", np.asarray(base_head["specular"]) * h_scale)
            if "ambient" in base_head:
                self._set_headlight_field(headlight, "ambient", np.asarray(base_head["ambient"]) * h_scale)

        info = {
            "selected_light_id": sid,
            "flashlight_position": source,
            "flashlight_target": target,
            "flashlight_direction": direction,
            "cutoff": float(model.light_cutoff[sid]) if hasattr(model, "light_cutoff") else None,
            "exponent": float(model.light_exponent[sid]) if hasattr(model, "light_exponent") else None,
            "headlight_scale": h_scale,
            "other_light_scale": {
                "diffuse": d_other,
                "specular": s_other,
                "ambient": a_other,
            },
            "selected_light_intensity": {
                "diffuse": d_sel,
                "specular": s_sel,
                "ambient": a_sel,
            },
        }
        if verbose:
            print(f"[Info] Applied single flashlight: {info}")
        return info


class TextureHelper:
    DEFAULT_TABLE_KEYWORDS = (
        "table",
        "desk",
        "counter",
        "workspace",
        "surface",
        "wood",
        "top",
        "tabletop",
        "study",
    )
    DEFAULT_EXCLUDE_KEYWORDS = (
        "mug",
        "caddy",
        "book",
        "bottle",
        "cup",
        "wall",
        "painting",
        "shelf",
        "legs",
        "robot",
        "gripper",
    )

    def __init__(
        self,
        sim_model: MjModel,
        random_seed: int,
        original_params_snap: Dict[str, Any],
    ):
        self.model = sim_model
        self.random_seed = random_seed
        self.original_params_snap = original_params_snap

    def _make_rng(self, rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        if rng is not None:
            return rng
        return np.random.RandomState(self.random_seed)

    @staticmethod
    def _iter_nested_envs(root_env: Any, max_depth: int = 6) -> List[Any]:
        if root_env is None:
            return []
        queue: List[Tuple[Any, int]] = [(root_env, 0)]
        out: List[Any] = []
        seen = set()
        while queue:
            obj, depth = queue.pop(0)
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            out.append(obj)
            if depth >= max_depth:
                continue
            for attr in ("env", "_env", "wrapped_env"):
                nxt = getattr(obj, attr, None)
                if nxt is not None:
                    queue.append((nxt, depth + 1))
        return out

    @classmethod
    def _find_sim_from_env(cls, env: Any) -> Optional[Any]:
        for obj in cls._iter_nested_envs(env):
            sim = getattr(obj, "sim", None)
            if sim is not None:
                return sim
        return None

    @classmethod
    def _find_env_with_reset(cls, env: Any) -> Optional[Any]:
        for obj in cls._iter_nested_envs(env):
            fn = getattr(obj, "reset_from_xml_string", None)
            if callable(fn):
                return obj
        return None

    @staticmethod
    def _collect_image_paths(
        texture_dir: str,
        recursive: bool,
        valid_exts: Tuple[str, ...],
    ) -> List[Path]:
        root = Path(texture_dir).expanduser().resolve()
        normalized_exts = tuple(x.lower() for x in valid_exts)
        paths = root.rglob("*") if recursive else root.glob("*")
        return sorted([p for p in paths if p.is_file() and p.suffix.lower() in normalized_exts])

    @staticmethod
    def _ensure_png_for_xml_texture(image_path: Path) -> str:
        path = image_path.expanduser().resolve()
        if path.suffix.lower() == ".png":
            return str(path)

        cache_dir = Path("/tmp/libero_texture_png_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        stat = path.stat()
        sig = f"{path}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}"
        digest = hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]
        out_path = cache_dir / f"{path.stem}_{digest}.png"
        if not out_path.exists():
            with Image.open(path) as im:
                im.convert("RGB").save(out_path, format="PNG")
        return str(out_path)

    @staticmethod
    def _get_xml_string(model: Optional[MjModel], sim: Optional[Any]) -> str:
        if sim is not None and hasattr(sim, "model") and hasattr(sim.model, "get_xml"):
            return sim.model.get_xml()
        if model is not None and hasattr(model, "get_xml"):
            return model.get_xml()
        raise RuntimeError("Cannot fetch XML string from sim/model")

    @staticmethod
    def _name_match(query: str, name: str) -> bool:
        q = (query or "").strip().lower()
        n = (name or "").strip().lower()
        if not q or not n:
            return False
        return n == q or n.endswith(q) or (q in n)

    @staticmethod
    def _contains_any(text: str, keywords: Sequence[str]) -> bool:
        t = (text or "").lower()
        for k in keywords:
            k2 = (k or "").strip().lower()
            if k2 and k2 in t:
                return True
        return False

    def apply_table_texture_via_xml_wrapper(
        self,
        texture_dir: str,
        env: Optional[Any] = None,
        model: Optional[MjModel] = None,
        sim: Optional[Any] = None,
        rng: Optional[np.random.RandomState] = None,
        recursive: bool = True,
        valid_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        preferred_material_names: Sequence[str] = ("desk", "table_texture", "table_mat"),
        preferred_texture_names: Sequence[str] = ("tex-desk", "tex-table", "textable"),
        allow_auto_detect: bool = True,
        table_keywords: Sequence[str] = DEFAULT_TABLE_KEYWORDS,
        exclude_keywords: Sequence[str] = DEFAULT_EXCLUDE_KEYWORDS,
        apply_to_all_keyword_textures: bool = True,
        reset_env: bool = True,
        return_xml_string: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        rng = self._make_rng(rng)

        if sim is None:
            sim = self._find_sim_from_env(env)

        image_paths = self._collect_image_paths(
            texture_dir=texture_dir,
            recursive=recursive,
            valid_exts=valid_exts,
        )
        if not image_paths:
            raise RuntimeError(f"No images found in {texture_dir}")

        source_image_path = image_paths[int(rng.randint(0, len(image_paths)))].resolve()
        selected_image_path = self._ensure_png_for_xml_texture(source_image_path)

        xml_string = self._get_xml_string(model=model, sim=sim)
        root = ET.fromstring(xml_string)
        asset = root.find("asset")
        if asset is None:
            raise RuntimeError("XML has no <asset> section")

        texture_elems = {
            str(tex.get("name")): tex
            for tex in asset.findall("texture")
            if tex.get("name")
        }
        material_to_texture = {
            str(mat.get("name")): str(mat.get("texture"))
            for mat in asset.findall("material")
            if mat.get("name") and mat.get("texture")
        }

        target_texture_names: List[str] = []
        resolve_details: List[Dict[str, str]] = []

        # 1) preferred material -> texture
        for mat_name in preferred_material_names:
            for xml_mat_name, tex_name in material_to_texture.items():
                if self._name_match(mat_name, xml_mat_name) and tex_name in texture_elems:
                    target_texture_names.append(tex_name)
                    resolve_details.append(
                        {
                            "resolve_source": "preferred_material",
                            "input_name": str(mat_name),
                            "resolved_material_name": xml_mat_name,
                            "resolved_texture_name": tex_name,
                        }
                    )

        # 2) preferred texture name
        for tex_name in preferred_texture_names:
            for xml_tex_name in texture_elems.keys():
                if self._name_match(tex_name, xml_tex_name):
                    target_texture_names.append(xml_tex_name)
                    resolve_details.append(
                        {
                            "resolve_source": "preferred_texture",
                            "input_name": str(tex_name),
                            "resolved_material_name": "",
                            "resolved_texture_name": xml_tex_name,
                        }
                    )

        # 3) keyword-based auto collect (focus on simplification)
        if apply_to_all_keyword_textures or allow_auto_detect:
            for mat_name, tex_name in material_to_texture.items():
                joined = f"{mat_name} {tex_name}".lower()
                if self._contains_any(joined, table_keywords) and not self._contains_any(joined, exclude_keywords):
                    if tex_name in texture_elems:
                        target_texture_names.append(tex_name)
                        resolve_details.append(
                            {
                                "resolve_source": "keyword_material",
                                "input_name": "",
                                "resolved_material_name": mat_name,
                                "resolved_texture_name": tex_name,
                            }
                        )

            for tex_name in texture_elems.keys():
                if self._contains_any(tex_name, table_keywords) and not self._contains_any(tex_name, exclude_keywords):
                    target_texture_names.append(tex_name)
                    resolve_details.append(
                        {
                            "resolve_source": "keyword_texture",
                            "input_name": "",
                            "resolved_material_name": "",
                            "resolved_texture_name": tex_name,
                        }
                    )

        # 4) fallback one texture
        if not target_texture_names:
            for tex_name in texture_elems.keys():
                if "desk" in tex_name.lower() or "table" in tex_name.lower():
                    target_texture_names = [tex_name]
                    resolve_details.append(
                        {
                            "resolve_source": "fallback_name_contains",
                            "input_name": "",
                            "resolved_material_name": "",
                            "resolved_texture_name": tex_name,
                        }
                    )
                    break

        if not target_texture_names:
            raise RuntimeError("Cannot resolve any texture target in XML")

        # de-dup keep order
        seen = set()
        target_texture_names = [
            name for name in target_texture_names
            if not (name in seen or seen.add(name))
        ]

        remove_keys = (
            "builtin",
            "rgb1",
            "rgb2",
            "mark",
            "markrgb",
            "width",
            "height",
            "random",
            "gridsize",
            "gridlayout",
        )
        modified_textures: List[Dict[str, Any]] = []
        for tex_name in target_texture_names:
            tex_elem = texture_elems.get(tex_name)
            if tex_elem is None:
                continue
            old_file = tex_elem.get("file")
            old_type = tex_elem.get("type")

            tex_elem.set("file", selected_image_path)
            tex_elem.set("type", "2d")
            removed_attrs = []
            for key in remove_keys:
                if key in tex_elem.attrib:
                    tex_elem.attrib.pop(key, None)
                    removed_attrs.append(key)

            modified_textures.append(
                {
                    "texture_name": tex_name,
                    "old_file": old_file,
                    "new_file": selected_image_path,
                    "old_type": old_type,
                    "new_type": "2d",
                    "removed_attrs": removed_attrs,
                }
            )

        new_xml_string = ET.tostring(root, encoding="unicode")

        reset_applied = False
        reset_env_class = None
        if reset_env:
            reset_target = self._find_env_with_reset(env)
            if reset_target is None:
                raise RuntimeError("reset_env=True but env has no reset_from_xml_string")
            reset_target.reset_from_xml_string(new_xml_string)
            reset_applied = True
            reset_env_class = type(reset_target).__name__

        info = {
            "mode": "xml_wrapper",
            "source_image_path": str(source_image_path),
            "selected_image_path": selected_image_path,
            "num_candidate_images": len(image_paths),
            "target_texture_names": target_texture_names,
            "num_target_textures": len(target_texture_names),
            "modified_textures": modified_textures,
            "resolve_details": resolve_details,
            "reset_applied": reset_applied,
            "reset_env_class": reset_env_class,
        }
        if return_xml_string:
            info["xml_string"] = new_xml_string
        if verbose:
            print(f"[Info] Applied table texture via XML wrapper: {info}")
        return info


class CameraShiftHelper:
    DEFAULT_CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
    DEFAULT_POS_RANGE = ((-0.14, 0.14), (-0.14, 0.14), (-0.12, 0.12))
    DEFAULT_ROT_RANGE_DEG = ((-15.0, 15.0), (-15.0, 3.0), (-15.0, 15.0))
    DEFAULT_FOVY_SHIFT_RANGE = (-8.0, 8.0)

    def __init__(
        self,
        sim_model: MjModel,
        random_seed: int,
    ):
        self.model = sim_model
        self.random_seed = random_seed
        self.camera_base_snap: Optional[Dict[str, Any]] = None

    def _make_rng(self, rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        return rng if rng is not None else np.random.RandomState(self.random_seed)

    @staticmethod
    def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
        n = float(qnorm(q))
        if n < 1e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(q, dtype=np.float64) / n

    @classmethod
    def _resolve_cameras(cls, model: MjModel, camera_names: Optional[Sequence[str]]) -> List[Tuple[str, int]]:
        names = list(cls.DEFAULT_CAMERA_NAMES) if camera_names is None else (
            [camera_names] if isinstance(camera_names, str) else list(camera_names)
        )
        names = [str(x) for x in names if str(x).strip()]
        names = list(dict.fromkeys(names))

        name2id = getattr(model, "camera_name2id", None) or getattr(model, "cam_name2id", None)
        if not callable(name2id):
            raise RuntimeError("sim.model has no camera_name2id/cam_name2id method")

        resolved: List[Tuple[str, int]] = []
        missing: List[str] = []
        for name in names:
            try:
                cid = int(name2id(name))
            except Exception:
                cid = -1
            if cid >= 0:
                resolved.append((name, cid))
            else:
                missing.append(name)

        if missing and (camera_names is not None):
            raise RuntimeError(f"Cannot resolve camera ids for names: {missing}")
        if not resolved:
            raise RuntimeError(f"Cannot resolve any valid camera ids from names: {names}")
        return resolved

    def clear_camera_base(self):
        self.camera_base_snap = None

    def capture_camera_base(
        self,
        model: Optional[MjModel] = None,
        camera_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        entries = []
        for camera_name, camera_id in self._resolve_cameras(model, camera_names):
            entries.append(
                {
                    "camera_name": camera_name,
                    "camera_id": int(camera_id),
                    "pos": np.asarray(model.cam_pos[camera_id], dtype=np.float64).copy(),
                    "quat": self._normalize_quat_wxyz(np.asarray(model.cam_quat[camera_id], dtype=np.float64)),
                    "fovy": float(model.cam_fovy[camera_id]) if hasattr(model, "cam_fovy") else None,
                }
            )
        self.camera_base_snap = {
            "camera_names": [e["camera_name"] for e in entries],
            "entries": entries,
        }
        return self.camera_base_snap

    def restore_camera_base(self, model: Optional[MjModel] = None) -> bool:
        model = self.model if model is None else model
        if self.camera_base_snap is None:
            return False
        for entry in self.camera_base_snap["entries"]:
            cid = int(entry["camera_id"])
            model.cam_pos[cid] = np.asarray(entry["pos"], dtype=np.float64)
            model.cam_quat[cid] = self._normalize_quat_wxyz(np.asarray(entry["quat"], dtype=np.float64))
            if hasattr(model, "cam_fovy") and entry["fovy"] is not None:
                model.cam_fovy[cid] = float(entry["fovy"])
        return True

    def apply_camera_shift(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        camera_names: Optional[Sequence[str]] = None,
        pos_shift_range: Optional[Sequence[Sequence[float]]] = None,
        rot_shift_range_deg: Optional[Sequence[Sequence[float]]] = None,
        fovy_shift_range: Optional[Sequence[float]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        rng = self._make_rng(rng)
        pos_rng = np.asarray(self.DEFAULT_POS_RANGE if pos_shift_range is None else pos_shift_range, dtype=np.float64).reshape(3, 2)
        rot_rng = np.asarray(self.DEFAULT_ROT_RANGE_DEG if rot_shift_range_deg is None else rot_shift_range_deg, dtype=np.float64).reshape(3, 2)
        pos_rng = np.sort(pos_rng, axis=1)
        rot_rng = np.sort(rot_rng, axis=1)

        custom_fovy = None
        if fovy_shift_range is not None:
            custom_fovy = np.sort(np.asarray(fovy_shift_range, dtype=np.float64).reshape(2))

        requested_names = list(self.DEFAULT_CAMERA_NAMES) if camera_names is None else (
            [camera_names] if isinstance(camera_names, str) else list(camera_names)
        )
        requested_names = [str(x) for x in requested_names if str(x).strip()]
        requested_names = list(dict.fromkeys(requested_names))

        if (self.camera_base_snap is None) or (self.camera_base_snap.get("camera_names", []) != requested_names):
            self.capture_camera_base(model=model, camera_names=requested_names)
        else:
            self.restore_camera_base(model=model)

        assert self.camera_base_snap is not None
        camera_infos: List[Dict[str, Any]] = []
        for entry in self.camera_base_snap["entries"]:
            cid = int(entry["camera_id"])
            name = str(entry["camera_name"])
            is_fov_only = ("eye_in_hand" in name.lower())

            base_pos = np.asarray(entry["pos"], dtype=np.float64)
            base_quat = self._normalize_quat_wxyz(np.asarray(entry["quat"], dtype=np.float64))

            if is_fov_only:
                delta_pos = np.zeros(3, dtype=np.float64)
                delta_rot_deg = np.zeros(3, dtype=np.float64)
                new_pos = base_pos.copy()
                new_quat = base_quat.copy()
            else:
                delta_pos = rng.uniform(pos_rng[:, 0], pos_rng[:, 1])
                delta_rot_deg = rng.uniform(rot_rng[:, 0], rot_rng[:, 1])
                delta_quat = np.asarray(
                    euler2quat(
                        np.deg2rad(float(delta_rot_deg[0])),
                        np.deg2rad(float(delta_rot_deg[1])),
                        np.deg2rad(float(delta_rot_deg[2])),
                        axes="sxyz",
                    ),
                    dtype=np.float64,
                )
                new_pos = base_pos + delta_pos
                new_quat = self._normalize_quat_wxyz(np.asarray(qmult(base_quat, delta_quat), dtype=np.float64))

            model.cam_pos[cid] = new_pos
            model.cam_quat[cid] = new_quat

            base_fovy = entry["fovy"]
            delta_fovy = None
            new_fovy = None
            fovy_rng = custom_fovy if custom_fovy is not None else (
                np.asarray(self.DEFAULT_FOVY_SHIFT_RANGE, dtype=np.float64) if is_fov_only else None
            )
            if hasattr(model, "cam_fovy") and (base_fovy is not None) and (fovy_rng is not None):
                delta_fovy = float(rng.uniform(float(fovy_rng[0]), float(fovy_rng[1])))
                new_fovy = float(np.clip(float(base_fovy) + delta_fovy, 1.0, 179.0))
                model.cam_fovy[cid] = new_fovy

            camera_infos.append(
                {
                    "camera_name": name,
                    "camera_id": cid,
                    "fov_only": is_fov_only,
                    "base_pos": base_pos,
                    "new_pos": np.asarray(new_pos, dtype=np.float64),
                    "delta_pos": np.asarray(delta_pos, dtype=np.float64),
                    "base_quat": base_quat,
                    "new_quat": self._normalize_quat_wxyz(new_quat),
                    "delta_rot_deg_xyz": np.asarray(delta_rot_deg, dtype=np.float64),
                    "base_fovy": base_fovy,
                    "new_fovy": new_fovy,
                    "delta_fovy": delta_fovy,
                }
            )

        info = {
            "camera_names": requested_names,
            "pos_shift_range": pos_rng,
            "rot_shift_range_deg": rot_rng,
            "fovy_shift_range": custom_fovy,
            "fovy_shift_range_fov_only_default": self.DEFAULT_FOVY_SHIFT_RANGE,
            "camera_infos": camera_infos,
        }
        if verbose:
            print(f"[Info] Applied camera shift: {info}")
        return info


class VisualHelper:
    LIGHT_PARAM_KEYS = (
        "light_active",
        "light_directional",
        "light_pos",
        "light_dir",
        "light_ambient",
        "light_diffuse",
        "light_specular",
        "light_cutoff",
        "light_exponent",
        "light_attenuation",
        "light_castshadow",
    )

    HEADLIGHT_PARAM_KEYS = ("active", "ambient", "diffuse", "specular")
    RAINBOW_CYCLE_STEPS = 90
    RAINBOW_SATURATION = 0.9
    RAINBOW_VALUE = 1.0

    def __init__(
        self,
        sim_model: MjModel,
        random_seed: int = 42,
    ):
        self.model = sim_model
        self.random_seed = random_seed
        self.original_params_snap = self.snapshot_params(self.model)
        self.dynamic_light_base_snap: Optional[Dict[str, Any]] = None
        self.dynamic_rainbow_start_hue: Optional[float] = None
        self._dynamic_rng = np.random.RandomState(self.random_seed)

        self.color_light_helper = ColorLightHelper(
            sim_model=self.model,
            random_seed=self.random_seed,
            original_params_snap=self.original_params_snap,
        )
        self.darken_helper = DarkenHelper(
            sim_model=self.model,
            random_seed=self.random_seed,
            original_params_snap=self.original_params_snap,
        )
        self.texture_helper = TextureHelper(
            sim_model=self.model,
            random_seed=self.random_seed,
            original_params_snap=self.original_params_snap,
        )
        self.camera_shift_helper = CameraShiftHelper(
            sim_model=self.model,
            random_seed=self.random_seed,
        )

    @classmethod
    def snapshot_params(cls, model: MjModel) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for key in cls.LIGHT_PARAM_KEYS:
            if hasattr(model, key):
                snap[key] = np.array(getattr(model, key)).copy()

        if hasattr(model, "vis") and hasattr(model.vis, "headlight"):
            head = model.vis.headlight
            head_snap = {}
            for key in cls.HEADLIGHT_PARAM_KEYS:
                if hasattr(head, key):
                    head_snap[key] = np.array(getattr(head, key)).copy()
            if head_snap:
                snap["headlight"] = head_snap
        return snap

    @classmethod
    def restore_params(cls, model: MjModel, snap: Dict[str, Any]):
        for key in cls.LIGHT_PARAM_KEYS:
            if key in snap and hasattr(model, key):
                getattr(model, key)[:] = snap[key]

        if "headlight" in snap and hasattr(model, "vis") and hasattr(model.vis, "headlight"):
            head = model.vis.headlight
            for key, value in snap["headlight"].items():
                if hasattr(head, key):
                    try:
                        getattr(head, key)[:] = value
                    except Exception:
                        setattr(head, key, value)

    def refresh_model_reference(
        self,
        sim_model: MjModel,
        refresh_snapshot: bool = True,
    ):
        self.model = sim_model
        if refresh_snapshot:
            self.original_params_snap = self.snapshot_params(self.model)

        for helper in (self.color_light_helper, self.darken_helper, self.texture_helper, self.camera_shift_helper):
            helper.model = self.model
            if refresh_snapshot:
                if hasattr(helper, "original_params_snap"):
                    helper.original_params_snap = self.original_params_snap
        self.clear_dynamic_light_base()
        self.clear_camera_shift_base()

    def restore_original_params(self, model: Optional[MjModel] = None):
        model = self.model if model is None else model
        self.camera_shift_helper.restore_camera_base(model=model)
        self.restore_params(model, self.original_params_snap)
        self.clear_dynamic_light_base()
        self.clear_camera_shift_base()

    @staticmethod
    def _compute_rainbow_rgb(
        step_idx: int,
        cycle_steps: int,
        saturation: float,
        value: float,
        start_hue: float = 0.0,
    ) -> np.ndarray:
        cycle = max(int(cycle_steps), 1)
        hue = (float(start_hue) + float(step_idx % cycle) / float(cycle)) % 1.0
        sat = float(np.clip(saturation, 0.0, 1.0))
        val = float(np.clip(value, 0.0, 1.0))
        return np.asarray(colorsys.hsv_to_rgb(hue, sat, val), dtype=np.float64)

    def clear_dynamic_light_base(self):
        self.dynamic_light_base_snap = None
        self.dynamic_rainbow_start_hue = None

    def clear_camera_shift_base(self):
        self.camera_shift_helper.clear_camera_base()

    def capture_dynamic_light_base(self, model: Optional[MjModel] = None):
        model = self.model if model is None else model
        self.dynamic_light_base_snap = self.snapshot_params(model)
        self.dynamic_rainbow_start_hue = float(self._dynamic_rng.uniform(0.0, 1.0))

    def apply_dynamic_rainbow_light(
        self,
        model: Optional[MjModel] = None,
        step_idx: int = 0,
        cycle_steps: Optional[int] = None,
        saturation: Optional[float] = None,
        value: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        if self.dynamic_light_base_snap is None:
            self.capture_dynamic_light_base(model=model)

        current_step = int(step_idx)
        cycle = self.RAINBOW_CYCLE_STEPS if cycle_steps is None else int(cycle_steps)
        sat = self.RAINBOW_SATURATION if saturation is None else float(saturation)
        val = self.RAINBOW_VALUE if value is None else float(value)
        start_hue = 0.0 if self.dynamic_rainbow_start_hue is None else float(self.dynamic_rainbow_start_hue)
        color_rgb = self._compute_rainbow_rgb(
            step_idx=current_step,
            cycle_steps=cycle,
            saturation=sat,
            value=val,
            start_hue=start_hue,
        )

        self.restore_params(model, self.dynamic_light_base_snap)
        color_info = self.apply_color_light(
            model=model,
            color_rgb=color_rgb,
            preserve_luminance=False,
            diffuse_scale=(1.0, 1.0),
            specular_scale=(1.0, 1.0),
            ambient_scale=(1.0, 1.0),
            affect_headlight=True,
            headlight_scale=(1.0, 1.0),
            base_from_current=True,
            verbose=verbose,
        )
        return {
            "step_idx": current_step,
            "cycle_steps": max(int(cycle), 1),
            "saturation": sat,
            "value": val,
            "start_hue": start_hue,
            "color_rgb": color_info.get("color_rgb"),
            "color_info": color_info,
        }

    def capture_camera_shift_base(
        self,
        model: Optional[MjModel] = None,
        camera_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return self.camera_shift_helper.capture_camera_base(
            model=model,
            camera_names=camera_names,
        )

    def apply_camera_shift(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        camera_names: Optional[Sequence[str]] = None,
        pos_shift_range: Optional[Sequence[Sequence[float]]] = None,
        rot_shift_range_deg: Optional[Sequence[Sequence[float]]] = None,
        fovy_shift_range: Optional[Sequence[float]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        return self.camera_shift_helper.apply_camera_shift(
            model=model,
            rng=rng,
            camera_names=camera_names,
            pos_shift_range=pos_shift_range,
            rot_shift_range_deg=rot_shift_range_deg,
            fovy_shift_range=fovy_shift_range,
            verbose=verbose,
        )

    def apply_color_light(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        color_rgb: Optional[Sequence[float]] = None,
        color_range: Tuple[float, float] = (0.25, 1.0),
        preserve_luminance: bool = False,
        diffuse_scale: Tuple[float, float] = (1.2, 2.0),
        specular_scale: Tuple[float, float] = (1.0, 1.8),
        ambient_scale: Tuple[float, float] = (0.8, 1.4),
        affect_headlight: bool = True,
        headlight_scale: Tuple[float, float] = (1.0, 1.6),
        base_from_current: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        return self.color_light_helper.apply_color_light(
            model=model,
            rng=rng,
            color_rgb=color_rgb,
            color_range=color_range,
            preserve_luminance=preserve_luminance,
            diffuse_scale=diffuse_scale,
            specular_scale=specular_scale,
            ambient_scale=ambient_scale,
            affect_headlight=affect_headlight,
            headlight_scale=headlight_scale,
            base_from_current=base_from_current,
            verbose=verbose,
        )

    def apply_single_flashlight(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        table_center_xyz: Sequence[float] = (0.0, 0.0, 0.9),
        table_half_size_xy: Sequence[float] = (0.03, 0.03),
        light_height_range: Tuple[float, float] = (1.6, 1.9),
        cutoff_range: Tuple[float, float] = (25.0, 45.0),
        exponent_range: Tuple[float, float] = (35.0, 50.0),
        attenuation: Sequence[float] = (1.0, 0.0, 0.08),
        diffuse_intensity: Tuple[float, float] = (2.0, 2.6),
        specular_intensity: Tuple[float, float] = (0.05, 0.18),
        ambient_intensity: Tuple[float, float] = (0.5, 0.6),
        other_diffuse_scale: Tuple[float, float] = (0.35, 0.55),
        other_specular_scale: Tuple[float, float] = (0.35, 0.45),
        other_ambient_scale: Tuple[float, float] = (0.35, 0.55),
        headlight_scale: Tuple[float, float] = (0.25, 0.35),
        selected_light_id: int = 0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        return self.darken_helper.apply_single_flashlight(
            model=model,
            rng=rng,
            table_center_xyz=table_center_xyz,
            table_half_size_xy=table_half_size_xy,
            light_height_range=light_height_range,
            cutoff_range=cutoff_range,
            exponent_range=exponent_range,
            attenuation=attenuation,
            diffuse_intensity=diffuse_intensity,
            specular_intensity=specular_intensity,
            ambient_intensity=ambient_intensity,
            other_diffuse_scale=other_diffuse_scale,
            other_specular_scale=other_specular_scale,
            other_ambient_scale=other_ambient_scale,
            headlight_scale=headlight_scale,
            selected_light_id=selected_light_id,
            verbose=verbose,
        )

    def apply_color_and_darken(
        self,
        model: Optional[MjModel] = None,
        rng: Optional[np.random.RandomState] = None,
        darken: bool = False,
        selected_light_id: Optional[int] = None,
        color_light: bool = False,
        table_center_xyz: Optional[Sequence[float]] = (0.0, 0.0, 0.9),
        table_half_size_xy: Optional[Sequence[float]] = (0.03, 0.03),
        color_rgb: Optional[Sequence[float]] = None,
        color_range: Tuple[float, float] = (0.25, 1.0),
        preserve_luminance: bool = False,
        diffuse_scale: Tuple[float, float] = (1.2, 2.0),
        specular_scale: Tuple[float, float] = (1.0, 1.8),
        ambient_scale: Tuple[float, float] = (0.8, 1.4),
        affect_headlight: bool = True,
        headlight_scale: Tuple[float, float] = (1.0, 1.6),
        verbose: bool = False,
    ) -> Dict[str, Any]:
        model = self.model if model is None else model
        center_xyz = (0.0, 0.0, 0.9) if table_center_xyz is None else table_center_xyz
        half_size_xy = (0.03, 0.03) if table_half_size_xy is None else table_half_size_xy
        use_flashlight = bool(darken) or (selected_light_id is not None)
        sid = 0 if selected_light_id is None else selected_light_id

        flashlight_info = None
        color_info = None

        if use_flashlight:
            flashlight_info = self.apply_single_flashlight(
                model=model,
                rng=rng,
                table_center_xyz=center_xyz,
                table_half_size_xy=half_size_xy,
                selected_light_id=sid,
                verbose=verbose,
            )

        if color_light:
            color_info = self.apply_color_light(
                model=model,
                rng=rng,
                color_rgb=color_rgb,
                color_range=color_range,
                preserve_luminance=preserve_luminance,
                diffuse_scale=diffuse_scale,
                specular_scale=specular_scale,
                ambient_scale=ambient_scale,
                affect_headlight=affect_headlight,
                headlight_scale=headlight_scale,
                base_from_current=use_flashlight,
                verbose=verbose,
            )

        return {
            "applied": use_flashlight or bool(color_light),
            "darken": bool(darken),
            "selected_light_id": sid if use_flashlight else None,
            "color_light": bool(color_light),
            "flashlight_info": flashlight_info,
            "color_info": color_info,
        }

    def apply_table_texture_via_xml_wrapper(
        self,
        texture_dir: str,
        env: Optional[Any] = None,
        model: Optional[MjModel] = None,
        sim: Optional[Any] = None,
        rng: Optional[np.random.RandomState] = None,
        recursive: bool = True,
        valid_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        preferred_material_names: Sequence[str] = ("desk", "table_texture", "table_mat"),
        preferred_texture_names: Sequence[str] = ("tex-desk", "tex-table", "textable"),
        allow_auto_detect: bool = True,
        table_keywords: Sequence[str] = TextureHelper.DEFAULT_TABLE_KEYWORDS,
        exclude_keywords: Sequence[str] = TextureHelper.DEFAULT_EXCLUDE_KEYWORDS,
        apply_to_all_keyword_textures: bool = True,
        reset_env: bool = True,
        return_xml_string: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        info = self.texture_helper.apply_table_texture_via_xml_wrapper(
            texture_dir=texture_dir,
            env=env,
            model=model,
            sim=sim,
            rng=rng,
            recursive=recursive,
            valid_exts=valid_exts,
            preferred_material_names=preferred_material_names,
            preferred_texture_names=preferred_texture_names,
            allow_auto_detect=allow_auto_detect,
            table_keywords=table_keywords,
            exclude_keywords=exclude_keywords,
            apply_to_all_keyword_textures=apply_to_all_keyword_textures,
            reset_env=reset_env,
            return_xml_string=return_xml_string,
            verbose=verbose,
        )

        if info.get("reset_applied", False):
            sim_after = TextureHelper._find_sim_from_env(env)
            if sim_after is not None and getattr(sim_after, "model", None) is not None:
                self.refresh_model_reference(sim_after.model, refresh_snapshot=True)
                info["model_reference_refreshed"] = True
            else:
                info["model_reference_refreshed"] = False
        return info
