import os
import re
import glob
from tqdm import tqdm
from libero.force.hdf5_replayer import HDF5Replayer
from libero.force.hdf5_compare import compare_hdf5



OUT_DIR = "libero_force_384"

dataset_root = "/home/geyuan/code/LIBERO-FT/libero/datasets/"
dataset_subname = "libero_10"  # `libero_90`, `libero_10`
output_dir = os.path.join(dataset_root, OUT_DIR, dataset_subname)
os.makedirs(output_dir, exist_ok=True)
video_save_dir = "/home/geyuan/code/LIBERO-FT/debug/replay_videos/"  # for visualization of replayed videos

## Modify the regex as you need to filter the hdf5 files to replay.
# pat = re.compile(r"(open|close).*drawer|microwave", re.IGNORECASE)
# pat = re.compile(r"(STUDY_SCENE3).*left", re.IGNORECASE)
pat = re.compile(r".*", re.IGNORECASE)

input_hdf5_fns = sorted(glob.glob(dataset_root + dataset_subname + "/*.hdf5"))
input_hdf5_fns = [fn for fn in input_hdf5_fns if pat.search(os.path.basename(fn))]
# print("Found hdf5 files:", input_hdf5_fns)
output_hdf5_fns = [
    os.path.join(
        output_dir,
        os.path.basename(fn).replace(".hdf5", "_wrench.hdf5")
    )
    for fn in input_hdf5_fns
]
print("Output hdf5 files:", output_hdf5_fns)


total_cnt = len(input_hdf5_fns)
for idx in tqdm(range(total_cnt), desc="Replaying"):
    input_hdf5_path = input_hdf5_fns[idx]
    output_hdf5_path = output_hdf5_fns[idx]

    replayer = HDF5Replayer(
        input_hdf5_path,
        camera_hw=(384, 384),
        retry_obs_source="env",
    )
    replayer.init_env()

    replayer.replay_hdf5_by_actions(
        hdf5_save_path=output_hdf5_path,
        video_save_dir=video_save_dir,
        save_max_videos=5,
    )
    replayer.close_env()
