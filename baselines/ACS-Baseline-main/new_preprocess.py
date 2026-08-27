import os
import random

import numpy as np
from scipy import signal
from tqdm import tqdm


def preprocess_ecg(data, target_len=1000, orig_fs=500, target_fs=100):
    """
    单条数据处理核心逻辑
    data shape: (leads, length) 或 (length, leads)
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"输入数据维度必须为 2 维，当前维度: {data.ndim}")

    swapped = False
    if data.shape[0] > data.shape[1]:
        data = data.T
        swapped = True

    num_samples = int(data.shape[1] * target_fs / orig_fs)
    num_samples = max(num_samples, 1)
    resampled_data = signal.resample(data, num_samples, axis=1)

    if resampled_data.shape[1] >= target_len:
        processed_data = resampled_data[:, :target_len]
    else:
        pad_width = target_len - resampled_data.shape[1]
        processed_data = np.pad(resampled_data, ((0, 0), (0, pad_width)), mode="constant")

    for i in range(processed_data.shape[0]):
        d_min = processed_data[i].min()
        d_max = processed_data[i].max()
        if d_max - d_min != 0:
            processed_data[i] = (processed_data[i] - d_min) / (d_max - d_min)
        else:
            processed_data[i] = 0

    if swapped:
        processed_data = processed_data.T

    return processed_data


def random_check_saved_files(out_path, num_checks=5):
    if not os.path.exists(out_path):
        print(f"Warning: output path does not exist: {out_path}")
        return

    files = [f for f in os.listdir(out_path) if f.endswith(".npy")]
    if not files:
        print(f"Warning: No .npy files found in {out_path} for checking")
        return

    sample_count = min(num_checks, len(files))
    sample_files = random.sample(files, sample_count)

    print(f"\nRandom shape check for {os.path.basename(out_path)} ({sample_count} files)")
    for file_name in sample_files:
        file_path = os.path.join(out_path, file_name)
        try:
            arr = np.load(file_path)
            print(f"{file_name}: shape={arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")


def run_preprocessing():
    base_dir = r"F:\new_20260812\MVMS-net-main"
    input_root = os.path.join(base_dir, "data", "ptbxl", "new_records")
    output_root = os.path.join(base_dir, "data", "have_done_records")

    sub_folders = ["train", "test"]

    for folder in sub_folders:
        in_path = os.path.join(input_root, folder)
        out_path = os.path.join(output_root, folder)

        if not os.path.exists(in_path):
            print(f"Warning: input path does not exist: {in_path}")
            continue

        os.makedirs(out_path, exist_ok=True)

        files = [f for f in os.listdir(in_path) if f.endswith(".npy")]
        if not files:
            print(f"Warning: No .npy files found in {in_path}")
            continue

        print(f"Processing {folder} set ({len(files)} files)...")

        for file_name in tqdm(files):
            try:
                file_in = os.path.join(in_path, file_name)
                data = np.load(file_in)
                processed_data = preprocess_ecg(data)
                file_out = os.path.join(out_path, file_name)
                np.save(file_out, processed_data.astype(np.float32))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        random_check_saved_files(out_path, num_checks=5)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
