import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks

from config import (
    BASE_DIR,
    FS_HZ,
    WINDOW_SAMPLES as W,
    WINDOW_OVERLAP as O,
    FALL_SEGMENT_SEC,
    MARGIN_RATIO,
    MAX_NONFALL_WINDOWS_PER_FILE,
    ALL_FEATS,
    RAW_WINDOWS_CSV,
    CLEAN_WINDOWS_CSV,
)


b, a = butter(2, [0.7 / (FS_HZ / 2), 3.0 / (FS_HZ / 2)], btype="band")


def ppg_to_bpm(ppg_raw: np.ndarray) -> float:
    ppg = ppg_raw / 1023.0
    ppg_f = filtfilt(b, a, ppg)
    peaks, _ = find_peaks(ppg_f, distance=int(0.3 * FS_HZ))
    if len(peaks) < 2:
        return np.nan
    rr = np.diff(peaks) / FS_HZ
    return 60.0 / rr.mean()


def load_trial_full(mat_path: str) -> pd.DataFrame:
    d = loadmat(mat_path)
    series = {k: d[k].ravel() for k in ALL_FEATS}
    L = min(len(v) for v in series.values())
    for k in series:
        series[k] = series[k][:L]
    df = pd.DataFrame(series)
    df["t_rel"] = np.arange(len(df)) / FS_HZ
    return df


def fall_segment_6s_around_90pct(
    df: pd.DataFrame,
    seg_sec: float = FALL_SEGMENT_SEC,
    margin_ratio: float = MARGIN_RATIO,
):
    acc_mag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    peak = acc_mag.max()
    thr = (1.0 - margin_ratio) * peak

    idx = np.where(acc_mag >= thr)[0]
    if len(idx) == 0:
        center = int(np.argmax(acc_mag))
    else:
        si_90 = int(idx[0])
        ei_90 = int(idx[-1]) + 1
        center = (si_90 + ei_90) // 2

    half_len = int((seg_sec * FS_HZ) / 2)
    si = max(0, center - half_len)
    ei = min(len(df), center + half_len)

    if ei - si < W:
        si = max(0, center - W // 2)
        ei = min(len(df), si + W)

    return si, ei


def segment_trial(df: pd.DataFrame, si: int, ei: int, w: int, o: int):
    windows = []
    n = 1
    while True:
        j = si + (w - o) * (n - 1)
        if j < ei:
            low = max(0, j - w)
            high = j
            win = df.iloc[low:high].reset_index(drop=True)
            if len(win) == w:
                windows.append(win)
            n += 1
        else:
            break
    return windows


def build_all_windows() -> pd.DataFrame:
    all_subject_dfs = []
    global_win_id = 0

    subjects = sorted(
        name
        for name in os.listdir(BASE_DIR)
        if name.startswith("subject_")
        and os.path.isdir(os.path.join(BASE_DIR, name))
    )

    for subject in subjects:
        print("Processing", subject)
        all_windows, labels, meta, bpm_list = [], [], [], []

        for cls in ["fall", "non-fall"]:
            folder = os.path.join(BASE_DIR, subject, cls)
            if not os.path.isdir(folder):
                continue

            for fname in os.listdir(folder):
                if not fname.endswith(".mat"):
                    continue
                path = os.path.join(folder, fname)
                df = load_trial_full(path)

                if cls == "fall":
                    si, ei = fall_segment_6s_around_90pct(
                        df, seg_sec=FALL_SEGMENT_SEC, margin_ratio=MARGIN_RATIO
                    )
                    win_list = segment_trial(df, si, ei, W, O)
                    selected_windows = win_list
                else:
                    si, ei = 0, len(df)
                    win_list = segment_trial(df, si, ei, W, O)
                    if len(win_list) > MAX_NONFALL_WINDOWS_PER_FILE:
                        idx = np.random.choice(
                            len(win_list),
                            size=MAX_NONFALL_WINDOWS_PER_FILE,
                            replace=False,
                        )
                        idx = np.sort(idx)
                        selected_windows = [win_list[i] for i in idx]
                    else:
                        selected_windows = win_list

                for wdf in selected_windows:
                    all_windows.append(wdf[ALL_FEATS + ["t_rel"]].to_numpy())
                    labels.append(1 if cls == "fall" else 0)
                    meta.append({"subject": subject, "file": fname, "class": cls})
                    bpm_list.append(ppg_to_bpm(wdf["heart"].values))

        if not all_windows:
            continue

        X = np.stack(all_windows, axis=0)
        y = np.array(labels)
        meta_df = pd.DataFrame(meta)
        bpm_arr = np.array(bpm_list)

        feat_cols = ALL_FEATS + ["t_rel"]
        num_windows, _, _ = X.shape
        rows = []
        for i in range(num_windows):
            wdf = pd.DataFrame(X[i], columns=feat_cols)
            wdf["win_id"] = global_win_id
            wdf["sample_in_win"] = np.arange(W)
            for col in meta_df.columns:
                wdf[col] = meta_df.loc[i, col]
            wdf["label"] = y[i]
            wdf["bpm"] = bpm_arr[i]
            rows.append(wdf)
            global_win_id += 1

        subj_df = pd.concat(rows, ignore_index=True)
        all_subject_dfs.append(subj_df)

    full_df = pd.concat(all_subject_dfs, ignore_index=True)
    return full_df


def save_raw_and_clean_windows():
    full_df = build_all_windows()
    full_df.to_csv(RAW_WINDOWS_CSV, index=False)
    print("Saved:", RAW_WINDOWS_CSV)
    print("Rows in full_df:", len(full_df))
    print("Unique win_id:", full_df["win_id"].nunique())
    print(full_df.groupby("win_id")["label"].agg(["count", "mean"]).head())

    win_labels = (
        full_df.groupby("win_id")["label"]
        .agg(count="count", mean_label="mean")
        .reset_index()
    )
    unique, counts = np.unique(win_labels["mean_label"].astype(int), return_counts=True)
    print("\nWindow-level label counts (0=non-fall, 1=fall):")
    print(dict(zip(unique, counts)))

    bad_win_ids = (
        full_df[["win_id", "bpm"]]
        .drop_duplicates()
        .query("bpm.isna()")["win_id"]
        .tolist()
    )
    print("Num windows with NaN bpm:", len(bad_win_ids))
    clean_df = full_df[~full_df["win_id"].isin(bad_win_ids)].copy()

    clean_df.to_csv(CLEAN_WINDOWS_CSV, index=False)
    print("Saved:", CLEAN_WINDOWS_CSV)
