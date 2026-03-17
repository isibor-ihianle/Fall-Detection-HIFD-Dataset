import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import antropy as ant

from config import FS_HZ, TARGET_COL


def simple_slope(x, y):
    if len(x) < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return float(num / den) if den != 0 else 0.0


def extract_features(window: pd.DataFrame) -> dict:
    feats = {}
    acc_cols = ["accX_m/s2", "accY_m/s2", "accZ_m/s2"]
    gyr_cols = ["gyrX_dps", "gyrY_dps", "gyrZ_dps"]

    for col in acc_cols + gyr_cols:
        v = window[col]
        feats[f"{col}_mean"] = v.mean()
        feats[f"{col}_std"] = v.std()
        feats[f"{col}_max"] = v.max()
        feats[f"{col}_min"] = v.min()
        feats[f"{col}_kurtosis"] = v.kurtosis()
        feats[f"{col}_skew"] = v.skew()
        feats[f"{col}_iqr"] = np.percentile(v, 75) - np.percentile(v, 25)

    acc_mag = np.sqrt((window[acc_cols] ** 2).sum(axis=1))
    gyr_mag = np.sqrt((window[gyr_cols] ** 2).sum(axis=1))

    feats["acc_mag_area"] = trapezoid(acc_mag)
    feats["gyr_mag_area"] = trapezoid(gyr_mag)
    feats["acc_energy"] = np.sum(acc_mag ** 2) / len(acc_mag)

    def zcr(series):
        v = np.asarray(series)
        return ((v[:-1] * v[1:]) < 0).sum()

    feats["acc_mag_zcr"] = zcr(acc_mag - acc_mag.mean())

    idx = np.arange(len(window))
    feats["accZ_slope"] = simple_slope(idx, window["accZ_m/s2"].values)
    feats["acc_mag_slope"] = simple_slope(idx, acc_mag.values)

    if acc_mag.var() == 0:
        feats["acc_mag_entropy"] = 0.0
    else:
        feats["acc_mag_entropy"] = float(
            ant.spectral_entropy(
                acc_mag.values, sf=FS_HZ, method="fft", normalize=True
            )
        )

    if gyr_mag.var() == 0:
        feats["gyr_mag_entropy"] = 0.0
    else:
        feats["gyr_mag_entropy"] = float(
            ant.spectral_entropy(
                gyr_mag.values, sf=FS_HZ, method="fft", normalize=True
            )
        )

    hr = window["heartRate_bpm"]
    feats["hr_mean"] = hr.mean()
    feats["hr_std"] = hr.std()
    feats["hr_min"] = hr.min()
    feats["hr_max"] = hr.max()
    feats["hr_kurtosis"] = hr.kurtosis()
    feats["hr_skew"] = hr.skew()
    feats["hr_iqr"] = np.percentile(hr, 75) - np.percentile(hr, 25)
    feats["hr_slope"] = simple_slope(idx, hr.values)

    hr_diffs = np.diff(hr.values)
    if len(hr_diffs) > 0:
        feats["hr_rmssd"] = np.sqrt(np.mean(hr_diffs ** 2))
        feats["hr_mad"] = np.mean(np.abs(hr_diffs))
    else:
        feats["hr_rmssd"] = 0.0
        feats["hr_mad"] = 0.0

    if hr.var() == 0:
        feats["hr_spectral_entropy"] = 0.0
    else:
        feats["hr_spectral_entropy"] = float(
            ant.spectral_entropy(
                hr.values, sf=FS_HZ, method="fft", normalize=True
            )
        )

    return feats


def build_windows_from_win_id(df: pd.DataFrame):
    rows = []
    for win_id, wdf in df.groupby("win_id"):
        feats = extract_features(wdf)
        lbl_series = wdf[TARGET_COL].astype(int)
        lbl_value = int(lbl_series.mode().iloc[0])

        feats["label"] = lbl_value
        feats["subject"] = wdf["subject"].iloc[0]
        feats["file"] = wdf["file"].iloc[0]
        feats["win_id"] = win_id
        rows.append(feats)

    Xy = pd.DataFrame(rows)
    y = Xy["label"].values
    X = Xy.drop(columns=["label", "subject", "file", "win_id"])
    return X, y, Xy
