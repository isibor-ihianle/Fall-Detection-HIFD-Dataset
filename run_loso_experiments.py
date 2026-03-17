import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import CLEAN_WINDOWS_CSV
from features import build_windows_from_win_id
from evaluation import run_all_models


def main():
    df = pd.read_csv(CLEAN_WINDOWS_CSV)

    df = df.rename(
        columns={
            "ax": "accX_m/s2",
            "ay": "accY_m/s2",
            "az": "accZ_m/s2",
            "droll": "gyrX_dps",
            "dpitch": "gyrY_dps",
            "dyaw": "gyrZ_dps",
            "bpm": "heartRate_bpm",
        }
    )

    if "label" not in df.columns:
        if "class" in df.columns:
            df["label"] = df["class"].map({"fall": 1, "non-fall": 0}).astype(int)
        else:
            raise ValueError("No 'class' or 'label' column found in the CSV")
    else:
        if df["label"].dtype == object:
            df["label"] = df["label"].map(
                {"fall": 1, "non-fall": 0}
            ).astype(int)

    X, y, Xy = build_windows_from_win_id(df)

    print("--- WINDOW VERIFICATION ---")
    unique, counts = np.unique(y, return_counts=True)
    print("Counts:", dict(zip(unique, counts)))
    print("Total windows:", len(Xy))
    print("Windows per subject (first 10):")
    print(Xy.groupby("subject")["win_id"].count().head(10))

    if len(unique) < 2:
        print("Only one class present; aborting.")
        return

    run_all_models(X, y, Xy)


if __name__ == "__main__":
    main()
