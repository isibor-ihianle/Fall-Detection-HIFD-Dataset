import os


FS_HZ = 50
WINDOW_SAMPLES = 100      
WINDOW_OVERLAP = 50        


FALL_SEGMENT_SEC = 6.0
MARGIN_RATIO = 0.10
MAX_NONFALL_WINDOWS_PER_FILE = 30


BASE_DIR = r"HR_IMU_falldetection_dataset-master"
OUT_DIR = r"Fall Dataset"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_WINDOWS_CSV = os.path.join(
    OUT_DIR, "all_subjects_windows_all_feats_bpm_6s_90pct.csv"
)
CLEAN_WINDOWS_CSV = os.path.join(
    OUT_DIR, "all_subjects_windows_all_feats_bpm_6s_90pct_clean.csv"
)

TARGET_COL = "label"
RANDOM_STATE = 42

ALL_FEATS = ["w", "x", "y", "z", "droll", "dpitch", "dyaw", "ax", "ay", "az", "heart"]
