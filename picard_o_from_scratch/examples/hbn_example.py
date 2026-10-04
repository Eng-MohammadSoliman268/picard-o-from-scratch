"""Example for the HBN EEGLAB dataset used during validation."""

import mne
from picard_o import PicardOConfig, fit_picard_o

EEG_PATH = "/content/drive/MyDrive/MDD/data/pre_processed/original_data/HBN/sub-NDARAC904DMU/eeg/sub-NDARAC904DMU_task-RestingState_eeg.set"

BAD_CHANNELS = ["E122", "E126", "E127", "E67", "E128", "E32", "E33", "E17"]

raw = mne.io.read_raw_eeglab(EEG_PATH, preload=True, verbose=False)
raw.drop_channels([ch for ch in BAD_CHANNELS if ch in raw.ch_names])
X = raw.get_data()

config = PicardOConfig(backend="cupy", tol=1e-7, max_iter=500)
result = fit_picard_o(X, config=config, verbose=True)

print("rank:", result.rank)
print("converged:", result.converged)
print("updates:", result.accepted_updates)
print("gradient:", result.final_gradient_norm)
print("runtime:", result.runtime_seconds)
