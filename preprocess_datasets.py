#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import logging
import json
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd
import librosa
import torchaudio
from tqdm import tqdm

# Set up logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------
# Helper Functions for Audio Preprocessing
# ------------------------------

def preprocess_audio_librosa(file_path, target_sr=16000, n_mels=128, fmax=8000):
    """
    Loads an audio file using librosa, resamples it to target_sr,
    and computes a mel spectrogram in decibels.
    Returns: (y, sr, mel_db) or (None, None, None) on failure.
    """
    try:
        logging.info("Loading audio with librosa: %s", file_path)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return y, sr, mel_db
    except Exception as e:
        logging.error("Error processing audio %s: %s", file_path, e)
        return None, None, None

def preprocess_audio_torchaudio(file_path, target_sr=44100, n_mels=128, fmax=8000):
    """
    Loads an audio file using torchaudio (for Maestro to preserve fidelity), resamples if needed,
    and computes a mel spectrogram using librosa.
    Returns: (y, sr, mel_db) or (None, None, None) on failure.
    """
    try:
        logging.info("Loading Maestro audio with torchaudio: %s", file_path)
        waveform, sr = torchaudio.load(file_path)
        y = waveform.squeeze().numpy()
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return y, sr, mel_db
    except Exception as e:
        logging.error("Error processing Maestro audio %s: %s", file_path, e)
        return None, None, None


# ------------------------------
# NSynth Processing
# ------------------------------

def load_nsynth_json(nsynth_subset_folder):
    """
    Loads the NSynth JSON metadata (e.g. examples.json) from a given subset folder.
    Returns a dictionary mapping file identifiers to metadata.
    """
    json_path = os.path.join(nsynth_subset_folder, "examples.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                nsynth_json = json.load(f)
            logging.info("Loaded NSynth JSON metadata from: %s", json_path)
            return nsynth_json
        except Exception as e:
            logging.error("Error loading NSynth JSON metadata: %s", e)
            return {}
    else:
        logging.warning("NSynth JSON metadata file not found in %s", nsynth_subset_folder)
        return {}

def process_single_nsynth(file, nsynth_metadata):
    y, sr, mel_db = preprocess_audio_librosa(file, target_sr=16000, n_mels=128, fmax=8000)
    sample = {"file_path": file, "sr": sr, "mel_db": mel_db}
    key = Path(file).stem  # e.g., "bass_synthetic_068-049-025"
    if key in nsynth_metadata:
        sample["metadata"] = nsynth_metadata[key]
    return sample if y is not None else None

def process_nsynth_subset(nsynth_subset_folder, save_folder, max_workers=None):
    """
    Processes one NSynth subset (e.g., nsynth-train, nsynth-valid, nsynth-test).
    Saves the processed data as an NPZ file named after the subset.
    """
    audio_files = glob.glob(os.path.join(nsynth_subset_folder, "**", "*.wav"), recursive=True)
    if not audio_files:
        logging.warning("No NSynth audio files found in folder: %s", nsynth_subset_folder)
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    nsynth_metadata = load_nsynth_json(nsynth_subset_folder)
    processed_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_nsynth, file, nsynth_metadata): file for file in audio_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"NSynth {Path(nsynth_subset_folder).name}"):
            result = future.result()
            if result is not None:
                processed_list.append(result)
    logging.info("Processed %d files from NSynth subset: %s", len(processed_list), Path(nsynth_subset_folder).name)
    os.makedirs(save_folder, exist_ok=True)
    subset_name = Path(nsynth_subset_folder).name
    nsynth_save_path = os.path.join(save_folder, f"{subset_name}_processed.npz")
    np.savez(nsynth_save_path, data=processed_list)
    logging.info("Saved NSynth processed data to: %s", nsynth_save_path)

def process_all_nsynth(nsynth_root_folder, save_folder):
    """
    Processes all NSynth subsets found in nsynth_root_folder (e.g., nsynth-train, nsynth-valid, nsynth-test).
    """
    subsets = [f for f in os.listdir(nsynth_root_folder) if os.path.isdir(os.path.join(nsynth_root_folder, f))]
    for subset in subsets:
        subset_folder = os.path.join(nsynth_root_folder, subset)
        process_nsynth_subset(subset_folder, save_folder)


# ------------------------------
# Maestro Processing (Merging CSV and JSON)
# ------------------------------

def load_maestro_metadata_csv(metadata_csv_path):
    """
    Loads Maestro metadata from a CSV file.
    Expected columns: canonical_composer, canonical_title, audio_filename, year, etc.
    """
    try:
        df = pd.read_csv(metadata_csv_path)
        logging.info("Loaded Maestro CSV metadata with %d entries.", len(df))
        return df
    except Exception as e:
        logging.error("Error loading Maestro CSV metadata from %s: %s", metadata_csv_path, e)
        return None

def load_maestro_json(maestro_folder):
    """
    Loads the Maestro JSON metadata (e.g., maestrov3.json) from the maestro_folder.
    Returns a dictionary mapping keys (e.g., relative audio file path) to additional metadata.
    """
    json_path = os.path.join(maestro_folder, "maestrov3.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                maestro_json = json.load(f)
            logging.info("Loaded Maestro JSON metadata from: %s", json_path)
            return maestro_json
        except Exception as e:
            logging.error("Error loading Maestro JSON metadata: %s", e)
            return {}
    else:
        logging.warning("Maestro JSON metadata file not found in %s", maestro_folder)
        return {}

def process_single_maestro(args):
    file, maestro_folder = args
    y, sr, mel_db = preprocess_audio_torchaudio(file, target_sr=44100, n_mels=128, fmax=8000)
    return y, sr, mel_db, file

def process_maestro_dataset_parallel(maestro_folder, metadata_csv_filename, save_folder, max_workers=None):
    csv_path = os.path.join(maestro_folder, metadata_csv_filename)
    df = load_maestro_metadata_csv(csv_path)
    if df is None:
        logging.error("Failed to load Maestro CSV metadata.")
        return

    # Load additional JSON metadata.
    maestro_json = load_maestro_json(maestro_folder)

    files_to_process = []
    for idx, row in df.iterrows():
        audio_filename = row.get("audio_filename")
        if pd.isnull(audio_filename):
            logging.warning("Row %d missing audio filename.", idx)
            continue
        file_path = os.path.join(maestro_folder, audio_filename)
        if not os.path.exists(file_path):
            logging.warning("Audio file not found: %s", file_path)
            continue
        files_to_process.append((file_path, maestro_folder))
    
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    processed_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_maestro, args): args[0] for args in files_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Maestro files"):
            result = future.result()
            if result[0] is not None:
                file_path = result[3]
                rel_path = os.path.relpath(file_path, maestro_folder)
                try:
                    csv_row = df[df["audio_filename"] == rel_path].iloc[0].to_dict()
                except Exception as e:
                    logging.warning("Could not find CSV metadata for %s: %s", rel_path, e)
                    csv_row = {}
                extra_meta = maestro_json.get(rel_path, {})
                merged_meta = {**csv_row, **extra_meta}
                sample = {
                    "file_path": file_path,
                    "metadata": merged_meta,
                    "sr": result[1],
                    "mel_db": result[2]
                }
                processed_list.append(sample)
    logging.info("Parallel processed %d Maestro audio files.", len(processed_list))
    os.makedirs(save_folder, exist_ok=True)
    maestro_save_path = os.path.join(save_folder, "maestro_processed.npz")
    np.savez(maestro_save_path, data=processed_list)
    logging.info("Saved Maestro processed data to: %s", maestro_save_path)


# ------------------------------
# Main Function
# ------------------------------

def main():
    current_dir = Path(__file__).parent

    # Define raw data folders.
    # For NSynth, assume three subsets: nsynth-train, nsynth-valid, nsynth-test are inside the "nsynth" folder.
    nsynth_root_folder = current_dir / "nsynth"
    # For Maestro, the folder contains subfolders (e.g., by year) and metadata files are in the root.
    maestro_raw_folder = current_dir / "maestro"

    # Define output folder for processed data.
    processed_folder = current_dir / "processed_data"
    nsynth_processed_folder = processed_folder / "nsynth"
    maestro_processed_folder = processed_folder / "maestro"

    # Process NSynth subsets.
    if nsynth_root_folder.exists():
        logging.info("Processing all NSynth subsets from: %s", nsynth_root_folder)
        process_all_nsynth(str(nsynth_root_folder), str(nsynth_processed_folder))
    else:
        logging.warning("NSynth raw folder not found: %s", nsynth_root_folder)
    
    # Process Maestro dataset.
    if maestro_raw_folder.exists():
        logging.info("Processing Maestro dataset from: %s", maestro_raw_folder)
        # Assuming the metadata CSV is named "maestrov3.csv" in the maestro folder.
        process_maestro_dataset_parallel(str(maestro_raw_folder), "maestrov3.csv", str(maestro_processed_folder))
    else:
        logging.warning("Maestro raw folder not found: %s", maestro_raw_folder)

if __name__ == "__main__":
    main()
