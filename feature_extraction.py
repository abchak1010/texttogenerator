#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Define your audio file path – update to your actual file
    file_path = r"C:\ai_music\song2.wav"

    if not os.path.exists(file_path):
        logging.error("Error: File not found at %s", file_path)
        sys.exit(1)
    else:
        logging.info("File exists. Proceeding to load...")

    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        logging.info("Audio loaded with sample rate: %d", sr)

        # Plot and save the waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform of the Song")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        waveform_plot_path = "waveform_plot.png"
        plt.savefig(waveform_plot_path)
        plt.close()
        logging.info("Waveform plot saved as '%s'", waveform_plot_path)

        # Compute the Mel spectrogram and convert to decibels
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Plot and save the Mel spectrogram
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap="coolwarm")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        mel_spectrogram_path = "mel_spectrogram.png"
        plt.savefig(mel_spectrogram_path)
        plt.close()
        logging.info("Mel spectrogram plot saved as '%s'", mel_spectrogram_path)

        # Extract tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times_list = [bt.item() for bt in beat_times]

        logging.info("Estimated Tempo: %.2f BPM", tempo)
        logging.info("First 10 Beat Times (s):")
        for i, bt in enumerate(beat_times_list[:10]):
            logging.info("Beat %d: %.4f seconds", i + 1, bt)

        # Save beat times to a text file
        beat_times_file = "beat_times.txt"
        with open(beat_times_file, "w") as f:
            for bt in beat_times_list:
                f.write(f"{bt:.4f}\n")
        logging.info("Beat times saved to '%s'.", beat_times_file)

    except Exception as e:
        logging.error("Error during processing: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
