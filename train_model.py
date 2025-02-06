#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------
# Dataset for Preprocessed Data
# ------------------------------

class ProcessedAudioDataset(Dataset):
    def __init__(self, npz_file):
        loaded = np.load(npz_file, allow_pickle=True)
        self.data = loaded["data"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx].item()  # Each sample is a dict
        # Keep mel spectrogram in its 2D shape: [n_mels, time] (e.g., [128, 128])
        mel_db = sample["mel_db"]
        return torch.tensor(mel_db, dtype=torch.float32), sample

# ------------------------------
# Advanced LSTM-based Generator with Dropout
# ------------------------------

class LSTMTextToMusicGenerator(nn.Module):
    def __init__(self, noise_dim=100, text_emb_dim=768, hidden_dim=256, num_layers=2, seq_length=128, feature_dim=128, dropout_prob=0.3):
        super(LSTMTextToMusicGenerator, self).__init__()
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.input_proj = nn.Linear(noise_dim + text_emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, text_embedding):
        # Concatenate noise and text embedding: [batch, noise_dim+text_emb_dim]
        x = torch.cat([noise, text_embedding], dim=1)
        x = self.input_proj(x)  # [batch, hidden_dim]
        x = self.dropout(x)
        # Repeat vector across the sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)  # [batch, seq_length, hidden_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_length, hidden_dim]
        output = self.output_proj(lstm_out)  # [batch, seq_length, feature_dim]
        output = self.tanh(output)
        return output  # Output represents a mel spectrogram sequence

# ------------------------------
# Loss Functions
# ------------------------------

def get_loss_function(loss_type="L1"):
    if loss_type == "L1":
        return nn.L1Loss()
    elif loss_type == "MSE":
        return nn.MSELoss()
    elif loss_type == "Perceptual":
        class PerceptualLoss(nn.Module):
            def __init__(self):
                super(PerceptualLoss, self).__init__()
                self.mse = nn.MSELoss()
            def forward(self, real, generated):
                real_stft = torch.stft(real, n_fft=2048, hop_length=512, return_complex=True).abs()
                gen_stft = torch.stft(generated, n_fft=2048, hop_length=512, return_complex=True).abs()
                return self.mse(real_stft, gen_stft)
        return PerceptualLoss()
    else:
        raise ValueError("Unknown loss type")

# ------------------------------
# Text Encoder Setup (using DistilBERT)
# ------------------------------

def get_text_embedding(text, tokenizer, text_model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = text_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.to(device)

# ------------------------------
# Training Pipeline with Early Stopping
# ------------------------------

def train_model(dataset, text_prompt="A calm piano melody", epochs=50, batch_size=16, learning_rate=1e-4, loss_type="L1", patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    # Split dataset into train and validation (80/20 split)
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    text_model.eval()
    
    generator = LSTMTextToMusicGenerator(noise_dim=100, text_emb_dim=768, hidden_dim=256, num_layers=2, seq_length=128, feature_dim=128, dropout_prob=0.3).to(device)
    
    # Using weight decay for regularization.
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = get_loss_function(loss_type)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0.0
        for i, (target_mel, sample_info) in enumerate(train_loader):
            if target_mel is None or target_mel.shape[0] == 0:
                continue
            target_mel = target_mel.to(device)  # Shape: [batch, n_mels, time] (assumed [128, 128])
            current_batch = target_mel.shape[0]
            noise = torch.randn(current_batch, 100, device=device)
            with torch.no_grad():
                text_emb = get_text_embedding(text_prompt, tokenizer, text_model, device)
                text_emb = text_emb.repeat(current_batch, 1)
            output = generator(noise, text_emb)  # [batch, seq_length, feature_dim]
            loss = criterion(output, target_mel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 10 == 0:
                logging.info("Epoch [%d/%d], Batch [%d/%d], Loss: %.4f", epoch+1, epochs, i, len(train_loader), loss.item())
        avg_train_loss = epoch_loss / len(train_loader)
        logging.info("Epoch [%d/%d] Average Training Loss: %.4f", epoch+1, epochs, avg_train_loss)
        
        # Validation Phase
        generator.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for target_mel, _ in val_loader:
                target_mel = target_mel.to(device)
                current_batch = target_mel.shape[0]
                noise = torch.randn(current_batch, 100, device=device)
                text_emb = get_text_embedding(text_prompt, tokenizer, text_model, device)
                text_emb = text_emb.repeat(current_batch, 1)
                output = generator(noise, text_emb)
                val_loss = criterion(output, target_mel)
                val_loss_total += val_loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        logging.info("Epoch [%d/%d] Validation Loss: %.4f", epoch+1, epochs, avg_val_loss)
        
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model checkpoint
            best_checkpoint = generator.state_dict()
            logging.info("Validation loss improved; saving checkpoint.")
        else:
            epochs_no_improve += 1
            logging.info("No improvement in validation loss for %d epoch(s).", epochs_no_improve)
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                break
                
    # Save the best model checkpoint
    checkpoints_folder = Path("checkpoints")
    checkpoints_folder.mkdir(exist_ok=True)
    checkpoint_path = checkpoints_folder / "text_to_music_generator.pt"
    torch.save(best_checkpoint, checkpoint_path)
    logging.info("Model saved to: %s", checkpoint_path)

# ------------------------------
# Main Function
# ------------------------------

def main():
    current_dir = Path(__file__).parent
    processed_folder = current_dir / "processed_data" / "maestro"
    maestro_npz = processed_folder / "maestro_processed.npz"
    if not maestro_npz.exists():
        logging.error("Processed Maestro data not found at: %s", maestro_npz)
        return
    logging.info("Loading processed Maestro dataset from: %s", maestro_npz)
    dataset = ProcessedAudioDataset(str(maestro_npz))
    logging.info("Loaded dataset with %d samples", len(dataset))
    train_model(dataset, text_prompt="A calm piano melody", epochs=50, batch_size=16, loss_type="Perceptual", patience=5)

if __name__ == "__main__":
    main()
