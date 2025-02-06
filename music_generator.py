#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained DistilBERT tokenizer and model once.
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = AutoModel.from_pretrained("distilbert-base-uncased")
except Exception as e:
    logging.error("Error loading DistilBERT models: %s", e)
    raise e

def real_text_encoder(text: str) -> torch.Tensor:
    """
    Encodes the text prompt using DistilBERT and returns the [CLS] token embedding (768-dim).
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = text_model(**inputs)
        # [CLS] token embedding is the first token of the last_hidden_state.
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.squeeze(0)  # Shape: [768]
    except Exception as e:
        logging.error("Error during text encoding: %s", e)
        raise e

class SimpleMusicGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, text_emb_dim: int = 768, hidden_dim: int = 256, output_length: int = 16000):
        """
        A simple generator that takes a concatenated vector of random noise and text embedding,
        and outputs a 1D waveform (audio) of fixed length.
        """
        super(SimpleMusicGenerator, self).__init__()
        input_dim = noise_dim + text_emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_length),
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, x):
        return self.model(x)

def generate_music(prompt: str = "A calm piano melody", noise_dim: int = 100, output_length: int = 16000, output_file: str = "generated_music.wav"):
    """
    Generates a music-like waveform conditioned on a text prompt.
    Steps:
      1. Encode the prompt using a real text encoder (DistilBERT).
      2. Generate random noise.
      3. Concatenate the noise and text embedding.
      4. Run the generator network.
      5. Scale the output to int16 and save as a WAV file.
    """
    try:
        # Select device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", device)

        # Encode the prompt
        text_embedding = real_text_encoder(prompt).to(device)  # Shape: [768]
    
        # Generate random noise vector
        noise = torch.randn((1, noise_dim), device=device)  # Shape: [1, noise_dim]
    
        # Expand text embedding to batch dimension and concatenate with noise
        text_embedding = text_embedding.unsqueeze(0)  # Shape: [1, 768]
        generator_input = torch.cat([noise, text_embedding], dim=1)  # Shape: [1, noise_dim + 768]
    
        # Instantiate the generator model and move it to the device
        model = SimpleMusicGenerator(noise_dim=noise_dim, text_emb_dim=768, hidden_dim=256, output_length=output_length).to(device)
    
        # (Optional: Load pre-trained weights here)
    
        # Generate the waveform
        with torch.no_grad():
            output = model(generator_input)  # Shape: [1, output_length]
    
        # Convert output to numpy and scale to int16 range
        output_np = output.cpu().numpy()
        waveform = (output_np * 32767).astype(np.int16)
    
        # Save the waveform as a WAV file (sampling rate 16000)
        torchaudio.save(output_file, torch.tensor(waveform), 16000)
        logging.info("Music generated and saved as '%s'.", output_file)
    except Exception as e:
        logging.error("Error in generate_music: %s", e)
        raise e

if __name__ == "__main__":
    logging.info("Music Generation Script is Running!")
    generate_music(prompt="A calm piano melody", output_file="generated_music.wav")
