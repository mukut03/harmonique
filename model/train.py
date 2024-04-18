#import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import Encoder, Decoder, VectorQuantizer
from data.data_preparation import AudioDataset

# Initialize model components
encoder = Encoder(input_channels=1, hidden_channels=256)
vector_quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=256)
decoder = Decoder(hidden_channels=256, output_channels=1)

# Dataset and DataLoader
dataset = AudioDataset(directory='path_to_audio')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Optimizer
params = list(encoder.parameters()) + list(vector_quantizer.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for mel_spectrogram in tqdm(dataloader):
        mel_spectrogram = mel_spectrogram.unsqueeze(1)  # Add channel dimension
        optimizer.zero_grad()
        z = encoder(mel_spectrogram)
        quantized, diff = vector_quantizer(z)
        reconstructed = decoder(quantized)
        loss = ((mel_spectrogram - reconstructed) ** 2).mean() + (diff ** 2).mean()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
