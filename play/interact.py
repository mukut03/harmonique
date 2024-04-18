from pynput import keyboard
from model.model import Encoder, Decoder, VectorQuantizer
import torch

def on_press(key, encoder, decoder, vector_quantizer, latent_vectors, current_position):
    try:
        if key.char == 'w':
            current_position[0] = max(0, current_position[0] - 1)
        elif key.char == 's':
            current_position[0] = min(latent_vectors.shape[0] - 1, current_position[0] + 1)
        elif key.char == 'a':
            current_position[1] = max(0, current_position[1] - 1)
        elif key.char == 'd':
            current_position[1] = min(latent_vectors.shape[1] - 1, current_position[1] + 1)

        latent_vector = latent_vectors[current_position[0], current_position[1]].unsqueeze(0).unsqueeze(0)
        generated_audio = decoder(vector_quantizer(latent_vector)[0]).squeeze(0).squeeze(0)
        print(f"Position: {current_position} - Playing generated audio")
    except AttributeError:
        print('Special key {0} pressed'.format(key))

def real_time_interaction(encoder, decoder, vector_quantizer, latent_grid_size=(5, 5)):
    latent_vectors = torch.randn(latent_grid_size[0], latent_grid_size[1], encoder.output_dim)
    current_position = [latent_grid_size[0] // 2, latent_grid_size[1] // 2]
    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, encoder, decoder, vector_quantizer, latent_vectors, current_position))
    listener.start()
    listener.join()

# Assuming models are already instantiated and trained
# real_time_interaction(encoder, decoder, vector_quantizer)
