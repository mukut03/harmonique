import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, hidden_channels, output_channels):
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        x_flat = x.view(-1, x.size(-1))
        distances = (torch.sum(x_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(x)
        return quantized, (x - quantized).detach() + quantized
