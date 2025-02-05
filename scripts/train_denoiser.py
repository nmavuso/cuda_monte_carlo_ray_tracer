import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.denoiser import BasicDenoiser

# Dummy dataset for training (replace with your actual data loading)
class NoisyImageDataset(Dataset):
    def __init__(self, num_samples=100, width=800, height=400):
        self.num_samples = num_samples
        self.width = width
        self.height = height

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random "clean" image
        clean_image = np.random.rand(3, self.height, self.width).astype(np.float32)
        # Add some noise
        noisy_image = clean_image + 0.1 * np.random.randn(3, self.height, self.width).astype(np.float32)
        return noisy_image, clean_image

def train(model_dir, epochs=10, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicDenoiser().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = NoisyImageDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), f"{model_dir}/denoiser_model.pth")
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a basic denoising model.")
    parser.add_argument("--model_dir", type=str, default="../models", help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    args = parser.parse_args()

    train(args.model_dir, args.epochs, args.batch_size)
