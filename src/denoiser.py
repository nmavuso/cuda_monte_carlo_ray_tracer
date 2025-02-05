import torch
import torch.nn as nn
import numpy as np

# Define a very basic CNN model for denoising
class BasicDenoiser(nn.Module):
    def __init__(self):
        super(BasicDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# Dummy function to simulate loading and using a trained denoiser
class Denoiser:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BasicDenoiser().to(self.device)
        
        # In a real scenario, load the trained model
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()

    def denoise(self, input_image, output_image, width, height):
        # Convert the input image (float array) to a PyTorch tensor
        input_tensor = torch.from_numpy(input_image.reshape((1, 3, height, width))).float().to(self.device)

        # Perform inference (dummy operation for now)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            #output_tensor = input_tensor  # Replace with actual inference

        # Convert the output tensor back to a NumPy array
        output_array = output_tensor.cpu().numpy().reshape((-1,))

        # Copy the denoised data to the output_image array
        output_image[:output_array.shape[0]] = output_array
