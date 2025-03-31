import torch.nn as nn

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.encoder = base_model(num_classes=out_dim)  # Pass num_classes to base_model

        # Manually set dim_mlp as the output size of the encoder
        dim_mlp = self.encoder.fc[0].in_features  # Get the in_features of the first Linear layer in fc

        # Add MLP projection head
        self.encoder.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder.fc
        )

    def forward(self, x):
        return self.encoder(x)
