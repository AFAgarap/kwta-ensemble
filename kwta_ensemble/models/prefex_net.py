import torch


class PrefexDNN(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Sequential,
        num_classes: int,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        code_dim = encoder.layers[:-2][-1].out_features
        self.layers = torch.nn.Sequential(
            *encoder.layers[:-2],
            torch.nn.Linear(in_features=code_dim, out_features=100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=100, out_features=num_classes),
        )

        self.device = device
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.layers(features)
        return logits
