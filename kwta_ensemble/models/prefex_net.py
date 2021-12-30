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
        self.layers = torch.nn.Sequential(
            *encoder[:-1],
            torch.nn.Linear(in_features=(64 * 7 * 7), out_features=100, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=2e-1),
            torch.nn.Linear(in_features=100, out_features=10, bias=False),
        )

        self.device = device
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.layers(features)
        return logits
