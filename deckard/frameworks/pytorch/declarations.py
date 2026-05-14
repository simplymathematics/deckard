# TinyNet: Minimal torch model for binary classification
class TinyNet(nn.Module if nn else object):
    """A minimal torch model for binary classification (2-layer MLP)."""

    def __init__(self, input_dim=10, hidden_dim=16, output_dim=2):
        if nn is None:
            raise ImportError("TinyNet requires torch to be installed.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
