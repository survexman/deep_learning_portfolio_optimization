# torch==2.1.0
import torch
from layer.variational_allocator import VariationalAllocator


class FullyConnectedTrivialNet(torch.nn.Module):

    def __init__(
            self,
            n_assets,
            lookback,
            p = 0,
            max_weight = 1
    ):
        super().__init__()

        n_features = n_assets * lookback

        self.dropout = torch.nn.Dropout(p = p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias = True)
        self.allocator = VariationalAllocator(n_assets = n_assets, max_weight = max_weight)

    def forward(self, x):
        n_samples, _, _, _ = x.shape
        x = x.view(n_samples, -1)  # flatten features
        x = self.dropout(x)
        x = self.linear(x)
        weights = self.allocator(x)

        return weights
