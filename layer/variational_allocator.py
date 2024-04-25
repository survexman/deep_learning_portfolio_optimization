# cvxpy==1.3.2
import cvxpy as cp
# cvxpylayers==0.1.6
from cvxpylayers.torch import CvxpyLayer
# torch==2.1.0.dev20230814+cu121
import torch


class VariationalAllocator(torch.nn.Module):
    """

    Parameters
    ----------
    n_assets : int
        Number of assets.

    max_weight : float
        A float between (0, 1] representing the maximum weight per asset.

    """

    def __init__(self, n_assets, max_weight = 1):
        super().__init__()

        x = cp.Parameter(n_assets)
        w = cp.Variable(n_assets)
        obj = x @ w + cp.sum(cp.entr(w))
        cons = [
            cp.sum(w) == 1.,
            w <= max_weight
        ]
        prob = cp.Problem(cp.Maximize(obj), cons)
        self.layer = CvxpyLayer(prob, [x], [w])

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        y = self.layer(x)[0]
        return y
