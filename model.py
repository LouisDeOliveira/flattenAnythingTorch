import torch
import torch.nn as nn


def _make_mlp(layer_sizes: list[int], leaky_slope: float = 0.01) -> nn.Sequential:
    """Build an MLP from a list of layer widths.

    All hidden activations are LeakyReLU; the final layer has no activation
    (caller is responsible for any output activation).
    """
    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.LeakyReLU(leaky_slope))
    return nn.Sequential(*layers)


class DeformNet(nn.Module):
    """Deform-Net M_d: R^2 -> R^2, offset-based UV deformation.

    xi_d':  4-layer MLP [2, 512, 512, 512, 64]
    xi_d'': 4-layer MLP [66, 512, 512, 512, 2]
    (paper Appendix A.1)
    """

    def __init__(self, h: int = 512, latent: int = 64):
        super().__init__()
        self.feature_net = _make_mlp([2, h, h, h, latent])
        self.offset_net = _make_mlp([latent + 2, h, h, h, 2])

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(g)
        offset = self.offset_net(torch.cat([feat, g], dim=-1))
        return g + offset


class WrapNet(nn.Module):
    """Wrap-Net M_w: R^2 -> R^3 x R^3 (position + normal).

    xi_w':  4-layer MLP [2, 512, 512, 512, 64]
    xi_w'': 4-layer MLP [66, 512, 512, 512, 6]
    (paper Appendix A.1)
    """

    def __init__(self, h: int = 512, latent: int = 64):
        super().__init__()
        self.feature_net = _make_mlp([2, h, h, h, latent])
        self.output_net = _make_mlp([latent + 2, h, h, h, 6])

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_net(q)
        out = self.output_net(torch.cat([feat, q], dim=-1))
        return out[:, :3], out[:, 3:]


class CutNet(nn.Module):
    """Cut-Net M_c: R^3 -> R^3, offset-based surface cutting.

    xi_c':  3-layer MLP [3, 512, 512, 64]
    xi_c'': 3-layer MLP [67, 512, 512, 3]
    (paper Appendix A.1)
    """

    def __init__(self, h: int = 512, latent: int = 64):
        super().__init__()
        self.feature_net = _make_mlp([3, h, h, latent])
        self.offset_net = _make_mlp([latent + 3, h, h, 3])

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(p)
        offset = self.offset_net(torch.cat([feat, p], dim=-1))
        return p + offset


class UnwrapNet(nn.Module):
    """Unwrap-Net M_u: R^3 -> R^2.

    xi_u: 3-layer MLP [3, 512, 512, 2]
    (paper Appendix A.1)
    """

    def __init__(self, h: int = 512):
        super().__init__()
        self.model = _make_mlp([3, h, h, 2])

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.model(p)


if __name__ == "__main__":
    g = torch.rand(100, 2)
    Md = DeformNet()
    Mw = WrapNet()
    Mc = CutNet()
    Mu = UnwrapNet()

    q = Md(g)
    p, p_n = Mw(q)
    p_cut = Mc(p)
    q_cycle = Mu(p_cut)

    print(f"{q.shape=}")
    print(f"{p.shape=}")
    print(f"{p_n.shape=}")
    print(f"{p_cut.shape=}")
    print(f"{q_cycle.shape=}")
