import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: torch.nn.Module,
        output_activation: torch.nn.Module,
        n_hidden_layers: int,
        hidden_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        self.layers.append(activation)

        for _ in range(self.n_hidden_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation)

        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        self.layers.append(output_activation)

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeformNet(torch.nn.Module):
    def __init__(
        self,
        h: int,
        n_hidden_layers: int,
        output_activation: torch.nn.Module = torch.nn.Identity(),
    ):
        super().__init__()
        self.feature_model = MLP(
            2,
            h,
            torch.nn.LeakyReLU(0.01),
            torch.nn.Identity(),
            n_hidden_layers,
            h,
        )

        self.offset_model = MLP(
            h + 2,
            2,
            torch.nn.LeakyReLU(0.01),
            output_activation,
            n_hidden_layers,
            h,
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g + self.offset_model(torch.cat([self.feature_model(g), g], dim=-1))


class WrapNet(torch.nn.Module):
    def __init__(
        self,
        h: int,
        n_hidden_layers: int,
    ):
        super().__init__()
        self.feature_model = MLP(
            2,
            h,
            torch.nn.LeakyReLU(0.01),
            torch.nn.Identity(),
            n_hidden_layers,
            h,
        )

        self.output_model = MLP(
            h + 2,
            6,
            torch.nn.LeakyReLU(0.01),
            torch.nn.Identity(),
            n_hidden_layers,
            h,
        )

    def forward(self, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.output_model(torch.cat([self.feature_model(Q), Q], dim=-1))
        p, p_n = X[:, :3], X[:, 3:]

        return p, p_n


class CutNet(torch.nn.Module):
    def __init__(self, h: int, n_hidden_layers: int):
        super().__init__()
        self.feature_model = MLP(
            3,
            h,
            torch.nn.LeakyReLU(0.01),
            torch.nn.Identity(),
            n_hidden_layers,
            h,
        )

        self.offset_model = MLP(
            h + 3,
            3,
            torch.nn.LeakyReLU(0.01),
            torch.nn.Identity(),
            n_hidden_layers,
            h,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return p + self.offset_model(torch.cat([self.feature_model(p), p], dim=-1))


class UnwrapNet(torch.nn.Module):
    def __init__(
        self,
        h: int,
        n_hidden_layers: int,
        output_activation: torch.nn.Module = torch.nn.Identity(),
    ):
        super().__init__()
        self.model = MLP(
            3,
            2,
            torch.nn.LeakyReLU(0.01),
            output_activation,
            n_hidden_layers,
            h,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.model(p)


if __name__ == "__main__":
    g = torch.rand(100, 2)
    Md = DeformNet(128, 3)
    Mw = WrapNet(128, 3)
    Mc = CutNet(128, 3)
    Mu = UnwrapNet(128, 3)

    q = Md(g)
    p, p_n = Mw(q)
    p_cut = Mc(p)
    q_cycle = Mu(p_cut)

    print(f"{q.shape=}")
    print(f"{p.shape=}")
    print(f"{p_n.shape=}")
    print(f"{p_cut.shape=}")
    print(f"{q_cycle.shape=}")
