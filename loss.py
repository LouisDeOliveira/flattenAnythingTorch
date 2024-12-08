import torch


class UnwrappingLoss(torch.nn.Module):
    def __init__(self, eps: float, K: int):
        super().__init__()
        self.eps = torch.tensor(eps, device="cuda")
        self.K = K

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(
            q, q
        )  # Here cdist and topk doesnt no scale well with N at all -> can use a CUDA KNN instead but isok :)
        knn = torch.topk(dists, self.K + 1, dim=-1, largest=False).values[
            :, 1:
        ]  # start at 1 to avoid itself with dist 0
        return torch.mean(torch.sum(torch.relu(self.eps - knn), dim=1))


class WrappingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_hat: torch.Tensor, p: torch.Tensor):
        dists = torch.cdist(p_hat, p).pow(2)  # Also doesnt scale well for large N :)

        return torch.mean(torch.min(dists, dim=0)[0]) + torch.mean(
            torch.min(dists, dim=1)[0]
        )


class CycleLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        p: torch.Tensor,
        p_c: torch.Tensor,
        q_h: torch.Tensor,
        q_h_c: torch.Tensor,
        p_n: torch.Tensor,
        p_n_c: torch.Tensor,
    ):
        l_p = torch.nn.functional.l1_loss(p, p_c)
        l_q = torch.nn.functional.l1_loss(q_h, q_h_c)
        l_n = torch.mean(
            1.0
            - torch.sum(p_n * p_n_c, dim=-1)
            / (torch.norm(p_n, dim=-1) * torch.norm(p_n_c, dim=-1))
        )

        return (
            l_p,
            l_q,
            l_n,
        )


class DiffLoss(torch.nn.Module):
    def __init__(self, alpha_conf: float = 1.0, alpha_stretch: float = 1.0) -> None:
        super().__init__()
        self.alpha_conf = alpha_conf
        self.alpha_stretch = alpha_stretch

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        dfx = torch.autograd.grad(
            p[:, 0],
            q,
            torch.ones_like(p[:, 0], dtype=torch.float, device=p.device),
            create_graph=True,
        )[0]
        dfy = torch.autograd.grad(
            p[:, 0],
            q,
            torch.ones_like(p[:, 0], dtype=torch.float, device=p.device),
            create_graph=True,
        )[0]
        dfz = torch.autograd.grad(
            p[:, 0],
            q,
            torch.ones_like(p[:, 0], dtype=torch.float, device=p.device),
            create_graph=True,
        )[0]

        J = torch.cat(
            [dfx.unsqueeze(-1), dfy.unsqueeze(-1), dfz.unsqueeze(-1)], dim=-1
        ).permute(0, 2, 1)

        JtJ = J.permute(0, 2, 1) @ J

        uu = JtJ[:, 0, 0]
        vv = JtJ[:, 1, 1]
        uv = JtJ[:, 0, 1]

        a = uu + vv
        b = torch.sqrt(4 * (uv**2) + (uu - vv) ** 2)

        l1 = 0.5 * (a + b)
        l2 = 0.5 * (a - b)

        l_conf = torch.nn.functional.l1_loss(l1, l2)
        l_stretch = torch.mean(torch.abs(l1 - 1.0)) + torch.mean(torch.abs(l2 - 1.0))

        return self.alpha_conf * l_conf + self.alpha_stretch * l_stretch
