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
        l_n = -torch.mean(
            torch.sum(p_n * p_n_c, dim=-1)
            / (torch.norm(p_n, dim=-1) * torch.norm(p_n_c, dim=-1))
        )

        return (
            l_p,
            l_q,
            l_n,
        )


class ConformalLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass
        # TODO implement diff loss :)
