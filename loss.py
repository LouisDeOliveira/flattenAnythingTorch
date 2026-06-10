import torch
import torch.nn.functional as F


class UnwrappingLoss(torch.nn.Module):
    def __init__(self, K: int = 8, max_pts: int | None = None):
        super().__init__()
        self.K = K
        self.max_pts = max_pts

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        # eps uses the full set's bounding box per paper Appendix A.1
        L_Q = (q.max(dim=0)[0] - q.min(dim=0)[0]).max()
        eps = 0.2 * L_Q / (N ** 0.5)

        if self.max_pts is not None and N > self.max_pts:
            q = q[torch.randperm(N, device=q.device)[:self.max_pts]]

        dists = torch.cdist(q, q)
        knn = torch.topk(dists, self.K + 1, dim=-1, largest=False).values[:, 1:]
        return torch.mean(torch.sum(torch.relu(eps - knn), dim=1))


class WrappingLoss(torch.nn.Module):
    def __init__(self, chunk_size: int | None = None):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, p_hat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Chamfer distance (squared). When chunk_size is set, processes rows in
        # blocks so peak memory is O(chunk × N) instead of O(N²).
        if self.chunk_size is None:
            dists = torch.cdist(p_hat, p).pow(2)
            return dists.min(dim=0)[0].mean() + dists.min(dim=1)[0].mean()

        cs = self.chunk_size
        min_01 = torch.cat([
            torch.cdist(p_hat[i:i + cs], p).pow(2).min(dim=1)[0]
            for i in range(0, len(p_hat), cs)
        ])
        min_10 = torch.cat([
            torch.cdist(p[i:i + cs], p_hat).pow(2).min(dim=1)[0]
            for i in range(0, len(p), cs)
        ])
        return min_01.mean() + min_10.mean()


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
        l_p = F.l1_loss(p, p_c)
        l_q = F.l1_loss(q_h, q_h_c)
        l_n = torch.mean(1.0 - F.cosine_similarity(p_n, p_n_c, dim=-1))
        return l_p, l_q, l_n


class DiffLoss(torch.nn.Module):
    """Conformality loss via Jacobian eigenvalues (paper Eq. 12-13).

    Computes ∂p/∂q via 3 reverse-mode VJP passes (one per output dim),
    assembling J ∈ R^{N×3×2}, then penalises |λ1 − λ2| of J^T J.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _jacobian(p3d: torch.Tensor, q2d: torch.Tensor) -> torch.Tensor:
        """Per-point Jacobian J ∈ R^{N×3×2}.

        3 reverse-mode passes, each computing one row of J for all N points.
        p3d: (N, 3) — WrapNet output connected to q2d in the graph
        q2d: (N, 2) — leaf UV tensor with requires_grad=True
        """
        ones = torch.ones(p3d.shape[0], device=p3d.device)
        rows = [
            torch.autograd.grad(
                p3d[:, i], q2d,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
            )[0]            # (N, 2)
            for i in range(3)
        ]
        return torch.stack(rows, dim=1)  # (N, 3, 2)

    @staticmethod
    def _conformality_from_jacobian(J: torch.Tensor) -> torch.Tensor:
        JtJ = J.permute(0, 2, 1) @ J    # (N, 2, 2)
        uu, vv, uv = JtJ[:, 0, 0], JtJ[:, 1, 1], JtJ[:, 0, 1]

        a = uu + vv
        disc = torch.clamp(4 * uv ** 2 + (uu - vv) ** 2, min=1e-8)
        b = torch.sqrt(disc)
        l1 = 0.5 * (a + b)
        l2 = 0.5 * (a - b)

        valid = l1.detach() > 1e-6
        if not valid.any():
            return torch.tensor(0.0, device=J.device, requires_grad=True)
        return torch.mean(torch.abs(l1[valid] - l2[valid]))

    def forward(
        self,
        p_cycle: torch.Tensor,
        q: torch.Tensor,
        p_hat: torch.Tensor,
        q_hat: torch.Tensor,
    ) -> torch.Tensor:
        J_f = self._jacobian(p_cycle, q)
        J_g = self._jacobian(p_hat, q_hat)
        return self._conformality_from_jacobian(J_f) + self._conformality_from_jacobian(J_g)
