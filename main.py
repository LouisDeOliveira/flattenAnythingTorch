import os

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from loss import CycleLoss, DiffLoss, UnwrappingLoss, WrappingLoss
from mesh import Mesh
from model import CutNet, DeformNet, UnwrapNet, WrapNet
from utils import generate_checkerboard_pcd_uv


def load_mesh(mesh_name: str) -> Mesh:
    if os.path.exists(path := to_absolute_path(f"./data/{mesh_name}")):
        if "tablier" in mesh_name:
            transform = Mesh.read_transform_from_txt(
                to_absolute_path("./data/frame.txt")
            )
            mesh = Mesh.from_file(path)
            mesh.apply_transform(transform)
            return mesh
        return Mesh.from_file(path)

    print("mesh not found, using nefertiti as default...")
    return Mesh.from_file(to_absolute_path("./data/nefertiti.obj"))


def uv_bounding_box_normalization(uv_points: torch.Tensor) -> torch.Tensor:
    centroids = ((uv_points.min(dim=0)[0] + uv_points.max(dim=0)[0]) / 2).unsqueeze(0)
    uv_points = uv_points - centroids
    max_d = (uv_points**2).sum(dim=-1).sqrt().max()
    return uv_points / max_d


def rescale_to_1(uv_points: torch.Tensor) -> torch.Tensor:
    min_uv = torch.stack([uv_points[:, 0].min(), uv_points[:, 1].min()])
    uv_points = uv_points - min_uv
    return uv_points / uv_points.max()


@hydra.main(config_path="./config", config_name="FAconf", version_base="1.3")
def train(cfg: DictConfig):
    print(cfg)

    n_iters = cfg.n_steps
    n_samples = cfg.n_samples
    log = cfg.log

    if log:
        writer = SummaryWriter(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )

    Md = DeformNet().to("cuda")
    Mw = WrapNet().to("cuda")
    Mc = CutNet().to("cuda")
    Mu = UnwrapNet().to("cuda")

    mesh = load_mesh(cfg.dataset)
    mesh.compute_normalization_params()
    mesh.normalize_mesh()

    optimizer = torch.optim.AdamW(
        list(Md.parameters())
        + list(Mw.parameters())
        + list(Mc.parameters())
        + list(Mu.parameters()),
        lr=1e-3,
        weight_decay=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    loss_wrap = WrappingLoss(chunk_size=cfg.get("chamfer_chunk_size", None))
    loss_unwrap = UnwrappingLoss(K=8, max_pts=cfg.get("unwrap_max_pts", None))
    loss_cycle = CycleLoss()
    loss_diff = DiffLoss()

    t = trange(n_iters, leave=True)
    for it in t:
        optimizer.zero_grad()

        barycenters = mesh.random_barycentric(n_samples)
        faces = mesh.sample_faces(n_samples)
        p = mesh.sample_from_barycenters(faces, barycenters)
        p_n = mesh.sample_smooth_normals(faces, barycenters)

        # G is uniformly sampled in [-1, 1]^2 per paper Section 3.1.1
        g = torch.rand((n_samples, 2), device="cuda") * 2 - 1

        # ── 3D → 2D → 3D branch ──────────────────────────────────────────
        p_cut = Mc(p)
        q = Mu(p_cut)
        p_c, p_n_c = Mw(q)  # full graph — cycle loss trains Mc, Mu, Mw

        # ── 2D → 3D → 2D branch ──────────────────────────────────────────
        q_h = Md(g)
        p_h, _ = Mw(q_h)  # full graph — wrap loss trains Md, Mw
        p_h_cut = Mc(p_h)
        q_h_c = Mu(p_h_cut)  # full graph — cycle loss trains Md, Mw, Mc, Mu

        # ── Extra WrapNet forward with leaf UV tensors for DiffLoss ──────
        # Detach so the Jacobian is purely of WrapNet (no higher-order paths).
        # Optionally subsample: 3×autograd.grad passes per Jacobian are expensive.
        n_diff = cfg.get("n_diff_samples", None)
        if n_diff is not None and n_diff < n_samples:
            diff_idx = torch.randperm(n_samples, device="cuda")[:n_diff]
            q_jac = q[diff_idx].detach().requires_grad_(True)
            q_h_jac = q_h[diff_idx].detach().requires_grad_(True)
        else:
            q_jac = q.detach().requires_grad_(True)
            q_h_jac = q_h.detach().requires_grad_(True)
        p_c_jac, _ = Mw(q_jac)
        p_h_jac, _ = Mw(q_h_jac)

        q_normalized = uv_bounding_box_normalization(q)
        q_h_normalized = uv_bounding_box_normalization(q_h)
        q_h_c_normalized = uv_bounding_box_normalization(q_h_c)

        # ── Losses ───────────────────────────────────────────────────────
        l_wrap = loss_wrap(p_h, p)

        l_unwrap = (
            loss_unwrap(q_normalized)
            + loss_unwrap(q_h_normalized)
            + loss_unwrap(q_h_c_normalized)
        )

        l_cycle_p, l_cycle_q, l_cycle_n = loss_cycle(p, p_c, q_h, q_h_c, p_n, p_n_c)
        l_cycle = l_cycle_p + l_cycle_q + l_cycle_n

        # Conformality: Jacobian of WrapNet at Q and Q_hat (paper Eq. 12-13).
        # Warm up over first 20% of training so the mapping has time to form
        # before the Jacobian regulariser kicks in (avoids early instability).
        conf_weight = 0.01 * min(1.0, it / (0.2 * n_iters))
        l_conf = loss_diff(p_c_jac, q_jac, p_h_jac, q_h_jac)

        # Weights from paper Appendix A.1
        loss: torch.Tensor = (
            1.0 * l_wrap + 0.01 * l_unwrap + 0.01 * l_cycle + conf_weight * l_conf
        )

        if log:
            writer.add_scalar("l_wrap", l_wrap.item(), global_step=it)
            writer.add_scalar("l_unwrap", l_unwrap.item(), global_step=it)
            writer.add_scalar("l_cycle_p", l_cycle_p.item(), global_step=it)
            writer.add_scalar("l_cycle_q", l_cycle_q.item(), global_step=it)
            writer.add_scalar("l_cycle_n", l_cycle_n.item(), global_step=it)
            writer.add_scalar("l_conf", l_conf.item(), global_step=it)
            writer.add_scalar("conf_weight", conf_weight, global_step=it)
            writer.add_scalar("weighted_loss", loss.item(), global_step=it)

        t.set_description(f"loss={loss.item():.5f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(Md.parameters())
            + list(Mw.parameters())
            + list(Mc.parameters())
            + list(Mu.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        if it > 0 and (it % cfg.checkpoint_freq == 0 or it == n_iters - 1) and log:
            with torch.no_grad():
                pcd = mesh.vertices
                cut_mesh = Mc(pcd)
                mesh_uvs = Mu(cut_mesh)
                mesh_uvs = rescale_to_1(mesh_uvs)
                mesh.set_uvs(mesh_uvs)
                mesh.generate_normal_map(512, f"./normal_map_{it}.png", k=1)

    with torch.no_grad():
        pcd = mesh.vertices
        cut_mesh = Mc(pcd)
        mesh_uvs = Mu(cut_mesh)
        mesh_uvs = rescale_to_1(mesh_uvs)
        mesh.set_uvs(mesh_uvs)
        mesh.to_file("./result.ply")


if __name__ == "__main__":
    train()
