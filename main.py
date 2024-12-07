import math
import os

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from loss import CycleLoss, UnwrappingLoss, WrappingLoss
from mesh import Mesh
from model import CutNet, DeformNet, UnwrapNet, WrapNet
from utils import generate_checkerboard_pcd_uv

AVAILABLE_MESHES = ["nefertiti", "bunny", "armadillo", "beetle", "hand", "dragon"]


def load_mesh(mesh_name: str) -> Mesh:

    if mesh_name in AVAILABLE_MESHES:
        return Mesh.from_file(to_absolute_path(f"./data/{mesh_name}.obj"))

    else:
        print("mesh not found, using nefertiti as default...")
        return Mesh.from_file(to_absolute_path(f"./data/nefertiti.obj"))


def uv_bounding_box_normalization(uv_points: torch.Tensor) -> torch.Tensor:
    # uv_points: [B, N, 2]
    centroids = ((uv_points.min(dim=0)[0] + uv_points.max(dim=0)[0]) / 2).unsqueeze(
        0
    )  # [B, 1, 2]
    uv_points = uv_points - centroids
    max_d = (uv_points**2).sum(dim=-1).sqrt().max(dim=-1)[0]  # [B, 1, 1]
    uv_points = uv_points / max_d

    return uv_points


@hydra.main(config_path="./config", config_name="FAconf", version_base="1.3")
def train(cfg: DictConfig):
    print(cfg)
    # set_all_seeds(42)

    h = cfg.models.hidden_size
    n_hidden_layers = cfg.models.n_hidden_layers

    n_iters = cfg.n_steps
    n_samples = cfg.n_samples

    log = cfg.log
    if log:
        writer = SummaryWriter(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )

    Md = DeformNet(h, n_hidden_layers).to("cuda")
    Mw = WrapNet(h, n_hidden_layers).to("cuda")
    Mc = CutNet(h, n_hidden_layers).to("cuda")
    Mu = UnwrapNet(h, n_hidden_layers).to("cuda")

    mesh = load_mesh(cfg.dataset)
    mesh.normalize_mesh()

    optimizer = torch.optim.AdamW(
        list(Md.parameters())
        + list(Mw.parameters())
        + list(Mc.parameters())
        + list(Mu.parameters()),
        lr=1e-3,
        weight_decay=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters, 1e-5)

    loss_wrap = WrappingLoss()
    loss_unwrap = UnwrappingLoss(
        eps=(2 / (math.ceil(math.sqrt(n_samples)) - 1)) * 0.25, K=8
    )
    loss_cycle = CycleLoss()

    t = trange(n_iters, leave=True)
    for it in t:
        optimizer.zero_grad()

        barycenters = mesh.random_barycentric(n_samples)
        faces = mesh.sample_faces(n_samples)
        p = mesh.sample_from_barycenters(faces, barycenters)
        g = torch.rand((n_samples, 2), device="cuda")
        # p_n = mesh.sample_face_normals(faces)
        p_n = mesh.sample_smooth_normals(faces, barycenters)

        p_cut = Mc(p)  # cut
        q = Mu(p_cut)  # unwrap
        p_c, p_n_c = Mw(q)  # wrap

        q_h = Md(g)  # deform
        p_h, p_n_h = Mw(q_h)  # wrap
        p_h_cut = Mc(p_h)  # cut

        q_h_c = Mu(p_h_cut)  # unwrap

        q = uv_bounding_box_normalization(q).squeeze()
        q_h = uv_bounding_box_normalization(q_h).squeeze()
        q_h_c = uv_bounding_box_normalization(q_h_c).squeeze()

        l_wrap = loss_wrap(p_h, p)
        l_unwrap = loss_unwrap(q) + loss_unwrap(q_h) + loss_unwrap(q_h_c)
        l_cycle_p, l_cycle_q, l_cycle_n = loss_cycle(p, p_c, q_h, q_h_c, p_n, p_n_c)

        loss: torch.Tensor = (
            l_wrap
            + 0.01 * l_unwrap
            + 0.01 * l_cycle_p
            + 0.01 * l_cycle_q
            + 0.005 * l_cycle_n
        )

        if log:
            writer.add_scalar("l_wrap", l_wrap.item(), global_step=it)
            writer.add_scalar("l_unwrap", l_unwrap.item(), global_step=it)
            writer.add_scalar("l_cycle_p", l_cycle_p.item(), global_step=it)
            writer.add_scalar("l_cycle_q", l_cycle_q.item(), global_step=it)
            writer.add_scalar("l_cycle_n", l_cycle_n.item(), global_step=it)
            writer.add_scalar("weighted loss", loss.item(), global_step=it)

        t.set_description(f"loss={loss.item():.5f}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        if it > 0 and (it % cfg.checkpoint_freq == 0 or it == n_iters - 1) and log:
            with torch.no_grad():
                pcd = mesh.sample_mesh(5000)
                cut_mesh = Mc(pcd)
                Mesh.write_pcd(cut_mesh, f"./cut_mesh_{it}.ply")
                mesh_uvs = Mu(cut_mesh)
                generate_checkerboard_pcd_uv(pcd, mesh_uvs, it)


if __name__ == "__main__":
    train()
