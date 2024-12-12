import math
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

        if mesh_name == "tablier_lod.ply":
            transform = Mesh.read_transform_from_txt(
                to_absolute_path(f"./data/frame.txt")
            )
            mesh = Mesh.from_file(path)
            mesh.apply_transform(transform)
            return mesh
        return Mesh.from_file(path)

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


def rescale_to_1(uv_points: torch.Tensor) -> torch.Tensor:
    min_u = torch.min(uv_points[:, 0]).unsqueeze(0)
    min_v = torch.min(uv_points[:, 1]).unsqueeze(0)

    min_uv = torch.cat([min_u, min_v], dim=0)
    uv_points = uv_points - min_uv
    return uv_points / torch.max(uv_points)


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

    Md = DeformNet(h, n_hidden_layers, torch.nn.Identity()).to("cuda")
    Mw = WrapNet(h, n_hidden_layers).to("cuda")
    Mc = CutNet(h, n_hidden_layers).to("cuda")
    Mu = UnwrapNet(h, n_hidden_layers, torch.nn.Identity()).to("cuda")

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters, 1e-5)

    loss_wrap = WrappingLoss()
    loss_unwrap = UnwrappingLoss(
        eps=(2 / (math.ceil(math.sqrt(n_samples)) - 1)) * 0.25, K=8
    )
    loss_cycle = CycleLoss()
    loss_diff = DiffLoss(alpha_stretch=1.0)
    t = trange(n_iters, leave=True)
    for it in t:
        optimizer.zero_grad()

        barycenters = mesh.random_barycentric(n_samples)
        faces = mesh.sample_faces(n_samples)
        p = mesh.sample_from_barycenters(faces, barycenters)
        g = torch.rand((n_samples, 2), device="cuda")
        p_n = mesh.sample_smooth_normals(faces, barycenters)

        p_cut = Mc(p)  # cut
        q = Mu(p_cut)  # unwrap
        p_c, p_n_c = Mw(q)  # wrap

        q_h = Md(g)  # deform
        p_h, p_n_h = Mw(q_h)  # wrap
        p_h_cut = Mc(p_h)  # cut

        q_h_c = Mu(p_h_cut)  # unwrap

        q_normalized = uv_bounding_box_normalization(q).squeeze()
        q_h_normalized = uv_bounding_box_normalization(q_h).squeeze()
        q_h_c_normalized = uv_bounding_box_normalization(q_h_c).squeeze()

        l_diff = loss_diff(p_c, q)
        l_wrap = loss_wrap(p_h, p)
        l_unwrap = (
            loss_unwrap(q_normalized)
            + loss_unwrap(q_h_normalized)
            + loss_unwrap(q_h_c_normalized)
        )
        l_cycle_p, l_cycle_q, l_cycle_n = loss_cycle(p, p_c, q_h, q_h_c, p_n, p_n_c)

        loss: torch.Tensor = (
            l_wrap
            + 0.01 * l_unwrap
            + 0.01 * l_cycle_p
            + 0.01 * l_cycle_q
            + 0.005 * l_cycle_n
            + 0.01 * l_diff
        )
        # TODO: loss coeffs
        if log:
            writer.add_scalar("l_wrap", l_wrap.item(), global_step=it)
            writer.add_scalar("l_unwrap", l_unwrap.item(), global_step=it)
            writer.add_scalar("l_cycle_p", l_cycle_p.item(), global_step=it)
            writer.add_scalar("l_cycle_q", l_cycle_q.item(), global_step=it)
            writer.add_scalar("l_cycle_n", l_cycle_n.item(), global_step=it)
            writer.add_scalar("l_diff", l_diff.item(), global_step=it)
            writer.add_scalar("weighted loss", loss.item(), global_step=it)

        t.set_description(f"loss={loss.item():.5f}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        if it > 0 and (it % cfg.checkpoint_freq == 0 or it == n_iters - 1) and log:
            with torch.no_grad():
                pcd = mesh.vertices
                cut_mesh = Mc(pcd)
                Mesh.write_pcd(cut_mesh, f"./cut_mesh_{it}.ply")
                mesh_uvs = Mu(cut_mesh)
                mesh_uvs = rescale_to_1(mesh_uvs)
                mesh.set_uvs(mesh_uvs)
                mesh.generate_normal_map(512, f"./normal_map_{it}.png", k=3)
                generate_checkerboard_pcd_uv(pcd, mesh_uvs, it)


if __name__ == "__main__":
    train()
