import numpy as np
import torch
import trimesh

from mesh import Mesh


def set_all_seeds(seed):
    """
    Set all seeds for reproducibility.

    :param seed: The seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_checkerboard_mesh_uv(
    mesh: Mesh,
    vertex_uvs: torch.Tensor,
    step: int,
):
    bw_checkerboard = torch.zeros(3, 500, 500, device="cuda")
    for i in range(10):
        for j in range(10):
            color = (
                torch.tensor([0.0, 0.0, 0.0], device="cuda")
                if (i + j) % 2 == 0
                else torch.tensor([1.0, 1.0, 1.0], device="cuda")
            )
            bw_checkerboard[:, i * 50 : (i + 1) * 50, j * 50 : (j + 1) * 50] = (
                color.unsqueeze(-1).unsqueeze(-1).broadcast_to((3, 50, 50))
            )

    vertex_colors = (
        torch.nn.functional.grid_sample(
            bw_checkerboard.unsqueeze(0),
            vertex_uvs.unsqueeze(0).unsqueeze(-2),
            align_corners=False,
        )
        .squeeze()
        .permute(1, 0)
    )
    uv_vertex_colors = torch.clamp(
        torch.cat(
            [
                0.5 * (vertex_uvs + 1.0),
                torch.zeros((vertex_uvs.shape[0], 1), device=vertex_uvs.device),
            ],
            dim=-1,
        ),
        0.0,
        1.0,
    )
    mesh.to("cpu")
    trimesh_mesh = trimesh.Trimesh(
        mesh.vertices.numpy(),
        mesh.faces.numpy(),
        vertex_colors=vertex_colors.cpu().numpy(),
    )
    trimesh_mesh.export(f"./test_checker_{step}_bs.obj")
    trimesh.Trimesh(
        mesh.vertices.numpy(),
        mesh.faces.numpy(),
        vertex_colors=uv_vertex_colors.cpu().numpy(),
    ).export(f"./test_uv_{step}_bs.obj")

    mesh.to("cuda")


def generate_checkerboard_pcd_uv(
    points: torch.Tensor,
    vertex_uvs: torch.Tensor,
    step: int,
):
    bw_checkerboard = torch.zeros(3, 800, 800, device="cuda")
    for i in range(20):
        for j in range(20):
            color = (
                torch.tensor([0.0, 0.0, 0.0], device="cuda")
                if (i + j) % 2 == 0
                else torch.tensor([1.0, 1.0, 1.0], device="cuda")
            )
            bw_checkerboard[:, i * 40 : (i + 1) * 40, j * 40 : (j + 1) * 40] = (
                color.unsqueeze(-1).unsqueeze(-1).broadcast_to((3, 40, 40))
            )

    vertex_colors = (
        torch.nn.functional.grid_sample(
            bw_checkerboard.unsqueeze(0),
            2 * vertex_uvs.unsqueeze(0).unsqueeze(-2) - 1.0,
            align_corners=False,
        )
        .squeeze()
        .permute(1, 0)
    )
    uv_vertex_colors = torch.clamp(
        torch.cat(
            [
                0.5 * (vertex_uvs + 1.0),
                torch.zeros((vertex_uvs.shape[0], 1), device=vertex_uvs.device),
            ],
            dim=-1,
        ),
        0.0,
        1.0,
    )

    Mesh.write_pcd(points, f"./test_checker_{step}.obj", vertex_colors)
    Mesh.write_pcd(points, f"./test_uv_{step}.obj", uv_vertex_colors)
