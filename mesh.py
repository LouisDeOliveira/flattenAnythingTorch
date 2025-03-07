import xml.etree.ElementTree as ET

import numpy as np
import plyfile
import torch
import trimesh
from PIL import Image


def make_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor):
    """
    Makes rotation matrices from sets of axes and angles around the axes.

    axis : [N,3]
    angle : [N, 1]
    """
    ux, uy, uz = torch.split(axis, 1, dim=-1)
    K = torch.zeros((axis.shape[0], 3, 3), device=axis.device)

    K[:, 0, 1] = -uz.squeeze()
    K[:, 0, 2] = uy.squeeze()
    K[:, 1, 0] = uz.squeeze()
    K[:, 1, 2] = -ux.squeeze()
    K[:, 2, 0] = -uy.squeeze()
    K[:, 2, 1] = ux.squeeze()

    L = torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(-1).transpose(2, 1))
    return (
        torch.cos(angle).unsqueeze(-1)
        * torch.eye(3, device=axis.device).repeat((K.shape[0], 1, 1))
        + torch.sin(angle).unsqueeze(-1) * K
        + (1 - torch.cos(angle).unsqueeze(-1)) * L
    )


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


class NormalizeLayer(torch.nn.Module):
    """
    Layer with no trainable parameters.
    Computes (x-center)/scale
    """

    def __init__(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor):
        return (x - self.center) / self.scale


class UnNormalizeLayer(torch.nn.Module):
    """
    Layer with no trainable parameters.
    Computes x*scale + center
    """

    def __init__(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor):
        return x * self.scale + self.center


class TransformLayer(torch.nn.Module):
    """
    Layer with no trainable parameter
    Computes R*x+t from a given 4x4 transformation matrix R|t
    """

    def __init__(self, T: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("R", T[:3, :3])
        self.register_buffer("t", T[:3, 3])

    @staticmethod
    def from_R_t(R: torch.Tensor, t: torch.Tensor) -> "TransformLayer":
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return TransformLayer(T)

    def forward(self, x: torch.Tensor):
        return x @ self.R.T + self.t


class Mesh:

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_normals: torch.Tensor,
        face_normals: torch.Tensor,
        uvs: torch.Tensor = None,
        vertex_colors: torch.Tensor = None,
        device: torch.device = "cuda",
    ) -> None:

        self.vertices = vertices.to(device)
        self.faces = faces.to(device)

        self.uvs = uvs
        if self.uvs is not None:
            self.uvs.to(device)

        self.vertex_colors = vertex_colors
        if self.vertex_colors is not None:
            self.vertex_colors.to(device)

        self.vertex_normals = vertex_normals.to(device)
        self.face_normals = face_normals.to(device)
        self.device = device

        self.face_areas: torch.Tensor = None
        self.compute_face_areas()

        self._normal_scale = torch.tensor([1.0], device=device)
        self._normal_center = torch.tensor([0.0, 0.0, 0.0], device=device)
        self._is_normal = False

    @staticmethod
    def from_file(path: str, device: torch.device = "cuda"):
        mesh: trimesh.base.Trimesh = trimesh.load(path, force="mesh")
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-10))
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        vertex_normals = torch.nn.functional.normalize(
            torch.tensor(mesh.vertex_normals, dtype=torch.float, device=device), dim=1
        )
        face_normals = torch.nn.functional.normalize(
            torch.tensor(mesh.face_normals, dtype=torch.float, device=device), dim=1
        )

        if hasattr(mesh.visual, "uv"):
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float, device=device)
        else:
            uvs = None

        return Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            uvs=uvs,
            device=device,
        )

    def to_file(self, file_path: str):
        """
        writes mesh to ply file
        for now can only write vertices faces ands uvs.
        """

        if file_path[-3::] == "obj":
            mesh = trimesh.base.Trimesh(
                self.vertices.cpu().numpy(),
                self.faces.cpu().numpy(),
                vertex_normals=self.vertex_normals.cpu().numpy(),
                vertex_colors=(
                    self.vertex_colors.cpu().numpy()
                    if self.vertex_colors is not None
                    else None
                ),
            )

            if self.uvs is not None:
                mesh.visual = trimesh.visual.TextureVisuals(self.uvs.cpu().numpy())
            mesh.export(file_path)

        elif file_path[-3::] == "ply":
            vertices = self.vertices.cpu().numpy()
            vertex_normals = self.vertex_normals.cpu().numpy()
            faces = self.faces.cpu().numpy()
            uvs = self.uvs.cpu().numpy()

            dtype_full = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
                ("u", "f4"),
                ("v", "f4"),
            ]

            v_elements = np.empty(vertices.shape[0], dtype=dtype_full)
            v_attributes = np.concatenate((vertices, vertex_normals, uvs), axis=1)
            v_elements[:] = list(map(tuple, v_attributes))

            faces = np.array(
                [tuple([face]) for face in faces.tolist()],
                dtype=[("vertex_indices", "i4", (3,))],
            )

            v = plyfile.PlyElement.describe(v_elements, "vertex")
            f = plyfile.PlyElement.describe(faces, "face")

            ply_data = plyfile.PlyData([v, f], text=False)
            ply_data.write(file_path)

        else:
            print("Can only export mesh as obj or ply for now")

    def compute_face_areas(self):
        triangles = self.vertices[self.faces]
        edges1 = triangles[:, 1] - triangles[:, 0]
        edges2 = triangles[:, 2] - triangles[:, 0]
        areas = 0.5 * torch.norm(
            torch.cross(edges1, edges2, dim=1), dim=1, keepdim=True
        )

        self.face_areas = areas

    def random_barycentric(self, n_samples: int):
        sqr_r1 = torch.sqrt(torch.rand(n_samples))
        r2 = torch.rand(n_samples)

        coords = torch.zeros((n_samples, 3))
        coords[:, 0] = 1.0 - sqr_r1
        coords[:, 1] = sqr_r1 * (1 - r2)
        coords[:, 2] = sqr_r1 * r2

        return coords.to(self.device)

    def sample_faces(
        self, n_samples: int, weighted: bool = True, replacement: bool = False
    ):
        """
        Samples N_samples face ids uniformly from the mesh (probability weighted by the area of the face if weighted is true)
        """
        if n_samples > self.faces.shape[0]:
            replacement = True
        if weighted:
            probs = (self.face_areas / torch.sum(self.face_areas)).squeeze()
            faces_to_sample = torch.multinomial(
                probs, n_samples, replacement=replacement
            )
            return faces_to_sample.to(self.device)
        else:
            return torch.randint(
                0, int(self.face_areas.shape[0]), (n_samples,), device=self.device
            )

    def sample_mesh(self, n_samples: int) -> torch.Tensor:
        face_idx = self.sample_faces(n_samples)
        triangles = self.vertices[self.faces[face_idx]]
        barycenters = self.random_barycentric(n_samples=n_samples)
        points = barycenters.unsqueeze(1) @ triangles
        return points.squeeze().to(self.device)

    def sample_from_barycenters(
        self, face_idx: torch.LongTensor, barycenters: torch.FloatTensor
    ):
        """
        Samples the mesh from:
        [N,] face ids to sample
        [N, 3] barycentric coordinates
        """

        triangles = self.vertices[self.faces[face_idx]]
        points = barycenters.unsqueeze(1) @ triangles
        return points.squeeze().to(self.device)

    def sample_uvs_from_barycenters(
        self, face_idx: torch.LongTensor, barycenters: torch.FloatTensor
    ):
        """
        Samples the mesh uvs from:
        [N,] face ids to sample
        [N, 3] barycentric coordinates
        """

        vertex_uvs = self.uvs[self.faces[face_idx]]
        uvs = barycenters.unsqueeze(1) @ vertex_uvs
        return uvs.squeeze().to(self.device)

    def sample_uv(self, n_samples: int):
        return torch.rand(n_samples, 2, device=self.device)

    def sample_smooth_normals(
        self, faces: torch.LongTensor, barycenters: torch.FloatTensor
    ) -> torch.Tensor:
        normals = self.vertex_normals[self.faces[faces]]
        interpolated_normals = (barycenters.unsqueeze(1) @ normals).squeeze()
        return interpolated_normals / torch.norm(
            interpolated_normals, dim=-1, keepdim=True
        )

    def sample_face_normals(
        self,
        faces: torch.LongTensor,
    ):
        return self.face_normals[faces]

    def compute_random_t_b(self, normals) -> tuple[torch.Tensor, torch.Tensor]:
        """
        computes random tangeants and bitangents from a set of normals
        normals : [n,3]
        output : tuple(tangeants:[n,3], bitangeants:[n,3])
        """

        normals = torch.nn.functional.normalize(normals, dim=1)

        tangeants = torch.rand(normals.shape, device=self.device)
        tangeants = (
            tangeants - torch.sum(tangeants * normals, dim=1, keepdim=True) * normals
        )
        tangeants = torch.nn.functional.normalize(tangeants, dim=1)

        bitangeants = torch.nn.functional.normalize(
            torch.cross(normals, tangeants, dim=1), dim=1
        )

        return tangeants, bitangeants

    # def compute_random_t_b(self, normals) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     computes random tangents and bitangents from a set of normals
    #     normals : [n,3]
    #     output : tuple(tangents:[n,3], bitangents:[n,3])

    #     FROM Tom Duff et al. "Building an Orthonormal Basis, Revisited"
    #     see http://jcgt.org/published/0006/01/01/paper.pdf

    #     """

    #     normals = torch.nn.functional.normalize(normals, dim=1)
    #     rots = make_rotation_matrix(
    #         normals,
    #         torch.pi * torch.rand((normals.shape[0], 1), device="cuda")
    #         - torch.pi / 2.0,
    #     )

    #     nx, ny, nz = torch.split(normals, 1, dim=-1)
    #     sign = torch.copysign(torch.ones((normals.shape[0], 1), device="cuda"), nz)

    #     a = -1.0 / (sign + nz)
    #     b = nx * ny * a

    #     tangents = torch.cat(
    #         [
    #             1.0 + sign * nx * nx * a,
    #             sign * b,
    #             -sign * nx,
    #         ],
    #         dim=-1,
    #     )
    #     bitangents = torch.cat(
    #         [
    #             b,
    #             sign + ny * ny * a,
    #             -ny,
    #         ],
    #         dim=-1,
    #     )
    #     tangents = torch.bmm(rots, tangents.unsqueeze(-1)).squeeze()
    #     bitangents = torch.bmm(rots, bitangents.unsqueeze(-1)).squeeze()
    #     return tangents, bitangents

    def compute_vertex_normals(self) -> None:
        """
        Recomputes the vertex normals (they should already exist if the mesh was loaded from trimesh)
        """
        vertex_normals = torch.zeros_like(self.vertices)
        triangles = self.vertices[self.faces]
        a = triangles[:, 1, :] - triangles[:, 0, :]
        b = triangles[:, 2, :] - triangles[:, 0, :]
        face_normals = torch.cross(a, b, dim=-1)

        vertex_normals.index_add_(0, self.faces[:, 0], face_normals)
        vertex_normals.index_add_(0, self.faces[:, 1], face_normals)
        vertex_normals.index_add_(0, self.faces[:, 2], face_normals)

        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)

    @staticmethod
    def write_pcd(point_cloud: torch.Tensor, output_path, color: torch.Tensor = None):
        """
        Writes a [N,3] tensor as a .ply pointcloud file
        """
        if color is not None:
            cloud = trimesh.PointCloud(
                point_cloud.detach().cpu().numpy(), color.detach().cpu().numpy()
            )

        else:
            cloud = trimesh.PointCloud(point_cloud.detach().cpu().numpy())

        cloud.export(output_path)

    @staticmethod
    def write_pcd_numpy(point_cloud: np.ndarray, output_path, color: np.ndarray = None):
        """
        Writes a [N,3] ndarray as a .ply pointcloud file
        """
        cloud = trimesh.PointCloud(point_cloud, color)
        cloud.export(output_path)

    def normalize_mesh(self):
        if not self._is_normal:
            self.compute_normalization_params()
            self.vertices = (self.vertices - self._normal_center) / self._normal_scale
            self._is_normal = True

        else:
            print("this mesh was already normalized.")

    def compute_normalization_params(self):
        min_vertex = torch.min(self.vertices, dim=0)[0]
        max_vertex = torch.max(self.vertices, dim=0)[0]
        scale = torch.max(max_vertex - min_vertex)
        center = (max_vertex + min_vertex) / 2
        self._normal_center = center
        self._normal_scale = scale

    def reset_mesh(self):
        if self._is_normal:
            self.vertices = self._normal_scale * self.vertices + self._normal_center
            self._is_normal = False

    def apply_transform(self, transform: torch.FloatTensor | np.ndarray) -> None:
        """
        Applies rigid transform to the mesh,  transform is a 4x4 tensor (or np array)
        [R, t]
        [0, 1]
        R can be a rotation or a scaling or a rotation multiplied with a scaling
        """

        # np arrays are double by default
        if isinstance(transform, np.ndarray):
            transform = transform.astype(np.float32)

        R = transform[:3, :3]
        T = transform[:3, 3]
        self.vertices = self.vertices @ R.T + T

        # fix vertex normals
        transpose_inverse_transform = torch.inverse(transform).transpose(0, 1)
        R = transpose_inverse_transform[:3, :3]
        T = transpose_inverse_transform[:3, 3]
        self.vertex_normals = self.vertex_normals @ R.T + T
        self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals, dim=1)

    @staticmethod
    def read_transform_from_txt(txt_path: str, device: str = "cuda") -> torch.Tensor:
        """
        Reads a transform matrix from a txt file from a coupole scan.
        returns 4x4 tensor
        """
        xml = ET.parse(txt_path).getroot()
        data = [float(data.text) for data in xml.findall("_")]

        transform = torch.tensor(
            np.array(data).astype(np.float32).reshape(4, 4),
            device=device,
        )

        return transform

    def to(self, device: torch.device):
        self.device = device
        self.face_areas = self.face_areas.to(device)
        self.faces = self.faces.to(device)
        if self.uvs is not None:
            self.uvs = self.uvs.to(device)
        self.vertices = self.vertices.to(device)
        self.vertex_normals = self.vertex_normals.to(device)
        self._normal_center = self._normal_center.to(device)
        self._normal_scale = self._normal_scale.to(device)

    def compute_naive_uv(self):
        """
        generates uv coordinates for each vertex by flattening the dimension of lowest variance and renormalizing to [0,1]
        """
        variance = torch.var(self.vertices, dim=0)
        flatten_dim = torch.argmin(variance)
        uvs = self.vertices.clone().detach()
        uvs = uvs[
            :, [i for i in range(uvs.shape[1]) if i != flatten_dim]
        ]  # drop low variance dim

        uvs -= uvs.min(dim=0, keepdim=True)[0]
        uvs /= uvs.max(dim=0, keepdim=True)[0]

        self.uvs = uvs

    def set_uvs(self, uvs: torch.Tensor) -> None:

        if uvs.shape[0] != self.vertices.shape[0]:
            raise ValueError(
                f"got {uvs.shape[0]} uv coords but the mesh has {self.vertices.shape[0]} vertices"
            )

        self.uvs = uvs.to(self.device)

    def generate_normal_map(
        self, size: int, output_path: str = None, k: int = 3
    ) -> None:
        """
        Bakes the normals of the mesh into a texture by sampling each face randomly k times
        and interpolating the normals.

        """
        n = self.faces.shape[0]
        barycenters = self.random_barycentric(k * n)
        faces = torch.arange(0, k * n, device=self.device) % n
        normals = self.sample_smooth_normals(faces, barycenters).cpu().numpy()
        uvs = self.sample_uvs_from_barycenters(faces, barycenters).cpu().numpy()

        texture = np.zeros((size, size, 3), dtype=np.float32)
        weights_map = np.zeros((size, size), dtype=np.float32)

        for uv, normal in zip(uvs, normals):
            u, v = uv

            x = u * (size - 1)
            y = (1 - v) * (size - 1)  # Flip V for image coordinates

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, size - 1)
            y1 = min(y0 + 1, size - 1)

            dx = x - x0
            dy = y - y0

            weights = {
                (x0, y0): (1 - dx) * (1 - dy),
                (x1, y0): dx * (1 - dy),
                (x0, y1): (1 - dx) * dy,
                (x1, y1): dx * dy,
            }

            for (x, y), weight in weights.items():
                texture[y, x] += weight * normal
                weights_map[y, x] += weight

        weights_map[weights_map == 0.0] = 1.0
        weights_map = weights_map[..., np.newaxis]
        texture = texture / weights_map

        texture = (texture + 1.0) / 2.0

        if output_path is not None:
            texture_uint = (texture * 255.0).astype(np.uint8)
            Image.fromarray(texture_uint, mode="RGB").save(output_path)

        return texture

    def get_AABB(self) -> torch.Tensor:
        min_, _ = torch.min(self.vertices, dim=0)
        max_, _ = torch.max(self.vertices, dim=0)

        return torch.stack([min_, max_], dim=0)

    def tessellate(
        self,
    ):
        """
        Tessellate a mesh by splitting each triangular face into four smaller triangles.

        Args:
            vertices (torch.Tensor): Tensor of shape [N, 3], vertex coordinates.
            faces (torch.Tensor): Tensor of shape [M, 3], face indices.
            attributes (torch.Tensor): Tensor of shape [N, D], optional vertex attributes.

        Returns:
            torch.Tensor: New vertices, shape [N', 3].
            torch.Tensor: New faces, shape [M', 3].
            torch.Tensor: New attributes, shape [N', D] (if attributes are provided).
        """
        # Extract vertex indices of each face
        v0, v1, v2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]

        # Compute midpoints for each edge
        edge01 = (self.vertices[v0] + self.vertices[v1]) / 2  # Midpoint of edge v0-v1
        edge12 = (self.vertices[v1] + self.vertices[v2]) / 2  # Midpoint of edge v1-v2
        edge20 = (self.vertices[v2] + self.vertices[v0]) / 2  # Midpoint of edge v2-v0

        # Combine vertices and edge midpoints
        new_vertices = torch.cat([self.vertices, edge01, edge12, edge20], dim=0)

        # Deduplicate vertices and find new indices
        new_vertices, inverse_indices = torch.unique(
            new_vertices, dim=0, return_inverse=True
        )

        # Map old vertices and midpoints to their new indices
        num_original = self.vertices.size(0)
        edge01_idx = inverse_indices[num_original : num_original + len(edge01)]
        edge12_idx = inverse_indices[
            num_original + len(edge01) : num_original + len(edge01) + len(edge12)
        ]
        edge20_idx = inverse_indices[num_original + len(edge01) + len(edge12) :]

        v0_new = inverse_indices[v0]
        v1_new = inverse_indices[v1]
        v2_new = inverse_indices[v2]

        # Create new faces
        new_faces = torch.cat(
            [
                torch.stack([v0_new, edge01_idx, edge20_idx], dim=1),
                torch.stack([v1_new, edge12_idx, edge01_idx], dim=1),
                torch.stack([v2_new, edge20_idx, edge12_idx], dim=1),
                torch.stack([edge01_idx, edge12_idx, edge20_idx], dim=1),
            ],
            dim=0,
        )

        # TODO: manage attributes

        self.vertices = new_vertices
        self.faces = new_faces


if __name__ == "__main__":
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    trans = TransformLayer(T)
    x = torch.ones(1000, 3)
    y = trans(x)
    print(y.shape)
