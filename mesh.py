import xml.etree.ElementTree as ET

import numpy as np
import plyfile
import torch
import trimesh


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
                self.vertices.numpy(),
                self.faces.numpy(),
                vertex_normals=self.vertex_normals.numpy(),
                vertex_colors=(
                    self.vertex_colors.numpy()
                    if self.vertex_colors is not None
                    else None
                ),
            )
            mesh.visual = trimesh.visual.TextureVisuals(self.uvs)
            mesh.export(file_path)

        elif file_path[-3::] == "ply":
            vertices = self.vertices.numpy()
            vertex_normals = self.vertex_normals.numpy()
            faces = self.faces.numpy()
            uvs = self.uvs.numpy()

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

        # TODO : generate better tangeants

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

    #     # TODO : generate better tangents
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
        cloud = trimesh.PointCloud(
            point_cloud.detach().cpu().numpy(), color.detach().cpu().numpy()
        )
        cloud.export(output_path)

    @staticmethod
    def write_pcd_numpy(point_cloud: np.ndarray, output_path, color: np.ndarray = None):
        """
        Writes a [N,3] ndarray as a .ply pointcloud file
        """
        cloud = trimesh.PointCloud(point_cloud, color)
        cloud.export(output_path)

    def normalize_mesh(self):
        min_vertex = torch.min(self.vertices, dim=0)[0]
        max_vertex = torch.max(self.vertices, dim=0)[0]
        scale = torch.max(max_vertex - min_vertex)
        center = (max_vertex + min_vertex) / 2
        self.vertices = (self.vertices - center) / scale
        self.compute_vertex_normals()

        self._normal_center = center
        self._normal_scale = scale
        self._is_normal = True

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

    def read_transform_from_txt(self, txt_path: str) -> torch.Tensor:
        """
        Reads a transform matrix from a txt file from a coupole scan.
        returns 4x4 tensor
        """
        xml = ET.parse(txt_path).getroot()
        data = [float(data.text) for data in xml.findall("_")]

        transform = torch.tensor(
            np.array(data).astype(np.float32).reshape(4, 4),
            device=self.device,
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


if __name__ == "__main__":
    # bunny_path = "assets/meshes/stanford-bunny.obj"
    # mesh = Mesh.from_file(bunny_path, "cpu")
    # points = mesh.sample_mesh(1000)
    # mesh.normalize_mesh()
    # Mesh.write_pcd(mesh.vertices, "./test.ply")
    axis = torch.rand(100, 3)
    angle = torch.rand(100, 1)

    print(make_rotation_matrix(axis, angle).shape)
