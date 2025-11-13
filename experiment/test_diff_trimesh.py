import unittest
import torch
import trimesh
import numpy as np

def point_to_triangle_distance_and_closest(points, triangles, normals=None, eps=1e-20):
    """
    Batched shortest distance from points to triangles in 3D.
    Also returns the closest points on the triangles.
    Handles degenerate triangles robustly.

    Args:
        points: (N,3) tensor of points
        tris:   (N,3,3) tensor of triangles
        eps:    small value to avoid division by zero

    Returns:
        distances: (N,) tensor of distances
        closest_points: (N,3) tensor of closest points on triangles
    """
    # Extract vertices
    A, B, C = triangles.unbind(dim=1)  # each (N, 3)
    NA, NB, NC = None, None, None
    if normals is not None:
        NA, NB, NC = normals.unbind(dim=1)  # each (N, 3)

    # Compute edges
    AB = B - A
    AC = C - A
    BC = C - B

    # Vector from A to point
    AP = points - A

    # Compute barycentric coordinates
    dot00 = torch.sum(AB * AB, dim=1)
    dot01 = torch.sum(AB * AC, dim=1)
    dot02 = torch.sum(AB * AP, dim=1)
    dot11 = torch.sum(AC * AC, dim=1)
    dot12 = torch.sum(AC * AP, dim=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if inside triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Projection onto plane
    proj = A + u.unsqueeze(1) * AB + v.unsqueeze(1) * AC

    # Closest points on edges
    # Edge AB
    t_ab = torch.clamp(torch.sum(AP * AB, dim=1) / (dot00 + eps), 0, 1)
    p_ab = A + t_ab.unsqueeze(1) * AB

    # Edge AC
    t_ac = torch.clamp(torch.sum(AP * AC, dim=1) / (dot11 + eps), 0, 1)
    p_ac = A + t_ac.unsqueeze(1) * AC

    # Edge BC
    BP = points - B
    dot_bc = torch.sum(BC * BC, dim=1)
    t_bc = torch.clamp(torch.sum(BP * BC, dim=1) / (dot_bc + eps), 0, 1)
    p_bc = B + t_bc.unsqueeze(1) * BC

    # Compute distances
    dist_proj = torch.norm(points - proj, dim=1)
    dist_ab = torch.norm(points - p_ab, dim=1)
    dist_ac = torch.norm(points - p_ac, dim=1)
    dist_bc = torch.norm(points - p_bc, dim=1)

    # Find minimum distance for outside points
    dist_out, idx_out = torch.min(
        torch.stack([dist_ab, dist_ac, dist_bc], dim=1), dim=1
    )
    closest_out = torch.stack([p_ab, p_ac, p_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    
    # compute u, v, w
    # u * AB + v * AC = w * A + u * B + v * C
    
    # Edge AB: (u, v) = (t_ab, 0)
    u_ab = t_ab
    v_ab = torch.zeros_like(t_ab)
    
    # Edge AC: (u, v) = (0, t_ac)
    u_ac = torch.zeros_like(t_ac)
    v_ac = t_ac
    
    # Edge BC: (u, v) = (1 - t_bc, t_bc)
    u_bc = 1 - t_bc
    v_bc = t_bc
    
    u_out = torch.stack([u_ab, u_ac, u_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    v_out = torch.stack([v_ab, v_ac, v_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    

    # Final results
    distances = torch.where(inside, dist_proj, dist_out)
    closest_points = torch.where(inside.unsqueeze(1), proj, closest_out)
    
    uv_coords = torch.stack([
        torch.where(inside, u, u_out),
        torch.where(inside, v, v_out)
    ], dim=1)
    
    if normals is not None:
        normal_intp = (1 - uv_coords[:, 0] - uv_coords[:, 1]).unsqueeze(1) * NA \
          + uv_coords[:, 0].unsqueeze(1) * NB + uv_coords[:, 1].unsqueeze(1) * NC
        normal_intp = normal_intp / (torch.norm(normal_intp, dim=1, keepdim=True) + 1e-12)
        return distances, closest_points, normal_intp
    else:
        return distances, closest_points

# 导入你的函数
# from your_module import point_to_triangle_distance_and_closest

class TestPointToTriangleDistance(unittest.TestCase):
    def setUp(self):
        """创建测试用的球体mesh"""
        # 创建一个半径为1.0，中心在原点的球体
        self.radius = 1.0
        self.mesh = trimesh.creation.icosphere(subdivisions=4, radius=self.radius)
        
        # 将mesh数据转换为torch tensor
        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32)
        self.faces = torch.tensor(self.mesh.faces, dtype=torch.long)
        self.normals = torch.tensor(self.mesh.vertex_normals, dtype=torch.float32)
        
    def _find_nearest_triangles(self, points):
        """辅助函数：为每个点找到最近的三角形面片"""
        points_np = points.detach().numpy() if isinstance(points, torch.Tensor) else points
        nearest_indices = []
        
        for point in points_np:
            face_index = self.mesh.nearest.on_surface([point])[2].item()
            nearest_indices.append(face_index)
        
        return torch.tensor(nearest_indices, dtype=torch.long)
    
    def test_closest_point_accuracy_on_surface(self):
        """测试1：球面上的点，验证最近点计算准确性"""
        # 在球面上采样测试点
        n_test = 100
        phi = torch.rand(n_test) * 2 * np.pi
        theta = torch.acos(1 - 2 * torch.rand(n_test))
        
        test_points = torch.stack([
            self.radius * torch.sin(theta) * torch.cos(phi),
            self.radius * torch.sin(theta) * torch.sin(phi),
            self.radius * torch.cos(theta)
        ], dim=1)
        
        # 找到最近的三角形
        face_indices = self._find_nearest_triangles(test_points)
        
        # 获取对应的三角形顶点
        triangles = self.vertices[self.faces[face_indices]]  # (N, 3, 3)
        
        # 调用函数
        distances, closest_points = point_to_triangle_distance_and_closest(
            test_points, triangles
        )
        
        # 验证距离接近0
        max_dist = distances.max().item()
        self.assertLess(max_dist, 1e-2, f"球面上的点最大距离过大: {max_dist}")
        
        # 验证最近点接近原始点
        point_diff = torch.norm(test_points - closest_points, dim=1)
        max_diff = point_diff.max().item()
        self.assertLess(max_diff, 1e-2, f"最近点偏差过大: {max_diff}")
        
        print(f"✓ 球面上点测试通过: 最大距离={max_dist:.6f}, 最大偏差={max_diff:.6f}")
    
    def test_closest_point_accuracy_off_surface(self):
        """测试2：球面外的点，验证距离正确性"""
        # 在球面外创建测试点
        n_test = 100
        phi = torch.rand(n_test) * 2 * np.pi
        theta = torch.acos(1 - 2 * torch.rand(n_test))
        
        # 距离球面0.5单位的点
        offset = 0.5
        test_points = torch.stack([
            (self.radius + offset) * torch.sin(theta) * torch.cos(phi),
            (self.radius + offset) * torch.sin(theta) * torch.sin(phi),
            (self.radius + offset) * torch.cos(theta)
        ], dim=1)
        
        # 找到最近的三角形
        face_indices = self._find_nearest_triangles(test_points)
        triangles = self.vertices[self.faces[face_indices]]
        
        # 调用函数
        distances, closest_points = point_to_triangle_distance_and_closest(
            test_points, triangles
        )
        
        # 验证距离接近offset
        dist_error = torch.abs(distances - offset).max().item()
        self.assertLess(dist_error, 1e-2, f"距离计算误差过大: {dist_error}")
        
        # 验证最近点在球面上（距离原点约等于半径）
        closest_norms = torch.norm(closest_points, dim=1)
        norm_error = torch.abs(closest_norms - self.radius).max().item()
        self.assertLess(norm_error, 1e-2, f"最近点不在球面上: {norm_error}")
        
        print(f"✓ 球面外点测试通过: 距离误差={dist_error:.6f}, 球面误差={norm_error:.6f}")
    
    def test_gradient_normal_computation(self):
        """测试3：通过自动微分计算法向"""
        n_test = 50
        # 创建需要梯度的测试点
        test_points = torch.randn(n_test, 3, requires_grad=True)
        test_points.data = test_points.data / torch.norm(test_points.data, dim=1, keepdim=True) * 1.5  # 半径1.5
        
        # 找到最近的三角形
        face_indices = self._find_nearest_triangles(test_points.detach())
        triangles = self.vertices[self.faces[face_indices]]
        normals = self.normals[self.faces[face_indices]]
        
        # 调用函数计算距离
        distances, closest_points, normals_intp = point_to_triangle_distance_and_closest(
            test_points, triangles, normals
        )
        
        # 对距离求和并反向传播
        total_distance = distances.sum()
        total_distance.backward()
        
        # 梯度应该等于法向方向
        gradients = test_points.grad
        
        # 计算理论法向 (point - closest_point) / distance
        diff = test_points.detach() - closest_points.detach()
        expected_normals = diff / torch.norm(diff, dim=1, keepdim=True)
        
        expected_normals = normals_intp
        
        # 验证梯度方向与理论法向一致
        cos_sim = torch.sum(gradients * expected_normals, dim=1) / \
                  (torch.norm(gradients, dim=1) * torch.norm(expected_normals, dim=1))
        
        # 由于梯度是距离的梯度，方向应该一致（可能有符号差异）
        cos_sim_abs = torch.abs(cos_sim)
        min_cos_sim = cos_sim_abs.min().item()
        
        self.assertGreater(min_cos_sim, 0.95, 
                          f"梯度方向与法向不一致, 最小余弦相似度: {min_cos_sim}")
        
        print(f"✓ 法向计算测试通过: 最小余弦相似度={min_cos_sim:.6f}")
    
    def test_degenerate_triangle_robustness(self):
        """测试4：退化三角形鲁棒性测试"""
        # 创建包含退化三角形的测试数据
        normal_triangles = torch.tensor([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # 正常三角形
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],  # 退化三角形（两点重合）
        ], dtype=torch.float32)
        
        test_points = torch.tensor([[0.2, 0.2, 0.1], [0.5, 0.0, 0.1]], dtype=torch.float32)
        
        # 应能正常计算不报错
        distances, closest_points = point_to_triangle_distance_and_closest(
            test_points, normal_triangles
        )
        
        # 验证输出形状正确
        self.assertEqual(distances.shape, (2,))
        self.assertEqual(closest_points.shape, (2, 3))
        
        # 验证距离非负
        self.assertTrue((distances >= 0).all())
        
        print("✓ 退化三角形鲁棒性测试通过")

if __name__ == '__main__':
    unittest.main()