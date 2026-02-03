import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def torch_normal_to_rot(normals: torch.Tensor):
    """
    Input: (N, 3) normals
    Output: (N, 3, 3) rotation matrices where col 0 is normal
    """
    # 保持原有逻辑，确保数值稳定性
    z_axis = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    tangent_guess = torch.zeros_like(z_axis)
    tangent_guess[..., 0] = 1.0
    mask = torch.abs(z_axis[..., 0]) > 0.9
    
    # 保持 dtype 和 device 一致
    tangent_guess[mask] = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    
    y_axis = torch.linalg.cross(z_axis, tangent_guess)
    y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-8)
    
    x_axis = torch.linalg.cross(y_axis, z_axis)
    
    rot = torch.stack([z_axis, x_axis, y_axis], dim=-1) 
    return rot

class ContactQPCvxpy:
    def __init__(self, miu_coef, solver_type="CLARABEL"):
        """
        Args:
            miu_coef: friction coefficients [miu_1, miu_2]
            solver_type: cvxpy solver (e.g., 'CLARABEL', 'SCS', 'ECOS')
        """
        self.miu_coef = miu_coef
        self.solver_type = solver_type
        
        # 缓存不同 N (接触点数量) 对应的 CvxpyLayer
        # Key: num_contact, Value: (layer, E_matrix_np)
        self.layer_cache = {}

    def _build_constraint_arrays(self, num_contact):
        """
        构建基于 NumPy 的约束矩阵 G, h 和变换矩阵 E。
        这些矩阵定义了 QP 的结构，对于固定的 N 是常数。
        """
        num_f_strength = num_contact * 6
        
        # 定义约束矩阵维度
        G_matrix = np.zeros((num_f_strength + num_contact + 1, num_f_strength))
        h_matrix = np.zeros((num_f_strength + num_contact + 1))

        # 1. Force constraint: -force <= -eps (force >= eps)
        G_matrix[range(0, num_f_strength), range(0, num_f_strength)] = -1.0
        h_matrix[range(0, num_f_strength)] = -1e-2

        # 2. Pressure constraint: pressure <= 1
        for i in range(num_contact):
            # 假设第 i 个接触点的参数在 6*i : 6*i+6
            # 这里需要根据原始逻辑确认哪些列对应 pressure。
            # 原代码逻辑: E_matrix 的构造方式决定了 parameters 的含义。
            # 此处保持与原始 _build_constraint 一致的索引逻辑
            G_matrix[num_f_strength + i, 6 * i : 6 * i + 6] = - 1.0
        h_matrix[-num_contact - 1 : -1] = 0.

        # 3. Sum pressure constraint: -sum(pressure) <= -0.1 (sum >= 0.1)
        G_matrix[-1, :] = -1.0
        h_matrix[-1] = -0.1

        # 构建 E Matrix (Contact Parameters -> Local Force)
        E_matrix = np.zeros((num_contact, 6, 6))
        E_matrix[:, 0, :] = 1
        E_matrix[:, 1, 0] = E_matrix[:, 2, 2] = self.miu_coef[0]
        E_matrix[:, 1, 1] = E_matrix[:, 2, 3] = -self.miu_coef[0]
        E_matrix[:, 3, 4] = self.miu_coef[1]
        E_matrix[:, 3, 5] = -self.miu_coef[1]

        return G_matrix, h_matrix, E_matrix

    def _get_or_create_layer(self, num_contact):
        """
        获取或创建对应接触点数量的 CvxpyLayer。
        """
        if num_contact in self.layer_cache:
            return self.layer_cache[num_contact]

        # 1. 获取常数矩阵
        G_np, h_np, E_np = self._build_constraint_arrays(num_contact)

        # 2. 定义 CVXPY 问题
        # 变量 x: 所有接触点的参数拼接 (Size: 6 * N)
        x = cp.Variable(num_contact * 6)
        
        # 参数 A: 映射矩阵 (Size: 6, 6*N)
        # 参数 g: 重力 Wrench (Size: 6)
        A_param = cp.Parameter((6, num_contact * 6))
        g_param = cp.Parameter(6)
        
        # 常数封装
        G_const = cp.Constant(G_np)
        h_const = cp.Constant(h_np)
        
        # 权重矩阵 W (对角阵)
        r = 1e1
        W_diag = np.array([1., 1., 1., r**2, r**2, r**2])
        W_const = cp.Constant(np.diag(W_diag))

        # 目标函数: Minimize 0.5 * || A*x + g ||_W^2
        # 即: 0.5 * (Ax + g)^T W (Ax + g)
        # 这等价于原始代码中的 solve_qp 形式
        error_term = A_param @ x + g_param
        objective = cp.Minimize(0.5 * cp.quad_form(error_term, W_const))

        # 约束条件
        constraints = [G_const @ x <= h_const]

        # 创建问题并检查 DPP (Differentiable Convex Programming) 兼容性
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        # 3. 创建 Layer
        layer = CvxpyLayer(problem, parameters=[A_param, g_param], variables=[x])
        
        # 存入缓存
        self.layer_cache[num_contact] = (layer, E_np)
        return layer, E_np

    def solve(
        self,
        pos: torch.Tensor,       # (N, 3)
        normal: torch.Tensor,    # (N, 3)
        gravity,                 # (6,) np.ndarray or torch.Tensor
        gravity_center,          # (3,) np.ndarray or torch.Tensor
    ):
        num_contact = pos.shape[0]
        device = pos.device

        # 1. 数据类型统一转换
        if not isinstance(gravity, torch.Tensor):
            gravity_torch = torch.from_numpy(gravity).to(dtype=torch.float32, device=device)
        else:
            gravity_torch = gravity.to(dtype=torch.float32, device=device)

        if not isinstance(gravity_center, torch.Tensor):
            g_center_torch = torch.from_numpy(gravity_center).to(dtype=torch.float32, device=device)
        else:
            g_center_torch = gravity_center.to(dtype=torch.float32, device=device)

        # 2. 计算旋转矩阵和相对位置 (全部使用 Torch 操作)
        rot = torch_normal_to_rot(normal) # (N, 3, 3)
        axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]
        
        relative_pos = pos - g_center_torch[None, :3]

        # 3. 构建 Grasp Matrix (N, 6, 6)
        grasp_matrix = torch.zeros((num_contact, 6, 6), dtype=torch.float32, device=device)
        
        grasp_matrix[:, :3, 0] = axis_0
        grasp_matrix[:, 3:, 3] = axis_0
        
        grasp_matrix[:, :3, 1] = axis_1
        grasp_matrix[:, 3:, 4] = axis_1
        
        grasp_matrix[:, :3, 2] = axis_2
        grasp_matrix[:, 3:, 5] = axis_2
        
        grasp_matrix[:, 3:, 0] = torch.linalg.cross(relative_pos, axis_0)
        grasp_matrix[:, 3:, 1] = torch.linalg.cross(relative_pos, axis_1)
        grasp_matrix[:, 3:, 2] = torch.linalg.cross(relative_pos, axis_2)

        # 4. 获取 CvxpyLayer 和 E Matrix
        layer, E_matrix_np = self._get_or_create_layer(num_contact)
        E_matrix_torch = torch.from_numpy(E_matrix_np).to(dtype=torch.float32, device=device)

        # 5. 计算 Param2Force (A Matrix)
        # param2force shape: (N, 6, 6)
        param2force = torch.matmul(grasp_matrix, E_matrix_torch) 
        
        # 变换维度并展平以匹配 cvxpylayers 参数要求
        # 对应原始代码: np.transpose(param2force, (1, 0, 2)).reshape(6, -1)
        # 这里的 A_torch 就是优化问题中的 parameter A
        A_torch = param2force.permute(1, 0, 2).reshape(6, -1) # Shape: (6, 6*N)

        # 6. 调用 CvxpyLayer 求解
        # layer 返回的是 tuple (solution,), solution shape: (6*N)
        # 传递 solver_args 以指定求解器
        try:
            solution, = layer(
                A_torch, 
                gravity_torch, 
                # solver_args={"solve_method": self.solver_type}
            )
        except Exception:
            # 如果 solver_args 不受支持（取决于 cvxpylayers 版本），尝试默认调用
            solution, = layer(A_torch, gravity_torch)

        # 7. 结果后处理 (计算 Error)
        # 将解 reshape 回 (N, 6)
        solution_reshaped = solution.reshape(-1, 6)
        
        # 计算 contact wrenches: (N, 6, 6) @ (N, 6, 1) -> (N, 6, 1) -> (N, 6)
        contact_wrenches = torch.matmul(
            param2force, 
            solution_reshaped.unsqueeze(-1)
        ).squeeze(-1)

        # 计算总 Wrench 和 误差
        total_wrench = torch.sum(contact_wrenches, dim=0) + gravity_torch
        
        # 加权 Error (Torque 部分权重更高)
        # 注意：为了保持梯度流，使用 clone 或非原地操作
        weighted_wrench = total_wrench.clone()
        weighted_wrench[3:] = weighted_wrench[3:] * 1e1
        
        wrench_error = torch.norm(weighted_wrench)
        
        # print(wrench_error, torch.norm(solution_reshaped, dim=1).sum())
        # wrench_error  = wrench_error + 1e-1 * torch.norm(solution_reshaped, dim=1).sum()
        
        return wrench_error

class ContactSOCPCvxpy:
    def __init__(self, miu_coef, solver_type="CLARABEL"):
        """
        Args:
            miu_coef: friction coefficient. 
                      Input can be list/array, but SOCP typically uses isotropic friction.
                      We will use miu_coef[0] as mu.
            solver_type: E.g., 'CLARABEL', 'SCS', 'ECOS'.
        """
        # 兼容列表输入，取第一个作为摩擦系数 mu
        if isinstance(miu_coef, (list, tuple, np.ndarray)):
            self.mu = float(miu_coef[0])
        else:
            self.mu = float(miu_coef)
            
        self.solver_type = solver_type
        
        # 缓存不同 N (接触点数量) 对应的 CvxpyLayer
        # Key: num_contact, Value: layer
        self.layer_cache = {}

    def _get_or_create_layer(self, num_contact):
        """
        构建 SOCP 问题的 CvxpyLayer。
        变量 f: (N, 3)，每行对应 [f_normal, f_tangent_1, f_tangent_2]
        """
        if num_contact in self.layer_cache:
            return self.layer_cache[num_contact]

        # 1. 定义变量
        # f[i] 是第 i 个接触点的局部力：[n, t1, t2]
        f = cp.Variable((num_contact, 3))
        
        # 2. 定义参数
        # A: (6, 3N) 巨大的 Grasp Matrix，将所有局部力映射到全局 Wrench
        # g: (6,) 重力/外部 Wrench
        A_param = cp.Parameter((6, num_contact * 3))
        g_param = cp.Parameter(6)
        
        # 3. 权重矩阵 (Diagonal)
        # 保持与原版一致的权重逻辑
        r = 1e1
        W_diag = np.array([1., 1., 1., r**2, r**2, r**2])
        W_const = cp.Constant(np.diag(W_diag))

        # 4. 目标函数
        # 将 f 展平为 (3N,) 向量以进行矩阵乘法
        # cp.reshape(f, (-1,), order='C') 意味着顺序是: f0_n, f0_t1, f0_t2, f1_n...
        f_flat = cp.reshape(f, (num_contact * 3,), order='C')
        
        error_term = A_param @ f_flat + g_param
        objective = cp.Minimize(0.5 * cp.quad_form(error_term, W_const))

        # 5. 约束条件 (SOCP 核心)
        constraints = []
        for i in range(num_contact):
            # a) 法向力 >= 0 (单边约束)
            constraints.append(f[i, 0] >= 1e-2)
            
            # b) 摩擦锥约束 (二阶锥): || f_tangent ||_2 <= mu * f_normal
            # f[i, 1:] 取的是 (t1, t2)
            constraints.append(cp.norm(f[i, 1:], 2) <= self.mu * f[i, 0])

        # 创建问题
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        # 创建 Layer
        layer = CvxpyLayer(problem, parameters=[A_param, g_param], variables=[f])
        
        self.layer_cache[num_contact] = layer
        return layer

    def solve(
        self,
        pos: torch.Tensor,       # (N, 3)
        normal: torch.Tensor,    # (N, 3)
        gravity,                 # (6,) np.ndarray or torch.Tensor
        gravity_center,          # (3,) np.ndarray or torch.Tensor
    ):
        """
        Solve the contact force optimization using SOCP.
        """
        num_contact = pos.shape[0]
        device = pos.device

        # 1. 类型转换
        if not isinstance(gravity, torch.Tensor):
            gravity_torch = torch.from_numpy(gravity).to(dtype=torch.float32, device=device)
        else:
            gravity_torch = gravity.to(dtype=torch.float32, device=device)

        if not isinstance(gravity_center, torch.Tensor):
            g_center_torch = torch.from_numpy(gravity_center).to(dtype=torch.float32, device=device)
        else:
            g_center_torch = gravity_center.to(dtype=torch.float32, device=device)

        # 2. 计算几何基 (Local Frame)
        # rot shape: (N, 3, 3). Columns are [Normal, Tangent1, Tangent2]
        # 注意: torch_normal_to_rot 返回的 col 0 是 Normal
        rot = torch_normal_to_rot(normal)
        
        # 3. 构建 Grasp Matrix A (Mapping Local Forces -> Global Wrench)
        # 我们需要构建一个 (6, 3N) 的矩阵。
        # 对于每个接触点 i，其对应的子矩阵 J_i (6, 3) 为：
        # [   R_i    ]  <- Force part (Rotation matrix columns)
        # [ S(r_i)R_i]  <- Torque part (Cross product of relative pos and axes)
        
        relative_pos = pos - g_center_torch[None, :3] # (N, 3)
        
        # Force Part: 直接就是旋转矩阵的列向量
        # axes (N, 3, 3) -> axis_0 (N, 3), axis_1 (N, 3), axis_2 (N, 3)
        axis_n = rot[..., 0] # Normal
        axis_t1 = rot[..., 1]
        axis_t2 = rot[..., 2]
        
        # Torque Part: r x axis
        torque_n = torch.linalg.cross(relative_pos, axis_n)
        torque_t1 = torch.linalg.cross(relative_pos, axis_t1)
        torque_t2 = torch.linalg.cross(relative_pos, axis_t2)
        
        # 组装 Jacobian (N, 6, 3)
        # Dim 1 (Row): 0-2 for Force, 3-5 for Torque
        # Dim 2 (Col): 0 for Normal, 1 for T1, 2 for T2
        J = torch.zeros((num_contact, 6, 3), dtype=torch.float32, device=device)
        
        # 填入 Force
        J[:, :3, 0] = axis_n
        J[:, :3, 1] = axis_t1
        J[:, :3, 2] = axis_t2
        
        # 填入 Torque
        J[:, 3:, 0] = torque_n
        J[:, 3:, 1] = torque_t1
        J[:, 3:, 2] = torque_t2
        
        # 变换为 (6, 3N) 以匹配 CVXPY 参数 A_param
        # 我们希望的顺序是: [J_0, J_1, ..., J_N] 横向拼接
        # permute(1, 0, 2) -> (6, N, 3)
        # reshape(6, -1) -> (6, 3*N)
        A_torch = J.permute(1, 0, 2).reshape(6, -1)

        # 4. 调用 Layer
        layer = self._get_or_create_layer(num_contact)
        
        try:
            # f_solution shape: (N, 3)
            f_solution, = layer(
                A_torch, 
                gravity_torch, 
                # solver_args={"solve_method": self.solver_type}
            )
        except Exception:
            f_solution, = layer(A_torch, gravity_torch)

        # 5. 后处理与误差计算
        # f_solution: (N, 3). Columns: [fn, ft1, ft2]
        
        # 计算产生的 Contact Wrenches
        # 此时 f_solution 需要展平来乘 A_torch，或者用 J 逐个乘
        # 这里用 J 计算更直观: (N, 6, 3) @ (N, 3, 1) -> (N, 6, 1) -> (N, 6)
        contact_wrenches = torch.matmul(J, f_solution.unsqueeze(-1)).squeeze(-1)
        
        # 计算总 Wrench 和 Error
        total_wrench = torch.sum(contact_wrenches, dim=0) + gravity_torch
        
        # 加权 (Torque 权重 * 10)
        weighted_wrench = total_wrench.clone()
        weighted_wrench[3:] = weighted_wrench[3:] * 1e1
        
        wrench_error = torch.norm(weighted_wrench)
        
        return wrench_error