from abc import abstractmethod
import torch
import numpy as np
import timeit

class QPSolver:
    
    solver: None
    
    g_matrix: None 
    
    G_matrix: None 
    
    l_matrix: None
    
    h_matrix: None 
    
    def init_problem(self, G_matrix, l_matrix, h_matrix):
        self.G_matrix = G_matrix
        self.l_matrix = l_matrix
        self.h_matrix = h_matrix

    @abstractmethod
    def solve(self, Q_matrix, semi_Q_matrix, solution=None):
        return solution

class BATCHED_RELUQP(QPSolver):

    @property
    def glh_type(self):
        return 'glh'
    
    def init_problem(self, G_matrix, l_matrix, h_matrix):
        super().init_problem(G_matrix, l_matrix, h_matrix)
        self.solver = _RELUQP(self.G_matrix.shape[0], self.G_matrix.shape[2], self.G_matrix.shape[1])
            
    def solve(self, Q_matrix, semi_Q_matrix, solution=None):
        return self.solver.solve(Q_matrix, self.G_matrix, self.l_matrix, self.h_matrix)  


@torch.jit.script
def _RELU(input, W, l, u, rho_ind, help_ind, nx: int, nc: int):
    input = torch.bmm(W[rho_ind, help_ind], input.unsqueeze(-1)).squeeze(-1)
    input[:, nx:nx+nc].clamp_(l, u)
    return input

@torch.jit.script
def compute_residuals(H, A, x, z, lam, rho, rho_min: float, rho_max: float):
    t1 = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
    t2 = torch.matmul(H, x.unsqueeze(-1)).squeeze(-1)
    t3 = torch.matmul(A.transpose(-1, -2), lam.unsqueeze(-1)).squeeze(-1)
    primal_res = torch.linalg.vector_norm(t1 - z, dim=-1, ord=torch.inf)
    dual_res = torch.linalg.vector_norm(t2 + t3, dim=-1, ord=torch.inf)
    numerator = torch.div(primal_res, torch.max(torch.linalg.vector_norm(t1, dim=-1, ord=torch.inf), torch.linalg.vector_norm(z, dim=-1, ord=torch.inf)))
    denom = torch.div(dual_res, torch.max(torch.linalg.vector_norm(t2, dim=-1, ord=torch.inf), torch.linalg.vector_norm(t3, dim=-1, ord=torch.inf)).clamp(min=0))
    rho = torch.clamp(rho * torch.sqrt(numerator / denom), rho_min, rho_max)
    return primal_res, dual_res, rho

class _RELUQP:
    
    def __init__(self, batch, nx, nc):
        self.batch = batch
        self.nx = nx 
        self.nc = nc
        
        self.rho=0.1
        self.rho_min=1e-3
        self.rho_max=1e3
        self.adaptive_rho_tolerance=5
        self.setup_rhos(nc, batch)
        
        self.max_iter=1000
        self.eps_abs=1e-3
        self.check_interval=25
        self.sigma= 1e-6 * torch.eye(nx).unsqueeze(0).unsqueeze(1)
        # self.Ic = tensor_args.to_device(torch.eye(nc)).unsqueeze(0).unsqueeze(0).repeat(self.rhos_matrix.shape[0],batch,1,1)
        # self.IW = tensor_args.to_device(torch.eye(nx+nc+nc)).unsqueeze(0).unsqueeze(0).repeat(1, batch, 1, 1)
        self.W_ks = torch.zeros(len(self.rhos)+1, batch, nx+nc+nc, nx+nc+nc)
        self.W_ks[0, ...] = torch.eye(nx+nc+nc)
        self.W_ks[1:, :, -nc:, -nc:] = torch.eye(nc)
        self.W_ks[1:, :, -nc:, nx:-nc] = - self.rhos_matrix
        self.help_arange = torch.arange(batch).long()
        self.output = torch.zeros((batch, nx+nc+nc))
        return 

    def setup_rhos(self, nc, batch):
        """
        Setup rho values for ADMM
        """
        rhos = [self.rho]
        rho = self.rho/self.adaptive_rho_tolerance
        while rho >= self.rho_min:
            rhos.append(rho)
            rho = rho/self.adaptive_rho_tolerance
        rho = self.rho*self.adaptive_rho_tolerance
        while rho <= self.rho_max:
            rhos.append(rho)
            rho = rho*self.adaptive_rho_tolerance
        rhos.sort()

        # conver to torch tensor
        self.rhos = torch.tensor(rhos)
        self.rho_ind = torch.argmin(torch.abs(self.rhos - self.rho)).repeat(self.batch) # [b]
        self.rhos_matrix = self.rhos.view(-1, 1, 1, 1) * torch.eye(nc).unsqueeze(0).unsqueeze(1)    # [r, 1, nc, nc]
        self.rhos_inv_matrix =  (1 / self.rhos.view(-1, 1, 1, 1)) * torch.eye(nc).unsqueeze(0).unsqueeze(1) # [r, 1, nc, nc]
        return 
    
    @torch.no_grad()
    def solve(self, H, A, l, u):
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        
        self.output[:] = 0    # initialize with 0. If disabled, use the last solution as initialization.
        
        # # # NOTE torch.inverse is very bad and will lead to poor converge behavior
        # # K = torch.inverse(H + self.sigma + A.transpose(-1,-2) @ (self.rhos_matrix @ A))
        # K_inv = H + self.sigma + A.transpose(-1,-2) @ (self.rhos_matrix @ A)
        # K_ArA = torch.linalg.solve(K_inv, (self.sigma - A.transpose(-1,-2) @ (self.rhos_matrix @ A)))
        # K_AT = torch.linalg.solve(K_inv, A.transpose(-1, -2))

        # W_ks = torch.cat([
        #     torch.cat([ K_ArA ,           2 * K_AT @ self.rhos_matrix,            - K_AT], dim=-1),
        #     torch.cat([ A @ K_ArA + A,   2 * A @ K_AT @ self.rhos_matrix - self.Ic,  -A @ K_AT + self.rhos_inv_matrix], dim=-1),
        #     torch.cat([ self.rhos_matrix @ A,               -self.rhos_matrix,      self.Ic], dim=-1),
        # ], dim=-2)
        # W_ks = torch.cat([self.IW, W_ks], dim=0)

        rhosA = torch.matmul(self.rhos_matrix, A, out=self.W_ks[1:, :, -self.nc:, :self.nx]) # [r, b, nc, nx]
        ArA = torch.matmul(A.transpose(-1,-2), rhosA, out=self.W_ks[1:, :, self.nx:2*self.nx, self.nx:2*self.nx]) # TMP! [r, b, nx, nx]
        assert self.nx < self.nc
        nK_inv = torch.add(self.sigma + H, ArA, out=self.W_ks[1:, :, self.nx:2*self.nx, 2*self.nx:3*self.nx])   # TMP! [r, b, nx, nx]
        sig_ArA = torch.sub(self.sigma, ArA, out=self.W_ks[1:, :, self.nx:2*self.nx, self.nx:2*self.nx]) # TMP! overlap ArA [r, b, nx, nx]
        nK_ArA = torch.linalg.solve(nK_inv, sig_ArA, out=self.W_ks[1:, :, :self.nx, :self.nx])   # [r, b, nx, nx]
        nK_AT = torch.linalg.solve(nK_inv, A.transpose(-1, -2), out=self.W_ks[1:, :, :self.nx, -self.nc:])    # [r, b, nx, nc]
        K_AT_rhos = torch.matmul(nK_AT, self.rhos_matrix, out=self.W_ks[1:, :, :self.nx, self.nx:-self.nc])   # [r, b, nx, nc]
        K_AT_rhos.mul_(2.0)
        torch.sub(0.0, nK_AT, out=self.W_ks[1:, :, :self.nx, -self.nc:])
        torch.matmul(A, self.W_ks[1:, :, :self.nx, :], out=self.W_ks[1:, :, self.nx:-self.nc, :])
        self.W_ks[1:, :, self.nx:-self.nc, :self.nx].add_(A)
        self.W_ks[1:, :, self.nx:-self.nc, self.nx:-self.nc].sub_(self.W_ks[1:, :, -self.nc:, -self.nc:])
        self.W_ks[1:, :, self.nx:-self.nc, -self.nc:].add_(self.rhos_inv_matrix)
        # assert (W_ks - self.W_ks).abs().max() < 1e-5
        # import pdb;pdb.set_trace()
        
        rho_ind = self.rho_ind
        rho = self.rhos[rho_ind]

        for k in range(1, self.max_iter + 1):
            self.output = _RELU(input=self.output, W=self.W_ks, l=l, u=u, rho_ind=(rho_ind+1), help_ind=self.help_arange, nx=self.nx, nc=self.nc)
            # rho update
            if k % self.check_interval == 0:
                x, z, lam = self.output[:, :self.nx], self.output[:, self.nx:self.nx+self.nc], self.output[:, self.nx+self.nc:self.nx+2*self.nc]
                primal_res, dual_res, rho = compute_residuals(H.squeeze(0), A.squeeze(0), x, z, lam, rho, self.rho_min, self.rho_max)

                rho_larger = (rho > self.rhos[rho_ind] * self.adaptive_rho_tolerance) & (rho_ind < len(self.rhos) - 1) & (rho_ind > -1)
                rho_smaller = (rho < self.rhos[rho_ind] / self.adaptive_rho_tolerance) & (rho_ind > 0)
                rho_ind = rho_ind + rho_larger.int() - rho_smaller.int()

                converge_flag = (primal_res < self.eps_abs * (self.nc ** 0.5)) & (dual_res < self.eps_abs * (self.nx ** 0.5))
                rho_ind = torch.where(converge_flag, -1, rho_ind)
                if torch.all(converge_flag):
                    break

        return x

def torch_normalize_vector(v: torch.Tensor) -> torch.Tensor:
    return v / (torch.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def torch_normal_to_rot(
    axis_0, rot_base1=torch.Tensor([[0., 1., 0.]]), rot_base2=torch.Tensor([[0., 0., 1.]])
):
    proj_xy = torch.abs(torch.sum(axis_0 * rot_base1, axis=-1, keepdims=True))
    axis_1 = torch.where(proj_xy > 0.99, rot_base2, rot_base1)

    axis_1 = torch_normalize_vector(
        axis_1 - torch.sum(axis_1 * axis_0, axis=-1, keepdims=True) * axis_0
    )
    axis_2 = torch.cross(axis_0, axis_1, axis=-1)

    return torch.stack([axis_0, axis_1, axis_2], axis=-1)

# optimization target
class QPSingle(torch.autograd.Function):
    # on cpu
    @staticmethod
    def forward(ctx,
                points: torch.Tensor, # (N, 3)
                normals: torch.Tensor, # (N, 3)
                # no gravity
                # gravity: np.ndarray, # (6,)
                gravity_center: torch.Tensor, # (6,)
                G_matrix: torch.Tensor, # (M, 6 * N)
                E_matrix: torch.Tensor, # (N, 6, 6)
                h_matrix: torch.Tensor, # (M,)
                ):
        rot = torch_normal_to_rot(normals)
        axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]
        relative_pos = points - gravity_center[None]
        grasp_matrix = torch.zeros((points.shape[0], 6, 6))
        grasp_matrix[:, :3, 0] = grasp_matrix[:, 3:, 3] = axis_0
        grasp_matrix[:, :3, 1] = grasp_matrix[:, 3:, 4] = axis_1
        grasp_matrix[:, :3, 2] = grasp_matrix[:, 3:, 5] = axis_2
        grasp_matrix[:, 3:, 0] = torch.cross(relative_pos, axis_0, axis=-1)
        grasp_matrix[:, 3:, 1] = torch.cross(relative_pos, axis_1, axis=-1)
        grasp_matrix[:, 3:, 2] = torch.cross(relative_pos, axis_2, axis=-1)

        param2force = grasp_matrix @ E_matrix  # [n, 6, 6]
        flatten_param2force = param2force.permute(1, 0, 2).reshape(6, -1)
        
        P_matrix = flatten_param2force.T @ flatten_param2force
        
        G_batch = G_matrix.unsqueeze(0)
        P_batch = P_matrix.unsqueeze(0)
        h_batch = h_matrix.unsqueeze(0)
        l_batch = torch.zeros_like(h_batch)
        l_batch.fill_(-torch.inf)
        
        solver = _RELUQP(G_batch.shape[0], G_batch.shape[2], G_batch.shape[1])
        solution = solver.solve(H=P_batch, A=G_batch, l=l_batch, u=h_batch)[0]
        
        torch.save({
            'H': P_matrix,
            'A': G_matrix,
            'u': h_matrix,
            'l': l_batch[0],
            'x': solution
        }, 'qp_data.pt')
        
        # print(solution)
        # print(h_matrix - G_matrix @ solution)
          
        solution = solution.reshape(-1, 6)
        contact_wrenches = (param2force @ solution[..., None]).squeeze(
            axis=-1
        )  # [n, 6]
        f_param = (E_matrix @ solution[..., None]).squeeze(axis=-1)  # [n, 6]
        
        wrench_error = torch.linalg.norm(torch.sum(contact_wrenches, axis=0))
        wrench_res = torch.sum(contact_wrenches, dim=0)  # (6,)
        
        ctx.save_for_backward(relative_pos, f_param, contact_wrenches, wrench_res)
        
        return wrench_error
    
    @staticmethod
    def backward(ctx, grad_output):
        # gradient of squared wrench error * 0.5
        relative_pos, f_param, contact_wrenches, wrench_res = ctx.saved_tensors
        # grad to points
        grad_points = torch.cross(contact_wrenches[:, 3:], wrench_res[None, :3], dim=-1)
        # grad to normals
        grad_normals = (wrench_res[None, :3] - torch.cross(relative_pos, wrench_res[None, 3:], dim=-1)) * f_param[:, 0][:, None]
        return grad_points, grad_normals, None, None, None, None

class QPSingleSolver:
    def __init__(self, miu_coef):
        self.miu_coef = miu_coef
        self.num_contact = -1
        return
      
    def _build_constraint(self):
        """
        Build G matrix and h matrix for constraints Gx <= h,
        using soft contact model with pyramid discretization.

        """
        num_f_strength = self.num_contact * 6
        G_matrix = torch.zeros((num_f_strength + self.num_contact + 1, num_f_strength))
        h_matrix = torch.zeros((num_f_strength + self.num_contact + 1))

        # - force <= 0
        G_matrix[range(0, num_f_strength), range(0, num_f_strength)] = -1.0
        # - force <= - eps
        h_matrix[range(0, num_f_strength)] = - 1e-2

        # pressure <= 1
        for i in range(self.num_contact):
            G_matrix[num_f_strength + i, 6 * i : 6 * i + 6] = 1.0
        h_matrix[-self.num_contact - 1 : -1] = 1.0

        # - sum pressure <= -0.1
        G_matrix[-1, :] = -1.0
        # remove this constraint by commenting out the following line
        # h_matrix[-1] = -1.0

        # https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
        E_matrix = torch.zeros((self.num_contact, 6, 6))
        E_matrix[:, 0, :] = 1
        E_matrix[:, 1, 0] = E_matrix[:, 2, 2] = self.miu_coef[0]
        E_matrix[:, 1, 1] = E_matrix[:, 2, 3] = -self.miu_coef[0]
        E_matrix[:, 3, 4] = self.miu_coef[1]
        E_matrix[:, 3, 5] = -self.miu_coef[1]

        return G_matrix, h_matrix, E_matrix
    
    def set_num_contact(self, num_contact):
        if self.num_contact != num_contact:
            self.num_contact = num_contact
            self.G_matrix, self.h_matrix, self.E_matrix = self._build_constraint()
        self.num_contact = num_contact

    def solve(
        self,
        pos: torch.Tensor,  # (N, 3)
        normal: torch.Tensor,  # (N, 3)
        gravity_center: torch.Tensor
    ):
        self.set_num_contact(pos.shape[0])
        wrench_error = QPSingle.apply(
            pos,
            normal,
            gravity_center,
            self.G_matrix,
            self.E_matrix,
            self.h_matrix
        )
        return wrench_error

    
if __name__ == "__main__":
    # test on simple QP
    # min 1/2 x' * H * x + g' * x
    # s.t. l <= A * x <= u
    H = torch.tensor([[1., 0., 0], [0., 1., 0.], [0., 0., 1.]])
    # g = torch.tensor([-8.0, -3, -3], dtype=torch.double)
    A = torch.tensor([[1.0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    l = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf], dtype=torch.double)
    u = torch.tensor([-3.0, -3., -10.0, -10, -10])

    batch = 1
    H = H.unsqueeze(0).repeat(batch, 1, 1)
    A = A.unsqueeze(0).repeat(batch, 1, 1)
    u = u.unsqueeze(0).repeat(batch, 1)
    l = l.unsqueeze(0).repeat(batch, 1)
    import time 
    
    qp = _RELUQP(batch, H.shape[1], A.shape[1])
    
    for i in range(1):
        s = time.time()
        x = qp.solve(H=H, A=A, l=l, u=u)
        print(x)
        print(u - torch.einsum('...ij,...j->...i', A, x))
        print(torch.einsum('...ij,...j->...i', A, x) - l)
        print(torch.einsum('...ij,...j->...i', H, x))
    
    points = torch.Tensor(
       [[ 0.0010,  0.0388, -0.1391],
        [ 0.0392, -0.0075, -0.1312],
        [ 0.0052,  0.0443, -0.0932],
        [ 0.0455, -0.0012, -0.0868],
        [ 0.0043,  0.0482, -0.0476],
        [ 0.0479,  0.0063, -0.0416],
        [-0.0192, -0.0369, -0.1174],
        [ 0.0081, -0.0435, -0.0970],
        [-0.0322,  0.0244, -0.1269],
        [-0.0325,  0.0357, -0.0646],
        [-0.0325,  0.0357, -0.0646]])
    normals = torch.Tensor(
       [[-3.1000e-02, -9.6815e-01,  2.4843e-01],
        [-9.7108e-01,  1.9809e-01,  1.3332e-01],
        [-1.1990e-01, -9.8507e-01,  1.2352e-01],
        [-9.9194e-01,  2.9101e-02,  1.2332e-01],
        [-9.2472e-02, -9.9572e-01, -1.8404e-04],
        [-9.9119e-01, -1.3239e-01,  3.2141e-03],
        [ 4.5211e-01,  8.8273e-01,  1.2799e-01],
        [-1.7877e-01,  9.7648e-01,  1.2054e-01],
        [ 7.9798e-01, -5.8991e-01,  1.2338e-01],
        [ 6.7016e-01, -7.3852e-01,  7.4007e-02],
        [ 6.7016e-01, -7.3852e-01,  7.4007e-02]])
    center = torch.Tensor([6.8909566e-07, 1.3253137e-07, 3.7056338e-02])
    
    solver = QPSingleSolver([0.1, 0.])
    err = solver.solve(points, normals, center)
    print(err) # 0.0445