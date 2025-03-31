import torch

from dexonomy.util.torch_rot_util import torch_normal_to_rot


def build_constraint_batched(batch, num_points, device):
    num_f_strength = num_points * 6

    # Constraints: l <= Gx <= h. init l is -inf, init h is inf
    G_matrix = torch.zeros(
        (
            batch,
            num_f_strength + num_points + 1,
            num_f_strength,
        ),
        dtype=torch.float32,
        device=device,
    )
    l_matrix = (
        torch.zeros(
            (
                batch,
                num_f_strength + num_points + 1,
            ),
            dtype=torch.float32,
            device=device,
        )
        - torch.inf
    )
    h_matrix = (
        torch.zeros(
            (
                batch,
                num_f_strength + num_points + 1,
            ),
            dtype=torch.float32,
            device=device,
        )
        + torch.inf
    )

    # -force <= -0.0001
    select_ind1 = range(0, num_f_strength)
    G_matrix[:, select_ind1, select_ind1] = -1.0
    h_matrix[:, :num_f_strength] = 0

    # pressure <= 1
    for i in range(num_points):
        G_matrix[:, num_f_strength + i, 6 * i : 6 * i + 6] = 1
    h_matrix[:, num_f_strength:-1] = 1

    # -sum pressure <= -0.1
    G_matrix[:, -1, :] = -1.0
    h_matrix[:, -1] = -1.0

    return G_matrix, l_matrix, h_matrix


def build_target_batched(pos, normal, gravity, gravity_center, miu_coef):
    batch, num_points = pos.shape[:2]

    # https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
    E_matrix = torch.zeros(
        (batch, num_points, 6, 6), dtype=torch.float32, device=pos.device
    )
    E_matrix[..., 0, :] = 1
    E_matrix[..., 1, 0] = E_matrix[..., 2, 2] = miu_coef[0]
    E_matrix[..., 1, 1] = E_matrix[..., 2, 3] = -miu_coef[0]
    E_matrix[..., 3, 4] = miu_coef[1]
    E_matrix[..., 3, 5] = -miu_coef[1]

    rot = torch_normal_to_rot(normal)
    axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]
    relative_pos = pos - gravity_center.unsqueeze(-2)
    grasp_matrix = torch.zeros(
        (batch, num_points, 6, 6), dtype=torch.float32, device=pos.device
    )
    grasp_matrix[..., :3, 0] = grasp_matrix[..., 3:, 3] = axis_0
    grasp_matrix[..., :3, 1] = grasp_matrix[..., 3:, 4] = axis_1
    grasp_matrix[..., :3, 2] = grasp_matrix[..., 3:, 5] = axis_2
    grasp_matrix[..., 3:, 0] = torch.cross(relative_pos, axis_0, dim=-1)
    grasp_matrix[..., 3:, 1] = torch.cross(relative_pos, axis_1, dim=-1)
    grasp_matrix[..., 3:, 2] = torch.cross(relative_pos, axis_2, dim=-1)

    param2force = grasp_matrix @ E_matrix

    flatten_param2force = param2force.transpose(-3, -2).reshape(
        param2force.shape[0], 6, -1
    )  # [b, n, 6, 4] -> [b, 6, n, 4] -> [b, 6, 4n]
    P_matrix = flatten_param2force.transpose(-1, -2) @ flatten_param2force
    q_matrix = (gravity.unsqueeze(-2) @ flatten_param2force).squeeze(-2)
    return P_matrix, q_matrix, param2force


def get_qp_error_batched(pos, normal, gravity, gravity_center, miu_coef):
    batch, n_points = pos.shape[:-1]
    device = pos.device

    G_matrix, l_matrix, h_matrix = build_constraint_batched(batch, n_points, device)
    P_matrix, q_matrix, param2force = build_target_batched(
        pos, normal, gravity, gravity_center, miu_coef
    )

    solution = batched_reluqp(P_matrix, q_matrix, G_matrix, l_matrix, h_matrix)

    solution = solution.reshape(batch, n_points, 6)
    contact_wrenches = (param2force @ solution[..., None]).squeeze(-1)  # [b, n, 6]
    wrench_error = (contact_wrenches.sum(dim=1) + gravity).norm(dim=-1)

    return contact_wrenches, wrench_error


def batched_reluqp(P_matrix, q_matrix, G_matrix, l_matrix, h_matrix):
    solver = _RELUQP(
        G_matrix.shape[0], G_matrix.shape[2], G_matrix.shape[1], G_matrix.device
    )
    return solver.solve(P_matrix, q_matrix, G_matrix, l_matrix, h_matrix)


@torch.jit.script
def _RELU(input, W, b, l, u, rho_ind, help_ind, tmp, nx: int, nc: int):
    torch.bmm(W[rho_ind, help_ind], input, out=tmp)
    input[:] = tmp
    input[..., 0].add_(b[rho_ind, help_ind, :, 0])
    input[:, nx : nx + nc, 0].clamp_(l, u)
    return input


@torch.jit.script
def compute_residuals(H, A, g, x, z, lam, rho, rho_min: float, rho_max: float):
    t1 = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
    t2 = torch.matmul(H, x.unsqueeze(-1)).squeeze(-1)
    t3 = torch.matmul(A.transpose(-1, -2), lam.unsqueeze(-1)).squeeze(-1)
    primal_res = torch.linalg.vector_norm(t1 - z, dim=-1, ord=torch.inf)
    dual_res = torch.linalg.vector_norm(t2 + t3 + g, dim=-1, ord=torch.inf)
    numerator = torch.div(
        primal_res,
        torch.max(
            torch.linalg.vector_norm(t1, dim=-1, ord=torch.inf),
            torch.linalg.vector_norm(z, dim=-1, ord=torch.inf),
        ),
    )
    denom = torch.div(
        dual_res,
        torch.max(
            torch.max(
                torch.linalg.vector_norm(t2, dim=-1, ord=torch.inf),
                torch.linalg.vector_norm(t3, dim=-1, ord=torch.inf),
            ),
            torch.linalg.vector_norm(g, dim=-1, ord=torch.inf),
        ),
    )
    rho = torch.clamp(rho * torch.sqrt(numerator / denom), rho_min, rho_max)
    return primal_res, dual_res, rho


class _RELUQP:

    def __init__(self, batch, nx, nc, device):
        self.device = device
        self.batch = batch
        self.nx = nx
        self.nc = nc

        self.rho = 0.1
        self.rho_min = 1e-3
        self.rho_max = 1e3
        self.adaptive_rho_tolerance = 5
        self.setup_rhos(nc)

        self.max_iter = 1000
        self.eps_abs = 1e-3
        self.check_interval = 25
        self.sigma = 1e-6 * torch.eye(nx, dtype=torch.float32, device=device).unsqueeze(
            0
        ).unsqueeze(1)

        # NOTE: W_ks[0] and b_ks[0] is identity transformation designed for converged cases
        self.W_ks = torch.zeros(
            (len(self.rhos) + 1, batch, nx + nc + nc, nx + nc + nc),
            dtype=torch.float32,
            device=device,
        )
        self.W_ks[0, ...] = torch.eye(nx + nc + nc)
        self.W_ks[1:, :, -nc:, -nc:] = torch.eye(nc)
        self.W_ks[1:, :, -nc:, nx:-nc] = -self.rhos_matrix
        self.b_ks = torch.zeros(
            (len(self.rhos) + 1, batch, nx + nc + nc, 1),
            dtype=torch.float32,
            device=device,
        )

        self.help_arange = torch.arange(batch, device=device).long()
        self.output = torch.zeros(
            (batch, nx + nc + nc, 1), dtype=torch.float32, device=device
        )
        self.tmp_output = torch.zeros(
            (batch, nx + nc + nc, 1), dtype=torch.float32, device=device
        )
        return

    def setup_rhos(self, nc):
        """
        Setup rho values for ADMM
        """
        rhos = [self.rho]
        rho = self.rho / self.adaptive_rho_tolerance
        while rho >= self.rho_min:
            rhos.append(rho)
            rho = rho / self.adaptive_rho_tolerance
        rho = self.rho * self.adaptive_rho_tolerance
        while rho <= self.rho_max:
            rhos.append(rho)
            rho = rho * self.adaptive_rho_tolerance
        rhos.sort()

        # conver to torch tensor
        self.rhos = torch.tensor(rhos, dtype=torch.float32, device=self.device)
        self.rho_ind = torch.argmin(torch.abs(self.rhos - self.rho)).repeat(
            self.batch
        )  # [b]
        self.rhos_matrix = self.rhos.view(-1, 1, 1, 1) * torch.eye(
            nc, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(
            1
        )  # [r, 1, nc, nc]
        self.rhos_inv_matrix = (1 / self.rhos.view(-1, 1, 1, 1)) * torch.eye(
            nc, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(
            1
        )  # [r, 1, nc, nc]
        return

    @torch.no_grad()
    def solve(self, H, g, A, l, u):
        H = H.unsqueeze(0)
        g = g.unsqueeze(0)
        A = A.unsqueeze(0)

        self.output[:] = (
            0  # initialize with 0. If disabled, use the last solution as initialization.
        )

        rhosA = torch.matmul(
            self.rhos_matrix, A, out=self.W_ks[1:, :, -self.nc :, : self.nx]
        )  # [r, b, nc, nx]
        ArA = torch.matmul(
            A.transpose(-1, -2),
            rhosA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, self.nx : 2 * self.nx],
        )  # TMP! [r, b, nx, nx]
        assert self.nx < self.nc
        nK_inv = torch.add(
            self.sigma + H,
            ArA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, 2 * self.nx : 3 * self.nx],
        )  # TMP! [r, b, nx, nx]

        torch.linalg.solve(
            -nK_inv, g.unsqueeze(-1), out=self.b_ks[1:, :, : self.nx, :]
        )  # [r, b, nx, nx]
        torch.matmul(
            A,
            self.b_ks[1:, :, : self.nx, :],
            out=self.b_ks[1:, :, self.nx : -self.nc, :],
        )

        sig_ArA = torch.sub(
            self.sigma,
            ArA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, self.nx : 2 * self.nx],
        )  # TMP! overlap ArA [r, b, nx, nx]
        nK_ArA = torch.linalg.solve(
            nK_inv, sig_ArA, out=self.W_ks[1:, :, : self.nx, : self.nx]
        )  # [r, b, nx, nx]
        nK_AT = torch.linalg.solve(
            nK_inv, A.transpose(-1, -2), out=self.W_ks[1:, :, : self.nx, -self.nc :]
        )  # [r, b, nx, nc]
        K_AT_rhos = torch.matmul(
            nK_AT, self.rhos_matrix, out=self.W_ks[1:, :, : self.nx, self.nx : -self.nc]
        )  # [r, b, nx, nc]
        K_AT_rhos.mul_(2.0)
        torch.sub(0.0, nK_AT, out=self.W_ks[1:, :, : self.nx, -self.nc :])
        torch.matmul(
            A,
            self.W_ks[1:, :, : self.nx, :],
            out=self.W_ks[1:, :, self.nx : -self.nc, :],
        )
        self.W_ks[1:, :, self.nx : -self.nc, : self.nx].add_(A)
        self.W_ks[1:, :, self.nx : -self.nc, self.nx : -self.nc].sub_(
            self.W_ks[1:, :, -self.nc :, -self.nc :]
        )
        self.W_ks[1:, :, self.nx : -self.nc, -self.nc :].add_(self.rhos_inv_matrix)

        rho_ind = self.rho_ind
        rho = self.rhos[rho_ind]

        for k in range(1, self.max_iter + 1):
            self.output = _RELU(
                input=self.output,
                W=self.W_ks,
                b=self.b_ks,
                l=l,
                u=u,
                rho_ind=(rho_ind + 1),
                help_ind=self.help_arange,
                tmp=self.tmp_output,
                nx=self.nx,
                nc=self.nc,
            )
            # rho update
            if k % self.check_interval == 0:
                x, z, lam = (
                    self.output[:, : self.nx, 0],
                    self.output[:, self.nx : self.nx + self.nc, 0],
                    self.output[:, self.nx + self.nc : self.nx + 2 * self.nc, 0],
                )
                primal_res, dual_res, rho = compute_residuals(
                    H.squeeze(0),
                    A.squeeze(0),
                    g.squeeze(0),
                    x,
                    z,
                    lam,
                    rho,
                    self.rho_min,
                    self.rho_max,
                )

                rho_larger = (
                    (rho > self.rhos[rho_ind] * self.adaptive_rho_tolerance)
                    & (rho_ind < len(self.rhos) - 1)
                    & (rho_ind > -1)
                )
                rho_smaller = (
                    rho < self.rhos[rho_ind] / self.adaptive_rho_tolerance
                ) & (rho_ind > 0)
                rho_ind = rho_ind + rho_larger.int() - rho_smaller.int()

                converge_flag = (primal_res < self.eps_abs * (self.nc**0.5)) & (
                    dual_res < self.eps_abs * (self.nx**0.5)
                )
                rho_ind = torch.where(converge_flag, -1, rho_ind)
                if torch.all(converge_flag):
                    break

        return x


if __name__ == "__main__":
    # NOTE: this version doesn't have equality constraint enhancement
    # min 1/2 x' * H * x + g' * x
    # s.t. l <= A * x <= u
    device = "cuda:0"
    H = torch.tensor(
        [[6, 2, 1], [2, 5, 2], [1, 2, 4.0]], dtype=torch.float32, device=device
    )
    g = torch.tensor([-8, -3, -3], dtype=torch.float32, device=device)
    A = torch.tensor(
        [[1.0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    l = torch.tensor([3.0, 0, -10.0, -10, -10], dtype=torch.float32, device=device)
    u = torch.tensor(
        [3.0, 0, torch.inf, torch.inf, torch.inf], dtype=torch.float32, device=device
    )

    batch = 1000
    H = H.unsqueeze(0).repeat(batch, 1, 1)
    g = g.unsqueeze(0).repeat(batch, 1)
    A = A.unsqueeze(0).repeat(batch, 1, 1)
    u = u.unsqueeze(0).repeat(batch, 1)
    import time

    qp = _RELUQP(batch, H.shape[1], A.shape[1], device)

    for i in range(10):
        s = time.time()
        x = qp.solve(H=H, g=g, A=A, l=l, u=u)
        print(x)
        print(time.time() - s)
