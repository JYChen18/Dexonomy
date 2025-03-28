import os
import sys
import numpy as np
from qpsolvers import solve_qp
import scipy

from dexonomy.util.np_rot_util import np_normalize_vector, np_normal_to_rot


class ContactQP:
    def __init__(self, miu_coef, solver_type="clarabel"):
        self.miu_coef = miu_coef
        self.solver_type = solver_type
        self.num_contact = -1
        return

    def build_constraint(self):
        """
        Build G matrix and h matrix for constraints Gx <= h,
        using soft contact model with pyramid discretization.

        """
        num_f_strength = self.num_contact * 6
        G_matrix = np.zeros((num_f_strength + self.num_contact + 1, num_f_strength))
        h_matrix = np.zeros((num_f_strength + self.num_contact + 1))

        # - force <= 0
        G_matrix[range(0, num_f_strength), range(0, num_f_strength)] = -1.0

        # pressure <= 1
        for i in range(self.num_contact):
            G_matrix[num_f_strength + i, 6 * i : 6 * i + 6] = 1.0
        h_matrix[-self.num_contact - 1 : -1] = 1.0

        # - sum pressure <= -0.1
        G_matrix[-1, :] = -1.0
        h_matrix[-1] = -1.0

        # https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
        E_matrix = np.zeros((self.num_contact, 6, 6))
        E_matrix[:, 0, :] = 1
        E_matrix[:, 1, 0] = E_matrix[:, 2, 2] = self.miu_coef[0]
        E_matrix[:, 1, 1] = E_matrix[:, 2, 3] = -self.miu_coef[0]
        E_matrix[:, 3, 4] = self.miu_coef[1]
        E_matrix[:, 3, 5] = -self.miu_coef[1]

        return G_matrix, h_matrix, E_matrix

    def solve(
        self,
        pos,
        normal,
        gravity,
        gravity_center,
        retract_force=None,
        retract_weight=1.0,
    ):
        """
        Parameters
        -------------------
        pos: np.array [n, 3]. If n is different from self.num_contact, update self.num_contact=n and initialize again.
        normal: np.array [n, 3]. Direction is from the object to hand
        gravity: np.array [6]
        gravity_center: np.array [6]
        retract_force: np.array [n, 3]. The solved force part of contact_wrenches should be close to it.
        retract_weight: float.

        Returns
        -------------------
        contact_wrenches: np.array [n, 6]
        wrench_error: np.array [6]
        """
        if pos.shape[0] != self.num_contact:
            self.num_contact = pos.shape[0]
            self.G_matrix, self.h_matrix, self.E_matrix = self.build_constraint()

        rot = np_normal_to_rot(normal)
        axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]
        # TODO: Do we need obb length to balance force and torque here?
        relative_pos = pos - gravity_center[None]
        grasp_matrix = np.zeros((self.num_contact, 6, 6))
        grasp_matrix[:, :3, 0] = grasp_matrix[:, 3:, 3] = axis_0
        grasp_matrix[:, :3, 1] = grasp_matrix[:, 3:, 4] = axis_1
        grasp_matrix[:, :3, 2] = grasp_matrix[:, 3:, 5] = axis_2
        grasp_matrix[:, 3:, 0] = np.cross(relative_pos, axis_0, axis=-1)
        grasp_matrix[:, 3:, 1] = np.cross(relative_pos, axis_1, axis=-1)
        grasp_matrix[:, 3:, 2] = np.cross(relative_pos, axis_2, axis=-1)

        param2force = grasp_matrix @ self.E_matrix  # [n, 6, 6]

        # [n, 6, 6] -> [6, n, 6] -> [6, 6n]
        flatten_param2force = np.transpose(param2force, (1, 0, 2)).reshape(6, -1)

        if retract_force is None:
            P_matrix = flatten_param2force.T @ flatten_param2force
            q_matrix = gravity @ flatten_param2force
        else:
            semi_P_matrix = np.zeros((3 * self.num_contact + 6, 6 * self.num_contact))
            for i in range(self.num_contact):
                semi_P_matrix[3 * i : 3 * i + 3, 6 * i : 6 * i + 6] = (
                    retract_weight
                ) * param2force[i, :3, :]
            semi_P_matrix[-6:, :] = flatten_param2force

            q_part = np.concatenate(
                [-retract_weight * retract_force.reshape(-1), gravity]
            )

            P_matrix = semi_P_matrix.T @ semi_P_matrix
            q_matrix = q_part @ semi_P_matrix

        # Minimize_x 1/2*x^T @ P_matrix @ x + q_matrix^T @ x
        # Subject to G_matrix @ x <= h_matrix
        # print(
        #     np.linalg.norm(P_matrix),
        #     np.linalg.norm(q_matrix),
        #     np.linalg.norm(self.G_matrix),
        #     np.linalg.norm(self.h_matrix),
        # )
        # print(
        #     np.sum(P_matrix),
        #     np.sum(q_matrix),
        #     np.sum(self.G_matrix),
        #     np.sum(self.h_matrix),
        # )
        solution = solve_qp(
            P=scipy.sparse.csc.csc_matrix(P_matrix),
            q=q_matrix,
            G=scipy.sparse.csc.csc_matrix(self.G_matrix),
            h=self.h_matrix,
            solver=self.solver_type,
        )
        # print(solution)
        if solution is None:
            return None, 1.0

        solution = solution.reshape(-1, 6)
        contact_wrenches = (param2force @ solution[..., None]).squeeze(
            axis=-1
        )  # [n, 6]
        wrench_error = np.linalg.norm(np.sum(contact_wrenches, axis=0) + gravity)
        return contact_wrenches, wrench_error


if __name__ == "__main__":

    miu_coef = [0.01, 0.002]
    contactqp = ContactQP(miu_coef)

    for i in range(10):
        print(i, "#" * 20)
        # pos = np.random.rand(num_contact, 3)
        # normal = np_normalize_vector(np.random.rand(num_contact, 3))
        pos = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        normal = np_normalize_vector(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        gravity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gravity_center = np.array([0.0, 0, 0.0])

        contact_wrenches, wrench_error = contactqp.solve(
            pos,
            normal,
            gravity,
            gravity_center,
            # retract_force=old_contact_wrenches[:, :3],
            # retract_weight=1.0,
        )
        print(contact_wrenches)
        print("error", wrench_error.mean())
