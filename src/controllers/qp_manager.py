import time
from typing import Tuple, Union

import numpy as np
import torch

# qpth
from qpth.qp import QPFunction, QPSolvers

# cvxpy
import cvxpy

# PDHG
from .solver.pdhg.qp_solver import PdhgSolver

from icecream import ic
from loguru import logger


class QPManager:
    # Solve QP problem:
    # minimize    (1/2)x'Px + q'x
    # subject to  Hx + b >= 0,
    # where x in R^n, b in R^m.
    # P: [num_envs, 12, 12], q: [num_envs, 12]
    # H: [num_envs, 28, 12], b: [num_envs, 28]
    def __init__(
        self,
        num_envs: int = 1,
        weight_ddq=np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 1.0]),
        Wr: float = 1e-4,
        Wt: float = 1e-4,
        foot_friction_coef: float = 0.7,
        max_z_force: float = 120.0,
        min_z_force: float = 0.0,
        max_joint_torque: float = 15.0,
        device: str = "cuda",
        iter: int = 20,
        warm_start: bool = True,
        friction_type: str = "cone",  # pyramid, cone
        solver_type="pdhg",  # pdhg, qpth
    ):
        self.device = device
        self.Wq = torch.tensor(weight_ddq, device=self.device, dtype=torch.float32)
        self.Wr = Wr
        self.Wt = Wt
        self.foot_friction_coef = foot_friction_coef
        self.max_z_force = max_z_force
        self.min_z_force = min_z_force
        self.max_joint_torque = max_joint_torque
        self.num_envs = num_envs

        self.iter = iter
        # for warm start
        self.warm_start = warm_start
        self.last_X = None

        # select solver
        self.solver_type = solver_type
        self.friction_type = friction_type
        self.update_solver()
        self.count = 1

    def update_solver(self):
        if self.solver_type == "pdhg" and self.friction_type == "cone":
            self.solver = self.pdhg_cone_solver
        elif self.solver_type == "pdhg" and self.friction_type == "pyramid":
            self.solver = self.pdhg_pyramid_solver
        elif self.solver_type == "qpth":
            self.solver = self.qpth_pyramid_solver
        elif self.solver_type == "baseline":
            self.solver = self.baseline_solver
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")

    def construct_objective_function(
        self,
        mass_mat: torch.tensor,  # num_envs x 12 x 12
        base_rot_mat_t,  # num_envs x 3 x 3
        desired_acc,  # num_envs x 6
        all_foot_jacobian,  # num_envs x 12 x 12
        dq: torch.tensor,  # num_envs x 12
        Wq: torch.tensor,  # 6
        Wf: float,
        Wt: float,
    ):
        """
        objective = |a_{actual}-a_{des}|_Q+|J^T f|_R+|(J \dot q)^T f|_T
        = |f|_{M^T Q M + J(r+t\dot q\qot q^T)J^T} + [-2M^T Q (a_{des}+g)]^T f
        return: P: num_envs x 12 x 12, q: num_envs x 12, g: num_envs x 6
        """
        # construct gravity vector
        g_in_base_frame = torch.zeros((self.num_envs, 6), device=self.device)
        g_in_base_frame[:, 2] = 9.81
        g_in_base_frame[:, :3] = torch.bmm(
            base_rot_mat_t, g_in_base_frame[:, :3, None]
        ).squeeze(2)
        # construct quad term
        Q = torch.zeros((self.num_envs, 6, 6), device=self.device) + Wq[None, :]
        MQ = torch.bmm(torch.transpose(mass_mat, 1, 2), Q)
        a = torch.bmm(MQ, mass_mat)  # M^T Q M
        Wf_mat = torch.eye(12, device=self.device) * Wf
        RT = Wt * torch.einsum("ij,ik->ijk", dq, dq) + Wf_mat[None, :]

        b = torch.bmm(
            torch.bmm(all_foot_jacobian, RT), all_foot_jacobian.transpose(1, 2)
        )

        quad_term = a + b
        # construct linear term
        linear_term = -torch.bmm(
            MQ, (desired_acc + g_in_base_frame)[:, :, None]
        ).squeeze(-1)
        return quad_term, linear_term, g_in_base_frame

    def solve_grf(
        self,
        mass_mat: torch.tensor,  # num_envs x 12 x 12
        desired_acc: torch.tensor,  # num_envs x 6
        base_rot_mat_t: torch.tensor,  # num_envs x 3 x 3
        all_foot_jacobian: torch.tensor,  # num_envs x 12 x 12
        dq: torch.tensor,  # num_envs x 12
        foot_contact_state: torch.tensor,  # num_envs x 4
    ):
        """
        Solve QP problem using PDHG solver
        return: grf: num_envs x 12
        """
        P, q, g_in_base_frame = self.construct_objective_function(
            mass_mat=mass_mat,
            base_rot_mat_t=base_rot_mat_t,
            desired_acc=desired_acc,
            all_foot_jacobian=all_foot_jacobian,
            dq=dq,
            Wq=self.Wq,
            Wf=self.Wr,
            Wt=self.Wt,
        )
        self.P, self.q = P, q
        self.base_rot_mat_t = base_rot_mat_t
        self.foot_contact_state = foot_contact_state
        self.all_foot_jacobian = all_foot_jacobian
        if not self.warm_start:
            self.last_X = None

        start_time = time.time()
        grf, self.last_X = self.solver(
            P=P,
            q=q,
            base_rot_mat_t=base_rot_mat_t,
            foot_contact_state=foot_contact_state,
            all_foot_jacobian=all_foot_jacobian,
            last_X=self.last_X,
            iter=self.iter,
        )
        solver_time = time.time() - start_time
        grf = grf.squeeze(-1)
        solved_acc = torch.bmm(mass_mat, grf[:, :, None]).squeeze(-1) + g_in_base_frame
        return grf, solved_acc, solver_time

    def pdhg_cone_solver(
        self,
        P: torch.tensor,
        q: torch.tensor,
        base_rot_mat_t: torch.tensor,
        foot_contact_state: torch.tensor,
        all_foot_jacobian: torch.tensor,
        last_X: Union[torch.tensor, None] = None,
        iter: int = 20,
    ):
        """
        Solve QP problem using PDHG solver
        Constraints: [friction cone 12, z force 4, joint torques 12]
        minimize    (1/2)x'Px + q'x
        subject to  Hx + b \in C,
        where x in R^n, b in R^m.

        P: [num_envs, 12, 12], q: [num_envs, 12]
        H: [num_envs, 28, 12], b: [num_envs, 28]
        return: grf [num_envs x 12]

        * friction cone: 4 soc constraints (for each H: 3*12)
        * z force: 4 hardtanh constraints
        * joint torques: 12 hardtanh constraints
        """
        solver = PdhgSolver(
            device=self.device, P=P, q=q, contact_state=foot_contact_state
        )

        normalized_friction_cone_in_world_frame = base_rot_mat_t.transpose(1, 2).clone()
        normalized_friction_cone_in_world_frame[:, :2, :] /= self.foot_friction_coef

        select_z_force_in_world_frame = normalized_friction_cone_in_world_frame[
            :, 2, :
        ]  # num_envs x 3
        for leg_id in range(4):
            # friction
            normalize_friction_cone_tensor = torch.zeros(
                (self.num_envs, 3, 12), device=self.device
            )
            normalize_friction_cone_tensor[:, :, (leg_id * 3) : (leg_id * 3 + 3)] = (
                normalized_friction_cone_in_world_frame  # num_envs x 3 x 3
            )
            solver.add_constraint(
                H=normalize_friction_cone_tensor,
                b=torch.zeros((self.num_envs, 3), device=self.device),
                type="soc",
                contact_filter=foot_contact_state[:, leg_id : leg_id + 1].repeat(1, 3),
            )

        # z force
        H = torch.kron(
            torch.eye(4, device=self.device), select_z_force_in_world_frame[:, None, :]
        ).reshape(self.num_envs, 4, 12)
        b = torch.zeros((self.num_envs, 4), device=self.device)
        solver.add_constraint(
            H=H, b=b, type="hard_tanh", a_min=self.min_z_force, a_max=self.max_z_force
        )  # only project for stance legs

        # joint torques
        JT = all_foot_jacobian.transpose(1, 2)  # num_envs x 12 x 12
        solver.add_constraint(
            H=JT,
            b=torch.zeros((self.num_envs, JT.shape[1]), device=self.device),
            type="hard_tanh",
            a_min=-self.max_joint_torque,
            a_max=self.max_joint_torque,
        )

        with torch.no_grad():
            primal_sols, X, *_ = solver.forward(warm_lambda_z=last_X, iters=iter)

        grf = primal_sols.squeeze(1)

        return grf, X

    def pdhg_pyramid_solver(
        self,
        P: torch.tensor,
        q: torch.tensor,
        base_rot_mat_t: torch.tensor,
        foot_contact_state: torch.tensor,
        all_foot_jacobian: torch.tensor,
        last_X: Union[torch.tensor, None] = None,
        iter: int = 20,
    ):
        """
        * friction pyramid: 16 relu constraints
        * z force: 4 hardtanh constraints
        * joint torques: 12 hardtanh constraints
        """
        solver = PdhgSolver(
            device=self.device, P=P, q=q, contact_state=foot_contact_state
        )
        # friction pyramid: 16 dim
        H_friction, b_friction = self.build_pyramid_constraint(base_rot_mat_t)
        solver.add_constraint(H=H_friction, b=b_friction, type="relu")

        # z force: 4 dim
        base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)  # num_envs x 3 x 3
        select_z_force_in_world_frame = base_rot_mat[:, 2, :].unsqueeze(
            1
        )  # num_envs x 1 x 3
        select_z_force_tensor = torch.kron(
            torch.eye(4, device=self.device), select_z_force_in_world_frame[:, None, :]
        ).reshape(self.num_envs, 4, 12)
        H_z_force = select_z_force_tensor
        b_z_force = torch.zeros((self.num_envs, 4), device=self.device)

        solver.add_constraint(
            H=H_z_force,
            b=b_z_force,
            type="hard_tanh",
            a_min=self.min_z_force,
            a_max=self.max_z_force,
        )  # only project for stance legs

        # joint torques: 12 dim
        JT = all_foot_jacobian.transpose(1, 2)  # num_envs x 12 x 12
        solver.add_constraint(
            H=JT,
            b=torch.zeros((self.num_envs, JT.shape[1]), device=self.device),
            type="hard_tanh",
            a_min=-self.max_joint_torque,
            a_max=self.max_joint_torque,
        )

        with torch.no_grad():
            primal_sols, X, *_ = solver.forward(warm_lambda_z=last_X, iters=iter)

        grf = primal_sols.squeeze(1)

        return grf, X

    # naive pdhg pyramid solver
    def qpth_pyramid_solver(
        self,
        P: torch.tensor,
        q: torch.tensor,
        base_rot_mat_t: torch.tensor,
        foot_contact_state: torch.tensor,
        all_foot_jacobian: torch.tensor,
        last_X: Union[torch.tensor, None] = None,
        iter: int = 500,
    ):
        """
        min 1/2 x'Px + q'x
        s.t. Ax - b = 0
             Gx - h <= 0
        """
        # Note that Hx + b >= 0 is equivalent to (-Hx) - b <= 0
        H_friction, b_friction = self.build_pyramid_constraint(base_rot_mat_t)
        H_z_force, b_z_force = self.build_z_force_constraint(
            base_rot_mat_t, foot_contact_state
        )
        H_joint_torque, b_joint_torque = self.build_joint_torque_constraint(
            all_foot_jacobian
        )
        H = torch.cat([H_friction, H_z_force, H_joint_torque], dim=1)
        b = torch.cat([b_friction, b_z_force, b_joint_torque], dim=1)

        G = -H
        h = b

        # Define QP function
        qf = QPFunction(
            verbose=-1,
            check_Q_spd=False,
            eps=1e-10,
            solver=QPSolvers.PDIPM_BATCHED,
            maxIter=iter,
        )
        # solve QP
        e = torch.autograd.Variable(torch.Tensor())
        grf = qf(P.double(), q.double(), G.double(), h.double(), e, e)
        grf = grf.float()

        return grf, None  # , qpth_violation, qp_cost, solver_time

    # build constraints: Hx + b >= 0
    def build_pyramid_constraint(self, base_rot_mat_t: torch.tensor):
        # constraint for friction pyramid
        H_friction = torch.zeros((self.num_envs, 16, 12), device=self.device)
        b_friction = torch.zeros((self.num_envs, 16), device=self.device)
        friction_pyramid = (
            torch.tensor(
                [
                    [1, 0, self.foot_friction_coef],
                    [-1, 0, self.foot_friction_coef],
                    [0, 1, self.foot_friction_coef],
                    [0, -1, self.foot_friction_coef],
                ],
                device=self.device,
            )
            .unsqueeze(0)
            .expand(self.num_envs, 4, 3)
        )  # num_envs x 4 x 3
        base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)  # num_envs x 3 x 3
        H_friction_part = torch.bmm(friction_pyramid, base_rot_mat)

        H_friction = torch.kron(torch.eye(4, device=self.device), H_friction_part)
        return H_friction, b_friction

    # build constraints: Hx + b >= 0
    def build_z_force_constraint(
        self, base_rot_mat_t: torch.tensor, foot_contact_state: torch.tensor
    ):
        # constraint for z force
        H_z_force = torch.zeros((self.num_envs, 8, 12), device=self.device)
        b_z_force = torch.zeros((self.num_envs, 8), device=self.device)
        z_force_select = torch.tensor(
            [[0, 0, 1], [0, 0, -1]], dtype=torch.float32, device=self.device
        )  # 2 x 3
        base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)  # num_envs x 3 x 3
        z_force_part = torch.bmm(
            z_force_select.unsqueeze(0).repeat(self.num_envs, 1, 1), base_rot_mat
        )  # num_envs x 2 x 3
        for leg_id in range(4):
            H_z_force[:, leg_id * 2 : leg_id * 2 + 2, leg_id * 3 : leg_id * 3 + 3] = (
                z_force_part
            )

        contact_ids = foot_contact_state.nonzero()
        b_z_force[contact_ids[:, 0], contact_ids[:, 1] * 2] = -self.min_z_force
        b_z_force[contact_ids[:, 0], contact_ids[:, 1] * 2 + 1] = self.max_z_force

        return H_z_force, b_z_force

    # build constraints: Hx + b >= 0
    def build_joint_torque_constraint(self, all_foot_jacobian: torch.tensor):
        # constraint for joint torques
        JT = -all_foot_jacobian.transpose(1, 2)

        H_joint_torque = torch.cat((JT, -JT), dim=1)
        b_joint_torque = (
            torch.tensor([self.max_joint_torque] * 24, device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )
        return H_joint_torque, b_joint_torque

    def baseline_solver(
        self,
        P: torch.tensor,
        q: torch.tensor,
        base_rot_mat_t: torch.tensor,
        foot_contact_state: torch.tensor,
        all_foot_jacobian: torch.tensor,
        last_X: Union[torch.tensor, None] = None,
        iter: int = 20,
    ):
        # Note that Hx + b >= 0 is equivalent to (-Hx) - b <= 0
        start_time = time.time()
        H_friction, b_friction = self.build_pyramid_constraint(base_rot_mat_t)
        H_z_force, b_z_force = self.build_z_force_constraint(
            base_rot_mat_t, foot_contact_state
        )
        H_joint_torque, b_joint_torque = self.build_joint_torque_constraint(
            all_foot_jacobian
        )
        H = torch.cat([H_friction, H_z_force, H_joint_torque], dim=1)
        b = torch.cat([b_friction, b_z_force, b_joint_torque], dim=1)

        P_cpu_np = P.squeeze(0).detach().cpu().numpy()
        q_cpu_np = q.squeeze(0).detach().cpu().numpy()
        H_cpu_np = H.squeeze(0).detach().cpu().numpy()
        b_cpu_np = b.squeeze(0).detach().cpu().numpy()

        import cvxpy

        x = cvxpy.Variable(len(q_cpu_np))
        objective = (
            0.5 * cvxpy.quad_form(x, 0.5 * (P_cpu_np + P_cpu_np.T)) + q_cpu_np.T @ x
        )
        constraints = [
            H_cpu_np @ x + b_cpu_np >= 0
        ]  # Define the constraints using H and b
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        # Solve the problem using OSQP via CVXPY
        problem.solve(solver=cvxpy.OSQP)
        solver_time = time.time() - start_time

        # check constraints
        cvxpy_violation = np.sum(
            np.maximum(-H_cpu_np @ x.value - b_cpu_np, np.zeros_like(b_cpu_np))
        )
        cvxpy_cost = (
            0.5 * x.value.T @ (0.5 * (P_cpu_np + P_cpu_np.T)) @ x.value
            + q_cpu_np.T @ x.value
        )

        return (
            x.value,
            None,
            cvxpy_violation,
            cvxpy_cost,
            solver_time,
        )

    def bench_pdhg_pyramid(
        self,
        P: torch.tensor,
        q: torch.tensor,
        base_rot_mat_t: torch.tensor,
        foot_contact_state: torch.tensor,
        all_foot_jacobian: torch.tensor,
        last_X: Union[torch.tensor, None] = None,
        iter: int = 20,
    ):
        solver = PdhgSolver(
            device=self.device, P=P, q=q, contact_state=foot_contact_state
        )
        # friction pyramid
        H_friction, b_friction = self.build_pyramid_constraint(base_rot_mat_t)
        solver.add_constraint(H=H_friction, b=b_friction, type="relu")

        # z force
        base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)  # num_envs x 3 x 3
        select_z_force_in_world_frame = base_rot_mat[:, 2, :].unsqueeze(
            1
        )  # num_envs x 1 x 3
        select_z_force_tensor = torch.kron(
            torch.eye(4, device=self.device), select_z_force_in_world_frame[:, None, :]
        ).reshape(self.num_envs, 4, 12)
        H_z_force = select_z_force_tensor
        b_z_force = torch.zeros((self.num_envs, 4), device=self.device)

        solver.add_constraint(
            H=H_z_force,
            b=b_z_force,
            type="hard_tanh",
            a_min=self.min_z_force,
            a_max=self.max_z_force,
        )  # only project for stance legs

        # joint torques
        JT = all_foot_jacobian.transpose(1, 2)  # num_envs x 12 x 12
        solver.add_constraint(
            H=JT,
            b=torch.zeros((self.num_envs, JT.shape[1]), device=self.device),
            type="hard_tanh",
            a_min=-self.max_joint_torque,
            a_max=self.max_joint_torque,
        )

        with torch.no_grad():
            (
                x_history,
                lambda_z_history,
                primal_residual_history,
                dual_residual_history,
            ) = solver.forward_each_iter(warm_lambda_z=last_X, iters=iter)

        return (
            x_history,
            lambda_z_history,
            primal_residual_history,
            dual_residual_history,
        )
