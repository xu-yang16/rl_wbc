import time
import torch
from torch import nn, bmm
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv

from loguru import logger


class PdhgSolver(nn.Module):
    """
    Solve QP problem:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b \in K,
    where x in R^n, b in R^m.
    """

    def __init__(
        self,
        device: str,
        P: torch.Tensor,
        q: torch.Tensor,
        alpha=1,
        beta=1,
        contact_state=None,
    ):
        """
        Initialize the QP solver.

        device: PyTorch device
        P: Matrix P in the objective
        q: Coefficients in the objective
        alpha, beta: conditioning parameters
        """
        super().__init__()
        self.device = device
        self.num_envs = P.shape[0]
        self.n = P.shape[1]
        self.P = P
        self.q = q.unsqueeze(-1)
        self.alpha = alpha
        self.beta = beta
        self.contact_state = contact_state

        # Initialize the constraint
        self.m = 0
        self.H = torch.zeros((self.num_envs, 0, self.n), device=self.device)
        self.b = torch.zeros((self.num_envs, 0), device=self.device)
        self.contact_filter = torch.zeros((self.num_envs, 0), device=self.device)

        self.constraint = []

    def add_constraint(
        self,
        H: torch.Tensor,
        b: torch.Tensor,
        type="soc",
        a_min=None,
        a_max=None,
        contact_filter=None,
    ):
        """
        Add a second-order cone constraint.

        H: Constraint matrix
        b: Constraint vector
        """
        if type in ["soc", "hard_tanh", "relu"]:
            self.constraint.append(
                {
                    "z_start": self.m,
                    "z_dim": b.shape[1],
                    "type": type,
                    "a_min": a_min,
                    "a_max": a_max,
                }
            )
            self.m += b.shape[1]
            self.H = torch.cat([self.H, H], dim=1)
            self.b = torch.cat([self.b, b], dim=1)
            if contact_filter is not None:
                self.contact_filter = torch.cat(
                    [self.contact_filter, contact_filter],
                    dim=1,
                )
            else:
                self.contact_filter = torch.cat(
                    [
                        self.contact_filter,
                        torch.repeat_interleave(
                            self.contact_state, b.shape[1] // 4, dim=1
                        ),
                    ],
                    dim=1,
                )
        else:
            raise ValueError(f"Unsupported constraint type: {type}")

    def project(self, z: torch.Tensor):
        """
        Project the variable z onto the feasible set.

        z: z=Hx+b in the QP problem
        """
        for c in self.constraint:
            start_idx = c["z_start"]
            end_idx = start_idx + c["z_dim"]
            if c["type"] == "soc":
                project_onto_soc(z[:, start_idx:end_idx])
            elif c["type"] == "hard_tanh":
                F.hardtanh(
                    z[:, start_idx:end_idx],
                    min_val=c["a_min"],
                    max_val=c["a_max"],
                    inplace=True,
                )
            elif c["type"] == "relu":
                F.relu(
                    z[:, start_idx:end_idx],
                    inplace=True,
                )
        # filter out the contact forces that are not in contact
        z = z * self.contact_filter.unsqueeze(-1)
        return z

    def get_AB(self):
        """
        Given P, q, H, b, and the conditioning parameters, compute matrices A and B used in the PDHG iterations.
        A=[F beta * F; alpha*(I-2F) I-2*alpha*beta * F]
        B=[mu; -2*alpha**beta*mu]
        """
        I = torch.eye(self.m, device=self.device)
        F = I - self.H @ inv(
            self.P + self.H.transpose(-1, -2) @ self.H
        ) @ self.H.transpose(-1, -2)
        mu = bmm(
            F, bmm(self.H, solve(self.P, self.q)) - self.b
        )  # solve (0.13) < inv(0.15) < pinv(0.31)
        A = torch.cat(
            [
                torch.cat([F, self.beta * F], dim=2),
                torch.cat(
                    [
                        self.alpha * (I - 2 * F),
                        I - 2 * self.alpha * self.beta * F,
                    ],
                    dim=2,
                ),
            ],
            dim=1,
        )  # num_envs * 2m * 2m
        B = torch.cat(
            [self.beta * mu, -2 * self.alpha * self.beta * mu], dim=1
        )  # num_envs * 2m
        return A, B

    def forward(self, warm_lambda_z=None, iters=100, return_residuals=False):
        """
        Solve the QP problem using PDHG.

        warm_lambda_z: Initial guess for the variable (lambda, z)
        iters: Number of PDHG iterations
        return_residuals: Whether to return the residuals
        """
        if warm_lambda_z is None:
            lambda_z = torch.zeros((self.num_envs, 2 * self.m, 1), device=self.device)
        else:
            lambda_z = warm_lambda_z

        self.b = self.b.unsqueeze(-1)
        A, B = self.get_AB()
        for _ in range(iters):
            lambda_z = bmm(A, lambda_z) + B
            lambda_z[:, self.m :] = self.project(lambda_z[:, self.m :])
        # get primal solution
        x = bmm(pinv(self.H), lambda_z[:, self.m :] - self.b)
        if return_residuals:
            primal_residual, dual_residual = self.compute_residuals(
                x, lambda_z, lambda_z[:, : self.m]
            )
            return x, lambda_z, primal_residual, dual_residual
        else:
            return x, lambda_z, None, None

    def forward_each_iter(self, warm_lambda_z=None, iters=100, return_residuals=False):
        """
        Solve the QP problem using PDHG.

        warm_lambda_z: Initial guess for the variable (lambda, z)
        iters: Number of PDHG iterations
        return_residuals: Whether to return the residuals
        """
        if warm_lambda_z is None:
            lambda_z = torch.zeros((self.num_envs, 2 * self.m), device=self.device)
        else:
            lambda_z = warm_lambda_z
        A, B = self.get_AB()

        x_history = torch.zeros((iters, self.num_envs, self.n), device=self.device)
        lambda_z_history = torch.zeros(
            (iters, self.num_envs, 2 * self.m), device=self.device
        )
        primal_residual_history = torch.zeros(iters, self.num_envs, device=self.device)
        dual_residual_history = torch.zeros(iters, self.num_envs, device=self.device)
        for k in range(iters):
            lambda_z = bmm(A, lambda_z) + B
            projected_z = self.project(lambda_z[:, self.m :])
            lambda_z = torch.cat([lambda_z[:, : self.m], projected_z], dim=1)

            # get primal solution
            x = bmm(pinv(self.H), lambda_z[:, self.m :] - self.b)

            primal_residual, dual_residual = self.compute_residuals(
                x, lambda_z[:, self.m :], lambda_z[:, : self.m]
            )
            x_history[k] = x
            lambda_z_history[k] = lambda_z
            primal_residual_history[k] = primal_residual
            dual_residual_history[k] = dual_residual

        return (
            x_history,
            lambda_z_history,
            primal_residual_history,
            dual_residual_history,
        )

    def compute_residuals(self, x, z, lam):
        # Compute primal residual: Hx + b - z
        primal_residual = bmm(self.H, x) + self.b - z

        # Compute dual residual: Px + q - H'u
        dual_residual = bmm(self.P, x) + self.q - bmm(self.H.transpose(-1, -2), lam)

        return torch.linalg.norm(primal_residual, dim=1), torch.linalg.norm(
            dual_residual, dim=1
        )


@torch.jit.script
def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return torch.bmm(A, b.unsqueeze(-1)).squeeze(-1)


@torch.jit.script
def project_onto_soc(z_part: torch.Tensor):
    """
    Project the variable z onto the second-order cone.
    z_part: The part of z corresponding to the second-order cone
    """
    z_part *= z_part[:, 2:3] > 0

    norm_xy = torch.linalg.vector_norm(z_part[:, :2], dim=1) + 1e-9
    result = (z_part[:, 2] / norm_xy).unsqueeze(1)

    z_part_clone = z_part.clone()
    z_part_clone[:, 2] = norm_xy
    projected_z_part = 0.5 * (1 + result) * z_part_clone

    onto_condition = result < 1
    z_part = projected_z_part * onto_condition + z_part * ~onto_condition
    return z_part
