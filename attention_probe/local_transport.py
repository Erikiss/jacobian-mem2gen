"""Matrix-free Tikhonov transport for one fixed local residual-map Jacobian.

For J = dF(x0)/dx, solve (J J^T + lambda^2 I) u = delta_target with
bounded conjugate gradients, then return delta_source = J^T u. JVP/VJP must
refer to the SAME native map and linearization point (e.g. context-local
post-block14 -> post-block18), not an averaged/global Jacobian lens.

The regularization principle matches Burny's jtransport/transport.py::ridge_inverse
in jacobian-mem2gen. Reference: P. C. Hansen, *Rank-Deficient and Discrete
Ill-Posed Problems*, SIAM, 1998, ch.4; https://doi.org/10.1137/1.9780898719697.
Unlike that SVD implementation this module forms no full Jacobian or inverse,
estimates no singular values, and makes no sigma_min/conditioning guarantee.

lambda_ is an explicit positive absolute value. A caller may prespecify
lambda_ = c * estimated_gain and must record c and the gain-estimation rule;
this solver never chooses damping from answers or convergence outcomes.
Convergence refers to the damped dual equation, not exact target reconstruction
or nonlinear endpoint agreement. The latter require separate measurements.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch

MAX_CG_ITERATIONS = 16


@dataclass(frozen=True)
class TransportResult:
    """Detached native-dtype source displacement plus finite JSON diagnostics."""

    delta_source: torch.Tensor
    diagnostics: dict[str, Any]


def _positive_real(value: Real, name: str, *, below_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be an explicit real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (below_one and result >= 1):
        raise ValueError(
            f"{name} must be finite, positive" + (" and below one" if below_one else "")
        )
    return result


def _norm64(value: torch.Tensor) -> float:
    """Scale first, so tiny finite FP32 displacements do not square to zero."""
    flat = value.detach().reshape(-1).to(torch.float64)
    scale = float(flat.abs().max().item())
    if scale == 0:
        return 0.0
    result = float(torch.linalg.vector_norm(flat / scale).item()) * scale
    if not math.isfinite(result):
        raise FloatingPointError("Vector norm exceeds finite FP64 diagnostic range")
    return result


def ridge_transport(
    jvp: Callable[[torch.Tensor], torch.Tensor],
    vjp: Callable[[torch.Tensor], torch.Tensor],
    delta_target: torch.Tensor,
    lambda_: float,
    *,
    max_iter: int = MAX_CG_ITERATIONS,
    rtol: float = 1e-5,
) -> TransportResult:
    """Return J^T (J J^T + lambda_^2 I)^-1 delta_target approximately.

    The right-hand side is internally normalized, then the solution is rescaled;
    this changes no mathematical objective and avoids tiny-RHS CG underflow.
    Operator vectors retain the target's FP32/FP64 dtype/device and exact shape;
    scalar reductions use FP64. Source and target may have different shapes.
    Callables receive detached copies and must return finite native tensors.

    max_iter is in [1,16]; there are at most max_iter dual CG products plus one
    final VJP and one reconstruction JVP. JVP(0) is skipped analytically. A zero
    target performs no CG/JVP and one VJP(0) to discover the source shape.
    Nonconvergence/breakdown returns the last finite iterate, explicitly flagged.
    Invalid/nonfinite callback data raises instead of publishing NaN tensors.
    """
    if not callable(jvp) or not callable(vjp):
        raise TypeError("jvp and vjp must be callables bound to one local Jacobian")
    if not isinstance(delta_target, torch.Tensor):
        raise TypeError("delta_target must be a torch.Tensor")
    if delta_target.dtype not in (torch.float32, torch.float64):
        raise ValueError("delta_target must use native FP32 or FP64 real arithmetic")
    if delta_target.numel() == 0 or not bool(torch.isfinite(delta_target).all().item()):
        raise ValueError("delta_target must be nonempty and finite")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer")
    if not 1 <= max_iter <= MAX_CG_ITERATIONS:
        raise ValueError("max_iter must be between 1 and16")
    lam = _positive_real(lambda_, "lambda_")
    tolerance = _positive_real(rtol, "rtol", below_one=True)
    squared = lam * lam
    if not math.isfinite(squared):
        raise ValueError("lambda_^2 must be finite")
    y = delta_target.detach().clone()
    damping = torch.tensor(squared, dtype=y.dtype, device=y.device)
    if not bool(torch.isfinite(damping).item()) or float(damping.item()) <= 0:
        raise ValueError("lambda_^2 overflows or underflows the native dtype")
    damping_value = float(damping.item())
    target_norm = _norm64(y)
    source_shape: tuple[int, ...] | None = None
    counts = {
        "jvp_calls": 0,
        "vjp_calls": 0,
        "cg_operator_calls": 0,
        "zero_jvp_shortcuts": 0,
    }

    def checked(value, name, shape=None):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must return a torch.Tensor")
        if value.dtype != y.dtype or value.device != y.device:
            raise ValueError(f"{name} must preserve native dtype/device")
        if value.numel() == 0 or (shape is not None and tuple(value.shape) != shape):
            raise ValueError(f"{name} returned an unexpected shape")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} returned nonfinite values")
        return value.detach().clone()

    def transpose(vector):
        nonlocal source_shape
        counts["vjp_calls"] += 1
        value = checked(vjp(vector.detach().clone()), "vjp", source_shape)
        if source_shape is None:
            source_shape = tuple(value.shape)
        return value

    def forward(vector):
        if not bool(torch.count_nonzero(vector).item()):
            counts["zero_jvp_shortcuts"] += 1
            return torch.zeros_like(y)
        counts["jvp_calls"] += 1
        return checked(jvp(vector.detach().clone()), "jvp", tuple(y.shape))

    def dual_product(vector):
        counts["cg_operator_calls"] += 1
        return checked(
            forward(transpose(vector)) + damping * vector,
            "dual operator",
            tuple(y.shape),
        )

    def inner(left, right):
        value = float(
            torch.dot(left.reshape(-1).double(), right.reshape(-1).double()).item()
        )
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite FP64 CG reduction")
        return value

    def finish(solution, iterations, termination, history, *, zero_target=False):
        if zero_target:
            source = transpose(torch.zeros_like(y))
            if bool(torch.count_nonzero(source).item()):
                raise ValueError("vjp(0) must be zero for a fixed linearized operator")
            delta = torch.zeros_like(source)
            reconstruction = torch.zeros_like(y)
            dual64 = torch.zeros_like(y, dtype=torch.float64)
        else:
            source = transpose(solution)
            delta = checked(
                (source.double() * target_norm).to(y.dtype), "rescaled source"
            )
            reconstruction = forward(delta)
            dual64 = solution.double() * target_norm
        reconstruction_error = reconstruction.double() - y.double()
        dual_residual = y.double() - reconstruction.double() - damping_value * dual64
        residual_norm, reconstruction_norm = (
            _norm64(dual_residual),
            _norm64(reconstruction_error),
        )
        residual_relative = residual_norm / target_norm if target_norm else 0.0
        reconstruction_relative = (
            reconstruction_norm / target_norm if target_norm else 0.0
        )
        source_norm = _norm64(delta)
        normalized_objective = (
            (
                reconstruction_relative**2
                + damping_value * (source_norm / target_norm) ** 2
            )
            if target_norm
            else 0.0
        )
        if not all(
            math.isfinite(v)
            for v in (residual_relative, reconstruction_relative, normalized_objective)
        ):
            raise FloatingPointError("Nonfinite final transport diagnostics")
        converged = residual_relative <= tolerance
        breakdown = (
            termination
            if termination
            in ("nonpositive_curvature", "nonfinite_step", "nonfinite_iterate")
            else None
        )
        if breakdown:
            status = "breakdown"
            converged = False
        elif converged:
            status = "converged"
        else:
            status = "not_converged"
            if termination == "recursive_tolerance":
                termination = "recursive_tolerance_not_confirmed"
        diagnostics = {
            "schema": "local_tikhonov_transport_v1",
            "method": "matrix_free_dual_conjugate_gradients",
            "status": status,
            "converged": converged,
            "termination": termination,
            "breakdown": breakdown,
            "iterations": iterations,
            "max_iter": int(max_iter),
            "rtol": tolerance,
            "lambda": lam,
            "lambda_squared_native": damping_value,
            "target_norm": target_norm,
            "source_norm": source_norm,
            "dual_solution_norm": _norm64(dual64),
            "dual_residual_norm": residual_norm,
            "dual_relative_residual": residual_relative,
            "linear_reconstruction_error_norm": reconstruction_norm,
            "linear_reconstruction_relative_error": reconstruction_relative,
            "normalized_tikhonov_objective": normalized_objective,
            "recursive_relative_residual_history": history,
            "rhs_normalization": "unit L2 internally, final solution rescaled",
            "zero_target": zero_target,
            "target_shape": list(y.shape),
            "source_shape": list(delta.shape),
            "dtype": str(y.dtype),
            "device": str(y.device),
            **counts,
            "full_jacobian_formed": False,
            "singular_values_computed": False,
            "convergence_scope": "damped local linear dual equation; not nonlinear endpoint agreement",
        }
        return TransportResult(delta.detach(), diagnostics)

    if target_norm == 0:
        return finish(torch.zeros_like(y), 0, "zero_target", [0.0], zero_target=True)
    rhs = (y.double() / target_norm).to(y.dtype)
    rhs_norm = _norm64(rhs)
    solution, residual = torch.zeros_like(rhs), rhs.clone()
    direction = residual.clone()
    rr = inner(residual, residual)
    history = [1.0]
    iterations, termination = 0, "iteration_limit"
    for _ in range(int(max_iter)):
        product = dual_product(direction)
        curvature = inner(direction, product)
        if curvature <= 0:
            termination = "nonpositive_curvature"
            break
        alpha = rr / curvature
        if not math.isfinite(alpha):
            termination = "nonfinite_step"
            break
        next_solution = solution + alpha * direction
        next_residual = residual - alpha * product
        if not bool(torch.isfinite(next_solution).all().item()) or not bool(
            torch.isfinite(next_residual).all().item()
        ):
            termination = "nonfinite_iterate"
            break
        next_rr = inner(next_residual, next_residual)
        solution, residual = next_solution, next_residual
        iterations += 1
        relative = math.sqrt(next_rr) / rhs_norm
        history.append(relative)
        if relative <= tolerance:
            termination = "recursive_tolerance"
            break
        beta = next_rr / rr
        direction = residual + beta * direction
        if not bool(torch.isfinite(direction).all().item()):
            termination = "nonfinite_iterate"
            break
        rr = next_rr
    return finish(solution, iterations, termination, history)
