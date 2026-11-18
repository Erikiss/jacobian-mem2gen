"""Bounded refinement of a fixed, damped, matrix-free local transport solve.

Solve (J J.T + lambda**2 I) y = b, then return x = J.T y.  This module
never changes damping, forms a Jacobian, or selects an answer-dependent arm.
Convergence concerns the damped dual equation, not exact reconstruction of b.

The callbacks must describe one fixed linear map and its adjoint. Vectors use
native FP32/FP64; reductions and the final residual diagnostic use FP64. A
recursive CG tolerance crossing is only a candidate: the true residual of the
native, rescaled iterate must confirm it. Reliable residual replacement restarts
CG if recurrence drift exceeds 10% or a candidate convergence is false.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from .local_transport import _norm64, _positive_real

MAX_REFINEMENT_ITERATIONS = 512


@dataclass(frozen=True)
class TransportRefinementResult:
    """Detached native source/dual vectors and JSON-safe measured diagnostics."""

    delta_source: torch.Tensor
    dual_solution: torch.Tensor
    diagnostics: dict[str, Any]


def refine_transport(
    jvp: Callable[[torch.Tensor], torch.Tensor],
    vjp: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    lambda_: float,
    *,
    max_iter: int = MAX_REFINEMENT_ITERATIONS,
    rtol: float = 1e-4,
    stop_requested: Callable[[], bool] | None = None,
    true_residual_interval: int = 16,
) -> TransportRefinementResult:
    """Refine a fixed Tikhonov problem without silently accepting false CG stops.

    ``stop_requested`` is a cooperative deadline/cancellation callback, checked
    before each iteration. It cannot preempt an executing operator. Finalizing
    a stopped iterate may require one VJP and one JVP after that callback fires;
    this bounded accounting cost supplies the actual final residual. The result
    records ``budget_exhausted`` even if that last residual happens to pass.

    Internally the RHS has unit norm. At each true check the dual iterate is
    rescaled and rounded to native dtype, x=VJP(y) and JVP(x) are evaluated, and
    b-JVP(x)-lambda_native_squared*y is reduced in FP64. Returned y and x are
    exactly these checked native tensors. No unrounded surrogate is certified.
    If needed, a restart uses this true residual divided by the original RHS
    norm. This handles the original-RHS rounding introduced by normalization.

    Invalid/nonfinite callback tensors raise. Finite iteration limits, requested
    stops and nonpositive curvature return a flagged last finite iterate. There
    are at most max_iter CG products plus max_iter+1 true residual evaluations.
    """
    if not callable(jvp) or not callable(vjp):
        raise TypeError("jvp and vjp must be callables for one fixed local map")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if target.dtype not in (torch.float32, torch.float64):
        raise ValueError("target must use native FP32 or FP64")
    if target.numel() == 0 or not bool(torch.isfinite(target).all().item()):
        raise ValueError("target must be nonempty and finite")
    for value, name in (
        (max_iter, "max_iter"),
        (true_residual_interval, "true_residual_interval"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= value <= MAX_REFINEMENT_ITERATIONS:
            raise ValueError(f"{name} must be between 1 and 512")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable or None")
    lam = _positive_real(lambda_, "lambda_")
    tolerance = _positive_real(rtol, "rtol", below_one=True)
    squared = lam * lam
    if not math.isfinite(squared):
        raise ValueError("lambda_^2 must be finite")
    b = target.detach().clone()
    damping = torch.tensor(squared, device=b.device, dtype=b.dtype)
    damping_value = float(damping.item())
    if not math.isfinite(damping_value) or damping_value <= 0:
        raise ValueError("lambda_^2 overflows or underflows native dtype")
    target_norm = _norm64(b)
    source_shape: tuple[int, ...] | None = None
    counts = {
        "jvp_calls": 0,
        "vjp_calls": 0,
        "cg_operator_calls": 0,
        "true_residual_evaluations": 0,
        "zero_jvp_shortcuts": 0,
        "stop_callback_checks": 0,
    }

    def checked(value, name, shape=None):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must return a torch.Tensor")
        if value.dtype != b.dtype or value.device != b.device:
            raise ValueError(f"{name} must preserve native dtype/device")
        if value.numel() == 0 or (shape is not None and tuple(value.shape) != shape):
            raise ValueError(f"{name} returned an unexpected shape")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} returned nonfinite values")
        return value.detach().clone()

    def transpose(value):
        nonlocal source_shape
        counts["vjp_calls"] += 1
        out = checked(vjp(value.detach().clone()), "vjp", source_shape)
        if source_shape is None:
            source_shape = tuple(out.shape)
        return out

    def forward(value):
        if not bool(torch.count_nonzero(value).item()):
            counts["zero_jvp_shortcuts"] += 1
            return torch.zeros_like(b)
        counts["jvp_calls"] += 1
        return checked(jvp(value.detach().clone()), "jvp", tuple(b.shape))

    def dual_product(value):
        counts["cg_operator_calls"] += 1
        return checked(
            forward(transpose(value)) + damping * value, "dual operator", tuple(b.shape)
        )

    def inner(left, right):
        value = float(
            torch.dot(left.reshape(-1).double(), right.reshape(-1).double()).item()
        )
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite FP64 CG reduction")
        return value

    true_history: list[dict[str, Any]] = []
    restart_history: list[dict[str, Any]] = []
    recursive_history = [0.0 if target_norm == 0 else 1.0]
    iterations = 0
    false_convergence_count = 0
    solution = torch.zeros_like(b)
    cached = None

    def evaluate(solution, iteration, reason, recursive_relative):
        counts["true_residual_evaluations"] += 1
        dual = checked(
            (solution.double() * target_norm).to(b.dtype),
            "rescaled dual",
            tuple(b.shape),
        )
        source = transpose(dual)
        if not bool(torch.count_nonzero(dual).item()) and bool(
            torch.count_nonzero(source).item()
        ):
            raise ValueError("vjp(0) must be zero for a fixed linearized map")
        reconstruction = forward(source)
        residual64 = (
            b.double() - reconstruction.double() - damping_value * dual.double()
        )
        residual_norm = _norm64(residual64)
        relative = residual_norm / target_norm if target_norm else 0.0
        if not math.isfinite(relative):
            raise FloatingPointError("Nonfinite true relative residual")
        true_history.append(
            {
                "iteration": iteration,
                "reason": reason,
                "recursive_relative_residual": recursive_relative,
                "true_relative_residual": relative,
                "true_residual_norm": residual_norm,
            }
        )
        return {
            "dual": dual,
            "source": source,
            "reconstruction": reconstruction,
            "residual64": residual64,
            "relative": relative,
            "residual_norm": residual_norm,
        }

    stop_reason = "zero_target" if target_norm == 0 else "max_iter"
    if target_norm:
        rhs = (b.double() / target_norm).to(b.dtype)
        rhs_norm = _norm64(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        rr = inner(residual, residual)
        for _ in range(int(max_iter)):
            if stop_requested is not None:
                counts["stop_callback_checks"] += 1
                requested = stop_requested()
                if not isinstance(requested, bool):
                    raise TypeError("stop_requested must return a bool")
                if requested:
                    stop_reason = "budget_exhausted"
                    break
            product = dual_product(direction)
            curvature = inner(direction, product)
            if curvature <= 0:
                stop_reason = "nonpositive_curvature"
                break
            alpha = rr / curvature
            if not math.isfinite(alpha):
                stop_reason = "nonfinite_step"
                break
            next_solution = solution + alpha * direction
            next_residual = residual - alpha * product
            if not bool(torch.isfinite(next_solution).all().item()) or not bool(
                torch.isfinite(next_residual).all().item()
            ):
                stop_reason = "nonfinite_iterate"
                break
            next_rr = inner(next_residual, next_residual)
            solution, residual = next_solution, next_residual
            cached = None
            iterations += 1
            relative = math.sqrt(next_rr) / rhs_norm
            recursive_history.append(relative)
            candidate = relative <= tolerance
            if candidate or iterations % int(true_residual_interval) == 0:
                cached = evaluate(
                    solution,
                    iterations,
                    "recursive_candidate" if candidate else "periodic",
                    relative,
                )
                if cached["relative"] <= tolerance:
                    stop_reason = "converged"
                    break
                true_normalized = (cached["residual64"] / target_norm).to(b.dtype)
                drift = _norm64(true_normalized - residual)
                unreliable = drift > 0.1 * _norm64(true_normalized)
                if candidate or unreliable:
                    if candidate:
                        false_convergence_count += 1
                    restart_history.append(
                        {
                            "iteration": iterations,
                            "reason": "false_recursive_convergence"
                            if candidate
                            else "residual_drift",
                            "residual_drift_norm": drift,
                            "true_relative_residual": cached["relative"],
                        }
                    )
                    residual = true_normalized.detach().clone()
                    direction = residual.clone()
                    rr = inner(residual, residual)
                    continue
            beta = next_rr / rr
            if not math.isfinite(beta):
                stop_reason = "nonfinite_step"
                break
            next_direction = residual + beta * direction
            if not bool(torch.isfinite(next_direction).all().item()):
                stop_reason = "nonfinite_iterate"
                break
            direction, rr = next_direction, next_rr

    if cached is None:
        cached = evaluate(solution, iterations, "final", recursive_history[-1])
    source = cached["source"]
    dual = cached["dual"]
    reconstruction_error = _norm64(cached["reconstruction"].double() - b.double())
    reconstruction_relative = reconstruction_error / target_norm if target_norm else 0.0
    source_norm = _norm64(source)
    objective = (
        reconstruction_relative**2 + damping_value * (source_norm / target_norm) ** 2
        if target_norm
        else 0.0
    )
    if not all(math.isfinite(value) for value in (reconstruction_relative, objective)):
        raise FloatingPointError("Nonfinite final transport diagnostics")
    breakdown = (
        stop_reason
        if stop_reason
        in ("nonpositive_curvature", "nonfinite_step", "nonfinite_iterate")
        else None
    )
    converged = cached["relative"] <= tolerance and breakdown is None
    if stop_reason == "budget_exhausted":
        status = "stopped"
    elif breakdown:
        status = "breakdown"
    elif converged:
        status = "converged"
        if stop_reason != "zero_target":
            stop_reason = "converged"
    else:
        status = "not_converged"
    diagnostics = dict(
        schema="local_transport_refinement_v1",
        method="reliable_matrix_free_dual_cg",
        status=status,
        converged=bool(converged),
        stop_reason=stop_reason,
        termination=stop_reason,
        breakdown=breakdown,
        iterations=iterations,
        max_iter=int(max_iter),
        rtol=tolerance,
        true_residual_interval=int(true_residual_interval),
        true_dual_relative_residual=cached["relative"],
        dual_relative_residual=cached["relative"],
        true_dual_residual_norm=cached["residual_norm"],
        dual_residual_norm=cached["residual_norm"],
        target_norm=target_norm,
        source_norm=source_norm,
        dual_solution_norm=_norm64(dual),
        linear_reconstruction_error_norm=reconstruction_error,
        linear_reconstruction_relative_error=reconstruction_relative,
        normalized_tikhonov_objective=objective,
        recursive_relative_residual_history=recursive_history,
        true_residual_history=true_history,
        restart_history=restart_history,
        restart_count=len(restart_history),
        false_convergence_count=false_convergence_count,
        rhs_normalization="unit L2 internally; native rescaled dual/source checked against original RHS",
        final_residual_scope="FP64 reduction of b - JVP(returned source) - native_lambda_squared * returned dual",
        finalization_budget="At most one VJP and one JVP after a stop request; executing operators cannot be preempted",
        zero_target=target_norm == 0,
        target_shape=list(b.shape),
        source_shape=list(source.shape),
        dtype=str(b.dtype),
        device=str(b.device),
        full_jacobian_formed=False,
        singular_values_computed=False,
        convergence_scope="damped local linear dual equation; not undamped reconstruction or nonlinear replay",
        **counts,
    )
    diagnostics["lambda"] = lam
    diagnostics["lambda_squared_native"] = damping_value
    return TransportRefinementResult(source.detach(), dual.detach(), diagnostics)
