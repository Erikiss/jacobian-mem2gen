"""CPU-only numerical references for bounded reliable local transport."""

import json

import pytest
import torch

from attention_probe.transport_refinement import refine_transport


def operators(matrix):
    return lambda value: matrix @ value, lambda value: matrix.T @ value


def dense_reference(matrix, target, damping):
    dual = torch.linalg.solve(
        matrix @ matrix.T + damping**2 * torch.eye(matrix.shape[0], dtype=matrix.dtype),
        target,
    )
    return matrix.T @ dual, dual


def check_true_residual(matrix, target, result):
    d = result.diagnostics
    residual = (
        target.double()
        - (matrix @ result.delta_source).double()
        - d["lambda_squared_native"] * result.dual_solution.double()
    )
    expected = float(residual.norm() / target.double().norm())
    assert d["true_dual_relative_residual"] == pytest.approx(expected, rel=1e-12)
    assert d["dual_relative_residual"] == d["true_dual_relative_residual"]
    assert d["converged"] == (expected <= d["rtol"])


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_rectangular_fp64_matches_dense_ridge_and_returned_dual(shape):
    gen = torch.Generator().manual_seed(813)
    matrix = torch.randn(shape, dtype=torch.float64, generator=gen)
    target = torch.randn(shape[0], dtype=torch.float64, generator=gen)
    saved = target.clone()
    result = refine_transport(*operators(matrix), target, 0.4, rtol=1e-11)
    source, dual = dense_reference(matrix, target, 0.4)
    torch.testing.assert_close(result.delta_source, source, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(result.dual_solution, dual, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(result.delta_source, matrix.T @ result.dual_solution)
    assert result.diagnostics["stop_reason"] == "converged"
    assert (
        not result.delta_source.requires_grad and not result.dual_solution.requires_grad
    )
    assert torch.equal(target, saved)
    check_true_residual(matrix, target, result)
    json.dumps(result.diagnostics, allow_nan=False)


def test_more_than_16_iterations_converges_on_known_ill_conditioned_problem():
    matrix = torch.diag(torch.logspace(-1, 1, 64, dtype=torch.float64))
    target = torch.ones(64, dtype=torch.float64)
    early = refine_transport(*operators(matrix), target, 0.03, max_iter=16, rtol=1e-8)
    refined = refine_transport(
        *operators(matrix), target, 0.03, max_iter=512, rtol=1e-8
    )
    expected, _ = dense_reference(matrix, target, 0.03)
    assert not early.diagnostics["converged"]
    assert early.diagnostics["stop_reason"] == "max_iter"
    assert 16 < refined.diagnostics["iterations"] <= 512
    assert refined.diagnostics["converged"]
    torch.testing.assert_close(refined.delta_source, expected, rtol=2e-7, atol=2e-8)
    check_true_residual(matrix, target, refined)
    assert refined.diagnostics["lambda"] == early.diagnostics["lambda"] == 0.03


def test_real_fp32_false_recursive_convergence_restarts_and_never_silently_passes():
    # Fixed linear/adjoint callbacks, not a stateful fake operator. The requested
    # tolerance lies below the native rescaling/rounding floor for this problem.
    matrix = torch.diag(torch.tensor([1.0, 10.0], dtype=torch.float32))
    target = torch.ones(2, dtype=torch.float32)
    result = refine_transport(*operators(matrix), target, 0.1, max_iter=12, rtol=1e-10)
    d = result.diagnostics
    assert min(d["recursive_relative_residual_history"]) < d["rtol"]
    assert d["false_convergence_count"] > 0 and d["restart_count"] > 0
    assert not d["converged"] and d["stop_reason"] == "max_iter"
    assert d["iterations"] == 12 and torch.isfinite(result.delta_source).all()
    check_true_residual(matrix, target, result)
    assert any(
        row["reason"] == "false_recursive_convergence" for row in d["restart_history"]
    )
    exact_dtype = refine_transport(
        *operators(matrix.double()), target.double(), 0.1, rtol=1e-10
    )
    assert exact_dtype.diagnostics["converged"]


@pytest.mark.parametrize("scale", [1.0, 1e-30])
def test_native_fp32_original_rhs_and_rounding_are_used_in_final_residual(scale):
    matrix = torch.tensor([[1.7, -0.2], [0.6, 2.3], [0.1, 0.8]], dtype=torch.float32)
    target = scale * torch.tensor([0.3, -0.7, 0.8], dtype=torch.float32)
    result = refine_transport(*operators(matrix), target, 0.3)
    expected, _ = dense_reference(matrix.double(), target.double(), 0.3)
    torch.testing.assert_close(
        result.delta_source.double(), expected, rtol=3e-5, atol=scale * 1e-6
    )
    assert result.delta_source.dtype == result.dual_solution.dtype == torch.float32
    assert result.diagnostics["converged"] and not result.diagnostics["zero_target"]
    check_true_residual(matrix, target, result)


def test_converged_ridge_may_intentionally_leave_large_target_mismatch():
    result = refine_transport(lambda v: v, lambda v: v, torch.tensor([1.0, -2.0]), 2.0)
    assert result.diagnostics["converged"]
    assert result.diagnostics["linear_reconstruction_relative_error"] == pytest.approx(
        0.8, abs=1e-7
    )
    torch.testing.assert_close(result.delta_source, torch.tensor([0.2, -0.4]))


def test_callback_deadline_stops_bounded_work_and_still_reports_actual_final_residual():
    matrix = torch.diag(torch.linspace(1, 20, 10, dtype=torch.float64))
    target = torch.ones(10, dtype=torch.float64)
    ticks = {"now": 0, "jvp": 0, "vjp": 0}

    def jvp(value):
        ticks["now"] += 1
        ticks["jvp"] += 1
        return matrix @ value

    def vjp(value):
        ticks["now"] += 1
        ticks["vjp"] += 1
        return matrix.T @ value

    result = refine_transport(
        jvp, vjp, target, 0.1, rtol=1e-10, stop_requested=lambda: ticks["now"] >= 5
    )
    d = result.diagnostics
    assert d["stop_reason"] == "budget_exhausted" and d["status"] == "stopped"
    assert d["iterations"] == 3 and d["cg_operator_calls"] == 3
    assert ticks["now"] == 8  # Three CG products + the disclosed final VJP/JVP.
    assert d["jvp_calls"] == ticks["jvp"] and d["vjp_calls"] == ticks["vjp"]
    assert not d["converged"]
    check_true_residual(matrix, target, result)


def test_immediate_stop_does_not_start_cg_or_publish_unmeasured_success():
    result = refine_transport(
        lambda x: x, lambda y: y, torch.ones(4), 0.1, stop_requested=lambda: True
    )
    d = result.diagnostics
    assert d["iterations"] == d["cg_operator_calls"] == d["jvp_calls"] == 0
    assert d["vjp_calls"] == 1 and d["true_dual_relative_residual"] == 1
    assert d["stop_reason"] == "budget_exhausted" and not d["converged"]
    assert torch.count_nonzero(result.delta_source) == 0


def test_zero_target_and_zero_jacobian_have_honest_distinct_interpretations():
    def forbidden(_):
        pytest.fail("JVP(0) should be skipped")

    zero = refine_transport(forbidden, lambda y: torch.zeros(3), torch.zeros(4), 0.1)
    assert zero.delta_source.shape == (3,) and zero.dual_solution.shape == (4,)
    assert (
        zero.diagnostics["converged"]
        and zero.diagnostics["stop_reason"] == "zero_target"
    )
    assert zero.diagnostics["iterations"] == zero.diagnostics["jvp_calls"] == 0
    unreachable = refine_transport(
        forbidden, lambda y: torch.zeros(3), torch.ones(4), 0.1
    )
    assert unreachable.diagnostics["converged"]
    assert unreachable.diagnostics["linear_reconstruction_relative_error"] == 1
    json.dumps(unreachable.diagnostics, allow_nan=False)


def test_finite_nonpositive_curvature_is_breakdown_not_convergence():
    result = refine_transport(lambda v: v, lambda v: -v, torch.ones(2), 0.1)
    assert result.diagnostics["status"] == "breakdown"
    assert result.diagnostics["stop_reason"] == "nonpositive_curvature"
    assert not result.diagnostics["converged"] and result.diagnostics["iterations"] == 0
    assert torch.isfinite(result.delta_source).all()


def test_no_dense_factorization_is_used_and_mutating_callbacks_cannot_corrupt_rhs(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("Matrix-free production solve must not use dense matrix routines")

    for name in ["solve", "inv", "pinv", "svd", "svdvals", "eigh", "eig"]:
        monkeypatch.setattr(torch.linalg, name, forbidden)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    saved = target.clone()
    result = refine_transport(
        lambda v: v.mul_(2), lambda v: v.mul_(2), target, 0.5, rtol=1e-12
    )
    torch.testing.assert_close(result.delta_source, 2 * saved / 4.25)
    assert torch.equal(target, saved)


@pytest.mark.parametrize("value", [0, 513, -1, True, 1.5])
def test_maximum_iteration_budget_is_strict(value):
    with pytest.raises((ValueError, TypeError)):
        refine_transport(lambda v: v, lambda v: v, torch.ones(2), 0.1, max_iter=value)


@pytest.mark.parametrize(
    "kind",
    ["nan_target", "nan_operator", "wrong_dtype", "changing_shape", "invalid_stop"],
)
def test_invalid_data_and_callback_contracts_raise_instead_of_returning_success(kind):
    target = torch.ones(2)
    jvp, vjp = lambda v: v, lambda v: v
    options = {}
    if kind == "nan_target":
        target[0] = float("nan")
    elif kind == "nan_operator":
        jvp = lambda v: v * float("nan")
    elif kind == "wrong_dtype":
        vjp = lambda v: v.double()
    elif kind == "changing_shape":
        jvp = lambda v: torch.ones(3)
    elif kind == "invalid_stop":
        options["stop_requested"] = lambda: None
    with pytest.raises((ValueError, TypeError)):
        refine_transport(jvp, vjp, target, 0.1, **options)


@pytest.mark.parametrize("lam", [0, -1, float("nan"), float("inf"), 1e-30, 1e30])
def test_nonpositive_nonfinite_or_unrepresentable_damping_rejected(lam):
    with pytest.raises(ValueError):
        refine_transport(lambda v: v, lambda v: v, torch.ones(2), lam)
