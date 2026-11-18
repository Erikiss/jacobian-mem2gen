"""Local transport tests: dense small references and nonlinear functions, no LM."""

from __future__ import annotations

import json

import pytest
import torch

from attention_probe.local_transport import ridge_transport


def reference(matrix, target, lam):
    return matrix.T @ torch.linalg.solve(
        matrix @ matrix.T + lam**2 * torch.eye(matrix.shape[0], dtype=matrix.dtype),
        target,
    )


def operators(matrix):
    return lambda v: matrix @ v, lambda u: matrix.T @ u


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_matches_dense_fp64_tikhonov_closed_form(shape):
    generator = torch.Generator().manual_seed(718)
    matrix = torch.randn(shape, dtype=torch.float64, generator=generator)
    target = torch.randn(
        shape[0], dtype=torch.float64, generator=generator
    ).requires_grad_(True)
    untouched = target.detach().clone()
    result = ridge_transport(*operators(matrix), target, 0.4, rtol=1e-12)
    expected = reference(matrix, target.detach(), 0.4)
    torch.testing.assert_close(result.delta_source, expected, atol=2e-12, rtol=2e-12)
    stationary = (
        matrix.T @ (matrix @ result.delta_source - target.detach())
        + 0.4**2 * result.delta_source
    )
    assert float(stationary.norm()) < 1e-11
    d = result.diagnostics
    assert d["converged"] and d["dual_relative_residual"] < 1e-12
    assert d["iterations"] <= 16 and d["cg_operator_calls"] <= 16
    assert d["vjp_calls"] == d["cg_operator_calls"] + 1
    assert d["jvp_calls"] <= d["cg_operator_calls"] + 1
    assert len(d["recursive_relative_residual_history"]) == d["iterations"] + 1
    assert not d["full_jacobian_formed"] and not d["singular_values_computed"]
    assert not result.delta_source.requires_grad
    assert torch.equal(target.detach(), untouched)
    json.dumps(d, allow_nan=False)


def test_lambda_is_squared_and_regularized_reconstruction_is_not_solver_failure():
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)
    result = ridge_transport(
        lambda x: x, lambda x: x, target, 2.0, max_iter=12, rtol=1e-12
    )
    torch.testing.assert_close(result.delta_source, target / 5, atol=1e-15, rtol=1e-15)
    assert result.diagnostics["lambda_squared_native"] == 4
    assert result.diagnostics["converged"]
    assert result.diagnostics["linear_reconstruction_relative_error"] == pytest.approx(
        0.8
    )


def test_rank_deficient_map_keeps_unreachable_component_in_reported_error():
    matrix = torch.diag(torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64))
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = ridge_transport(*operators(matrix), target, 0.5, rtol=1e-12)
    torch.testing.assert_close(
        result.delta_source, reference(matrix, target, 0.5), atol=1e-12, rtol=1e-12
    )
    assert result.diagnostics["converged"]
    assert result.diagnostics["linear_reconstruction_error_norm"] >= 2
    assert result.delta_source[1] == 0


def test_matrix_free_path_never_calls_dense_factorization(monkeypatch):
    matrix = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    expected = reference(matrix, target, 0.7)

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Production solver must use supplied JVP/VJP, never a dense inverse/factorization"
        )

    for name in ("solve", "inv", "pinv", "svd", "svdvals", "eig", "eigh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    result = ridge_transport(*operators(matrix), target, 0.7, rtol=1e-12)
    torch.testing.assert_close(result.delta_source, expected, atol=1e-12, rtol=1e-12)


def test_zero_target_has_rectangular_source_shape_without_cg_or_jvp():
    matrix = torch.ones((4, 3), dtype=torch.float32)

    def no_jvp(_value):
        pytest.fail("Zero target must not enter JVP/CG or divide by its norm")

    result = ridge_transport(no_jvp, lambda u: matrix.T @ u, torch.zeros(4), 0.1)
    assert (
        result.delta_source.shape == (3,)
        and torch.count_nonzero(result.delta_source) == 0
    )
    d = result.diagnostics
    assert (
        d["iterations"] == d["cg_operator_calls"] == d["jvp_calls"] == 0
        and d["vjp_calls"] == 1
    )
    assert (
        d["converged"]
        and d["termination"] == "zero_target"
        and d["dual_relative_residual"] == 0
    )
    json.dumps(d, allow_nan=False)


def test_zero_jacobian_skips_zero_jvp_and_distinguishes_fit_from_dual_convergence():
    def no_jvp(_value):
        pytest.fail("JVP(0) is known analytically and must be skipped")

    result = ridge_transport(
        no_jvp, lambda _u: torch.zeros(3), torch.tensor([1.0, -2.0, 3.0, 0.5]), 0.25
    )
    assert torch.count_nonzero(result.delta_source) == 0
    assert result.diagnostics["converged"]
    assert result.diagnostics["linear_reconstruction_relative_error"] == pytest.approx(
        1
    )
    assert result.diagnostics["jvp_calls"] == 0
    assert result.diagnostics["zero_jvp_shortcuts"] >= 1
    json.dumps(result.diagnostics, allow_nan=False)


def test_tiny_fp32_target_does_not_underflow_cg_norm_to_zero():
    matrix = torch.diag(torch.tensor([2.0, 3.0]))
    target = torch.tensor([1e-30, -2e-30])
    result = ridge_transport(*operators(matrix), target, 0.5)
    expected = reference(matrix.double(), target.double(), 0.5).float()
    assert torch.count_nonzero(result.delta_source) == 2
    torch.testing.assert_close(result.delta_source, expected, atol=1e-36, rtol=2e-6)
    assert (
        result.diagnostics["target_norm"] > 0 and not result.diagnostics["zero_target"]
    )
    assert result.diagnostics["converged"]
    json.dumps(result.diagnostics, allow_nan=False)


def test_iteration_limit_returns_finite_flagged_approximation_with_true_residual():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 5.0, 9.0], dtype=torch.float64))
    target = torch.ones(4, dtype=torch.float64)
    result = ridge_transport(*operators(matrix), target, 0.01, max_iter=1, rtol=1e-12)
    d = result.diagnostics
    assert (
        not d["converged"]
        and d["status"] == "not_converged"
        and d["termination"] == "iteration_limit"
    )
    assert d["iterations"] == d["cg_operator_calls"] == 1
    assert (
        d["dual_relative_residual"] > 1e-12
        and torch.isfinite(result.delta_source).all()
    )
    json.dumps(d, allow_nan=False)


def test_nonpositive_curvature_is_explicit_breakdown_not_nan_success():
    result = ridge_transport(lambda v: v, lambda u: -u, torch.ones(3), 0.1)
    d = result.diagnostics
    assert d["status"] == "breakdown" and d["breakdown"] == "nonpositive_curvature"
    assert (
        not d["converged"] and d["iterations"] == 0 and d["dual_relative_residual"] == 1
    )
    assert torch.count_nonzero(result.delta_source) == 0
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize(
    "lam", [0.0, -1.0, float("nan"), float("inf"), 1e-30, 1e30, 1e308]
)
def test_invalid_or_unrepresentable_damping_is_rejected(lam):
    with pytest.raises(ValueError):
        ridge_transport(lambda v: v, lambda u: u, torch.ones(2), lam)


@pytest.mark.parametrize("value", [0, 17, -1, True, 2.5])
def test_iteration_budget_is_strict_and_bounded(value):
    with pytest.raises((ValueError, TypeError)):
        ridge_transport(lambda v: v, lambda u: u, torch.ones(2), 0.1, max_iter=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, float("nan"), float("inf"), True])
def test_tolerance_is_explicit_finite_positive_and_below_one(value):
    with pytest.raises((ValueError, TypeError)):
        ridge_transport(lambda v: v, lambda u: u, torch.ones(2), 0.1, rtol=value)


@pytest.mark.parametrize(
    "target",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([], dtype=torch.float32),
        torch.ones(2, dtype=torch.float16),
        torch.ones(2, dtype=torch.int64),
    ],
)
def test_nonfinite_empty_or_wrong_dtype_target_is_rejected(target):
    with pytest.raises(ValueError):
        ridge_transport(lambda v: v, lambda u: u, target, 0.1)


@pytest.mark.parametrize(
    "kind", ["dtype", "target_shape", "source_shape", "nonfinite", "not_tensor"]
)
def test_callback_contract_rejects_casting_broadcasting_and_bad_values(kind):
    calls = 0

    def transpose(u):
        nonlocal calls
        calls += 1
        if kind == "source_shape" and calls > 1:
            return torch.ones(3, dtype=u.dtype)
        if kind == "nonfinite":
            return torch.full_like(u, float("nan"))
        if kind == "not_tensor":
            return [1.0, 2.0]
        return u

    def forward(v):
        if kind == "dtype":
            return v.float()
        if kind == "target_shape":
            return v[:1]
        return v

    with pytest.raises((ValueError, TypeError)):
        ridge_transport(forward, transpose, torch.ones(2, dtype=torch.float64), 0.1)


def test_callback_arguments_cannot_mutate_cg_state_or_original_target():
    matrix = torch.tensor([[2.0, 0.2], [0.0, 3.0]], dtype=torch.float64)
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)
    initial = target.clone()

    def forward(v):
        answer = matrix @ v
        v.fill_(999)
        return answer

    def transpose(u):
        answer = matrix.T @ u
        u.fill_(999)
        return answer

    result = ridge_transport(forward, transpose, target, 0.3, rtol=1e-12)
    torch.testing.assert_close(
        result.delta_source, reference(matrix, target, 0.3), atol=1e-12, rtol=1e-12
    )
    assert torch.equal(target, initial)


def test_nonlinear_local_map_matches_dense_linearization_but_endpoint_is_separate():
    matrix = torch.tensor(
        [[1.0, 0.3, 0.0], [0.2, 1.4, 0.1], [0.0, -0.2, 0.8]], dtype=torch.float64
    )
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)

    def nonlinear(x):
        return torch.tanh(matrix @ x + bias) + 0.05 * x.square()

    target = torch.tensor([0.2, -0.1, 0.15], dtype=torch.float64)

    def at(point):
        return (
            lambda v: torch.autograd.functional.jvp(nonlinear, point, v)[1],
            lambda u: torch.autograd.functional.vjp(nonlinear, point, u)[1],
        )

    point = torch.tensor([-0.2, 0.3, 0.4], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        nonlinear, point
    )  # Test-only dense ground truth.
    result = ridge_transport(*at(point), target, 0.2, rtol=1e-12)
    torch.testing.assert_close(
        result.delta_source, reference(jacobian, target, 0.2), atol=1e-12, rtol=1e-12
    )
    predicted = jacobian @ result.delta_source
    errors = []
    for step in (1e-3, 5e-4):
        actual = nonlinear(point + step * result.delta_source) - nonlinear(point)
        errors.append(float((actual - step * predicted).norm()))
    assert (
        errors[1] < errors[0] * 0.3
    )  # Quadratic remainder shrinks faster than the linear displacement.
    changed = ridge_transport(*at(point + 0.6), target, 0.2, rtol=1e-12)
    assert float((changed.delta_source - result.delta_source).norm()) > 1e-3
    assert result.diagnostics["converged"] and changed.diagnostics["converged"]
    assert "not nonlinear endpoint" in result.diagnostics["convergence_scope"]
