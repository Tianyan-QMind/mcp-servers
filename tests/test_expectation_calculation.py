"""Offline preparation, measurement mapping and real-SDK calculation contracts."""

import pytest
from pydantic import ValidationError

from tianyan_mcp.expectation_calculation import (
    calculate_expectation,
    prepare_measurements,
)
from tianyan_mcp.expectation_models import Measurement, ObservableTerm


def term(pauli: str, coefficient: float = 1.0) -> ObservableTerm:
    return ObservableTerm(pauli=pauli, coefficient=coefficient)


def test_preparation_normalizes_and_appends_rotations() -> None:
    plan = prepare_measurements(["H Q9"], [term("Z9 X1"), term("Y9"), term("X1 Z9", -0.5)])
    assert plan.observable == (term("X1 Z9", 0.5), term("Y9"))
    assert plan.qubits == (1, 9)
    assert plan.jobs[0].circuit == "H Q9\nH Q1\nM Q1\nM Q9"
    assert plan.jobs[1].circuit == "H Q9\nRX Q9 1.5707963267948966\nM Q9"
    assert plan.jobs[0].qubits == (1, 9)


def test_batch_keeps_input_order_and_constant_needs_no_jobs() -> None:
    plan = prepare_measurements(["X Q1", "H Q1"], [term("Z1")])
    assert [job.circuit_index for job in plan.jobs] == [0, 1]
    constant = prepare_measurements(["I Q1"], [term("Z1"), term("Z1", -1)])
    assert constant.jobs == ()
    assert calculate_expectation(constant.observable, []) == 0.0
    assert calculate_expectation([term("I", 0.25)], []) == 0.25


@pytest.mark.parametrize("pauli", ["", "X0", "Z-1", "Z1 X1", "I Z1", "A2", "ZZ", "Z1 garbage"])
def test_reject_invalid_observables(pauli: str) -> None:
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1"], [term(pauli)])


@pytest.mark.parametrize("coefficient", [True, float("nan"), float("inf"), 1_000_001])
def test_reject_invalid_coefficients(coefficient: float) -> None:
    with pytest.raises(ValidationError):
        term("Z1", coefficient)


@pytest.mark.parametrize(
    "circuit",
    [
        "",
        "M Q1",
        "H Q1\nM Q1",
        "RESET Q1",
        "IF Q1",
        "RX Q1 pi/2",
        "RX Q1 nan",
        "CZ Q1 Q1",
        "H Q0",
        "H Q1 extra",
        "exec('x')",
    ],
)
def test_reject_nonstatic_or_invalid_qcis(circuit: str) -> None:
    with pytest.raises(ValueError):
        prepare_measurements([circuit], [term("Z1")])


def test_resource_limits_are_checked_before_submission() -> None:
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1"] * 9, [term("Z1")])
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1"] * 8, [term("X1"), term("Y1"), term("Z1")])
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1"], [term("Z1")] * 17)
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1"], [term(" ".join(f"Z{i}" for i in range(1, 34)))])
    with pytest.raises(ValueError):
        prepare_measurements(["H Q1\n" * 4000], [term("X1")])


def test_noncontinuous_qubits_and_reversed_header_are_mapped_explicitly() -> None:
    observable = [term("Z1", 2), term("Z9", 3), term("Z1 Z9", 5)]
    measurements = [
        Measurement(pauli="Z1", qubits=(1,), header=(1,), probability={"1": 1.0}),
        Measurement(pauli="Z9", qubits=(9,), header=(9,), probability={"0": 1.0}),
        Measurement(pauli="Z1 Z9", qubits=(1, 9), header=(9, 1), probability={"10": 1.0}),
    ]
    assert calculate_expectation(observable, measurements) == -4.0


@pytest.mark.parametrize(
    "probability",
    [
        {"0": 0.9},
        {"0": 1.1, "1": -0.1},
        {"x": 1.0},
        {"00": 1.0},
        {"0": float("nan")},
        {"0": True},
        {},
    ],
)
def test_invalid_probabilities_are_not_normalized(probability: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calculate_expectation(
            [term("Z1")],
            [Measurement(pauli="Z1", qubits=(1,), header=(1,), probability=probability)],
        )


@pytest.mark.parametrize("header", [(), (1, 1), (9,)])
def test_mismatched_headers_fail(header: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        calculate_expectation(
            [term("Z1")],
            [Measurement(pauli="Z1", qubits=(1,), header=header, probability={"0": 1.0})],
        )


def test_missing_or_duplicate_measurements_fail() -> None:
    measurement = Measurement(pauli="Z1", qubits=(1,), header=(1,), probability={"0": 1.0})
    with pytest.raises(ValueError):
        calculate_expectation([term("Z1")], [])
    with pytest.raises(ValueError):
        calculate_expectation([term("Z1")], [measurement, measurement])


def test_each_term_uses_its_own_measurement_even_if_a_superset_is_listed_first() -> None:
    measurements = [
        Measurement(pauli="Z1 Z9", qubits=(1, 9), header=(1, 9), probability={"00": 1.0}),
        Measurement(pauli="Z1", qubits=(1,), header=(1,), probability={"1": 1.0}),
    ]
    assert calculate_expectation([term("Z1 Z9"), term("Z1")], measurements) == 0.0
