"""Offline contract checks against the pinned native SDK (not a fake estimator)."""

import math

import numpy as np
import pytest


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_measurement_expectation(basis: str) -> None:
    from cqlib.qis import PauliString

    pauli = PauliString.from_str(basis)
    assert pauli.expectation_probs([(pauli, {"0": 0.75, "1": 0.25})]) == 0.5


def test_hamiltonian_and_identity() -> None:
    from cqlib.qis import Hamiltonian, PauliString

    zz = PauliString.from_str("ZZ")
    x = PauliString.from_str("IX")
    identity = PauliString.from_str("II")
    h = Hamiltonian.from_list([(zz, 0.5), (x, -0.25), (identity, 0.125)])
    assert h.expectation_probs([(zz, {"00": 0.5, "11": 0.5}), (x, {"01": 1.0})]) == 0.875
    assert identity.expectation_probs([]) == 1.0


def test_pauli_bit_order() -> None:
    from cqlib.qis import PauliString

    basis = PauliString.from_str("ZZ")
    assert PauliString.from_str("IZ").expectation_probs([(basis, {"01": 1.0})]) == -1.0
    assert PauliString.from_str("ZI").expectation_probs([(basis, {"01": 1.0})]) == 1.0


@pytest.mark.parametrize("sign", [1, -1])
def test_y_rotation_sign(sign: int) -> None:
    from cqlib.qis import Hamiltonian, PauliString, Statevector

    state = Statevector(1)
    state.apply_rx(0, -sign * math.pi / 2)
    y = PauliString.from_str("Y")
    # The pinned SDK's Statevector.expectation(Y) has a sign discrepancy.
    # Use the standard matrix as an independent oracle, not that SDK method.
    matrix_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    direct = float(np.vdot(state.data, matrix_y @ state.data).real)
    assert direct == pytest.approx(sign)
    state.apply_rx(0, math.pi / 2)
    probabilities = {str(i): p for i, p in enumerate(state.probabilities())}
    assert y.expectation_probs([(y, probabilities)]) == pytest.approx(direct)
    assert Hamiltonian.from_list([(y, 1.0)]).expectation_probs(
        [(y, probabilities)]
    ) == pytest.approx(direct)
