"""Prepare bounded QCIS measurements and delegate estimation to pinned cqlib."""

import re
from math import isfinite

from cqlib.qis import Hamiltonian, PauliString

from tianyan_mcp.expectation_models import (
    Measurement,
    ObservableTerm,
    PreparedJob,
    PreparedMeasurements,
    normalize_observable,
    parse_pauli,
)

SINGLE_QUBIT_GATES = frozenset({"I", "H", "X", "Y", "Z", "X2P", "X2M", "Y2P", "Y2M"})
ROTATIONS = frozenset({"RX", "RY", "RZ"})
QUBIT_PATTERN = re.compile(r"Q[1-9][0-9]{0,8}")
NUMBER_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def validate_preparation(circuit: str) -> str:
    if not circuit.strip() or len(circuit) > 20_000:
        raise ValueError("QCIS circuit must contain 1..20000 characters")
    lines: list[str] = []
    for line in circuit.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        gate, *args = tokens
        if gate in SINGLE_QUBIT_GATES and len(args) == 1:
            qubits = args
        elif gate == "CZ" and len(args) == 2 and args[0] != args[1]:
            qubits = args
        elif gate in ROTATIONS and len(args) == 2:
            if NUMBER_PATTERN.fullmatch(args[1]) is None or not isfinite(float(args[1])):
                raise ValueError("rotation requires a finite numeric angle in radians")
            qubits = args[:1]
        else:
            raise ValueError(
                "unsupported QCIS instruction or arity; input must have no measurements"
            )
        if any(QUBIT_PATTERN.fullmatch(qubit) is None for qubit in qubits):
            raise ValueError("invalid QCIS physical qubit ID")
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def prepare_measurements(
    circuits: list[str] | tuple[str, ...],
    observable: list[ObservableTerm] | tuple[ObservableTerm, ...],
) -> PreparedMeasurements:
    if not 1 <= len(circuits) <= 8:
        raise ValueError("requires 1..8 input circuits")
    terms = normalize_observable(observable)
    bases = [term.pauli for term in terms if term.pauli != "I"]
    if len(circuits) * len(bases) > 16:
        raise ValueError("expanded measurements exceed 16 jobs")
    preparations = [validate_preparation(circuit) for circuit in circuits]
    jobs: list[PreparedJob] = []
    for index, circuit in enumerate(preparations):
        for basis in bases:
            axes = parse_pauli(basis)
            suffix: list[str] = []
            for qubit, axis in axes.items():
                if axis == "X":
                    suffix.append(f"H Q{qubit}")
                elif axis == "Y":
                    suffix.append(f"RX Q{qubit} 1.5707963267948966")
            suffix.extend(f"M Q{qubit}" for qubit in axes)
            jobs.append(
                PreparedJob(
                    circuit_index=index,
                    pauli=basis,
                    qubits=tuple(axes),
                    circuit="\n".join([circuit, *suffix]),
                )
            )
    qubits = tuple(sorted({qubit for term in terms for qubit in parse_pauli(term.pauli)}))
    # Build the actual SDK object before the caller may submit the first cloud job.
    build_hamiltonian(terms, qubits)
    return PreparedMeasurements(observable=terms, qubits=qubits, jobs=tuple(jobs))


def sdk_pauli(pauli: str, qubits: tuple[int, ...]) -> PauliString:
    axes = parse_pauli(pauli)
    # Rightmost character is SDK bit 0. Use compact indices, not physical IDs.
    return PauliString.from_str("".join(axes.get(qubit, "I") for qubit in reversed(qubits)) or "I")


def build_hamiltonian(terms: tuple[ObservableTerm, ...], qubits: tuple[int, ...]) -> Hamiltonian:
    return Hamiltonian.from_list(
        [(sdk_pauli(term.pauli, qubits), term.coefficient) for term in terms]
    )


def map_probabilities(measurement: Measurement, qubits: tuple[int, ...]) -> dict[str, float]:
    header = measurement.header
    expected = tuple(parse_pauli(measurement.pauli))
    if (
        measurement.qubits != expected
        or not header
        or len(set(header)) != len(header)
        or set(header) != set(expected)
    ):
        raise ValueError("measurement header does not match physical qubits")
    if any(type(qubit) is not int for qubit in header):
        raise ValueError("measurement header requires integer physical IDs")
    probabilities = measurement.probability
    if not probabilities:
        raise ValueError("empty measurement probabilities")
    mapped: dict[str, float] = {}
    for key, value in probabilities.items():
        if len(key) != len(header) or re.fullmatch("[01]+", key) is None:
            raise ValueError("invalid probability key or bit width")
        if type(value) not in {int, float} or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid probability value")
        # Tianyan resultStatus column 0 corresponds to the least-significant bit.
        physical_bits = dict(zip(header, reversed(key), strict=True))
        mapped["".join(physical_bits.get(qubit, "0") for qubit in reversed(qubits))] = value
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("measurement probabilities are not normalized")
    return mapped


def calculate_expectation(
    observable: list[ObservableTerm] | tuple[ObservableTerm, ...], results: list[Measurement]
) -> float:
    terms = normalize_observable(observable)
    expected = {term.pauli for term in terms if term.pauli != "I"}
    if len(results) != len(expected) or {result.pauli for result in results} != expected:
        raise ValueError("missing, duplicate or unexpected measurement basis")
    qubits = tuple(sorted({qubit for term in terms for qubit in parse_pauli(term.pauli)}))
    # SDK picks the first compatible basis. Smaller supports must come first:
    # a smaller basis cannot cover a larger term; equal-size different bases
    # cannot be compatible. Each term therefore uses its own sampled basis.
    ordered = sorted(results, key=lambda result: len(result.qubits))
    measurements = [
        (sdk_pauli(result.pauli, qubits), map_probabilities(result, qubits)) for result in ordered
    ]
    value = build_hamiltonian(terms, qubits).expectation_probs(measurements)
    if not isfinite(value):
        raise ValueError("SDK returned a non-finite expectation")
    return value
